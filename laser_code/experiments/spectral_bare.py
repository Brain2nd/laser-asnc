import gc
import json
import math
import os
import time

import torch
from transformers import AutoModelForCausalLM

MODELS = [
    "EleutherAI/pythia-70m",
    "EleutherAI/pythia-160m",
    "EleutherAI/pythia-410m",
    "EleutherAI/pythia-1b",
    "EleutherAI/pythia-1.4b",
    "EleutherAI/pythia-2.8b",
    "EleutherAI/pythia-6.9b",
    "EleutherAI/pythia-12b",
]

RESULT = "/home/dgxspark/Desktop/A2S/spectral_results.json"
DEVICE = "cuda"


@torch.no_grad()
def power_iter_sigma(W, iters=200, tol=1e-6):
    """Largest singular value of W via power iteration (fp32)."""
    W = W.to(DEVICE, dtype=torch.float32)
    m, n = W.shape
    v = torch.randn(n, device=DEVICE, dtype=torch.float32)
    v = v / v.norm()
    sigma_prev = 0.0
    for i in range(iters):
        u = W @ v
        un = u.norm()
        if un < 1e-20:
            return 0.0
        u = u / un
        v = W.t() @ u
        sigma = v.norm().item()
        if sigma < 1e-20:
            return 0.0
        v = v / sigma
        if abs(sigma - sigma_prev) / max(sigma, 1e-20) < tol:
            break
        sigma_prev = sigma
    return sigma


@torch.no_grad()
def run_model(name):
    print(f"\n{'='*60}\n== {name}\n{'='*60}", flush=True)
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        name, torch_dtype=torch.float16, low_cpu_mem_usage=True
    )
    print(f"[{name}] loaded in {time.time()-t0:.1f}s", flush=True)

    layers = model.gpt_neox.layers
    L = len(layers)
    per_layer = []
    log_sum = 0.0  # sum of log(sigma) for numerical stability

    t0 = time.time()
    for i, layer in enumerate(layers):
        g1 = layer.input_layernorm.weight.data.float().to(DEVICE)            # [h]
        W_qkv = layer.attention.query_key_value.weight.data.float().to(DEVICE)  # [3h, h]
        W_o   = layer.attention.dense.weight.data.float().to(DEVICE)            # [h, h]
        g2 = layer.post_attention_layernorm.weight.data.float().to(DEVICE)   # [h]
        W_f1 = layer.mlp.dense_h_to_4h.weight.data.float().to(DEVICE)           # [4h, h]
        W_f2 = layer.mlp.dense_4h_to_h.weight.data.float().to(DEVICE)           # [h, 4h]

        # Absorb LN gamma into the subsequent weight (scale columns).
        W1_eff = W_qkv * g1.unsqueeze(0)
        W3_eff = W_f1 * g2.unsqueeze(0)

        s1 = power_iter_sigma(W1_eff)  # qkv · diag(g1)
        s2 = power_iter_sigma(W_o)     # W_O
        s3 = power_iter_sigma(W3_eff)  # fc_in · diag(g2)
        s4 = power_iter_sigma(W_f2)    # fc_out

        per_layer.append({"layer": i, "qkv_g1": s1, "o": s2, "fc1_g2": s3, "fc2": s4})
        log_sum += math.log(s1) + math.log(s2) + math.log(s3) + math.log(s4)

        del W_qkv, W_o, W_f1, W_f2, W1_eff, W3_eff, g1, g2
        torch.cuda.empty_cache()

    print(f"[{name}] spectral iter in {time.time()-t0:.1f}s", flush=True)

    M = 4 * L
    total_product = math.exp(log_sum)
    avg_norm = math.exp(log_sum / M)

    # per-gap stats
    all_sigmas = []
    for row in per_layer:
        all_sigmas.extend([row["qkv_g1"], row["o"], row["fc1_g2"], row["fc2"]])
    all_sigmas.sort()
    median = all_sigmas[len(all_sigmas) // 2]

    summary = {
        "num_layers": L,
        "M_gaps": M,
        "total_product": total_product,
        "log_total_product": log_sum,
        "avg_per_gap_norm_geomean": avg_norm,
        "min_sigma": min(all_sigmas),
        "max_sigma": max(all_sigmas),
        "median_sigma": median,
        "per_layer": per_layer,
    }
    print(f"[{name}] L={L}, M={M}, Π σ = {total_product:.4g}, "
          f"geomean σ = {avg_norm:.4f}, range [{min(all_sigmas):.3f}, "
          f"{max(all_sigmas):.3f}], median {median:.3f}", flush=True)

    del model
    gc.collect()
    torch.cuda.empty_cache()
    return summary


def load_results():
    if os.path.exists(RESULT):
        with open(RESULT) as f:
            return json.load(f)
    return {}


def save_results(r):
    with open(RESULT, "w") as f:
        json.dump(r, f, indent=2, ensure_ascii=False)


def main():
    results = load_results()
    for m in MODELS:
        key = m.split("/")[-1]
        if key in results and "total_product" in results[key]:
            print(f"[skip] {key}", flush=True)
            continue
        try:
            r = run_model(m)
            results[key] = r
            save_results(results)
        except Exception as e:
            import traceback
            traceback.print_exc()
            results[key] = {"error": str(e)}
            save_results(results)

    print("\n=== Summary ===")
    print(f"{'model':<20}{'L':>4}{'M':>5}{'Π σ':>14}{'geomean σ':>14}{'median σ':>12}")
    for m in MODELS:
        key = m.split("/")[-1]
        r = results.get(key, {})
        if "total_product" in r:
            print(f"{key:<20}{r['num_layers']:>4}{r['M_gaps']:>5}"
                  f"{r['total_product']:>14.4g}{r['avg_per_gap_norm_geomean']:>14.4f}"
                  f"{r['median_sigma']:>12.4f}")


if __name__ == "__main__":
    main()
