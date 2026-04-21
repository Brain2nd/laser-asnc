"""Diagnostic: raw spectral norms (no γ absorption), γ statistics, and
comparison to the γ-absorbed version."""
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

RESULT = "/home/dgxspark/Desktop/A2S/spectral_diag.json"
DEVICE = "cuda"


@torch.no_grad()
def sigma(W, iters=200, tol=1e-6):
    W = W.to(DEVICE, dtype=torch.float32)
    m, n = W.shape
    v = torch.randn(n, device=DEVICE, dtype=torch.float32)
    v = v / v.norm()
    prev = 0.0
    for _ in range(iters):
        u = W @ v
        un = u.norm()
        if un < 1e-20:
            return 0.0
        u = u / un
        v = W.t() @ u
        s = v.norm().item()
        if s < 1e-20:
            return 0.0
        v = v / s
        if abs(s - prev) / max(s, 1e-20) < tol:
            break
        prev = s
    return s


@torch.no_grad()
def analyze(name):
    print(f"\n== {name} ==", flush=True)
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        name, torch_dtype=torch.float16, low_cpu_mem_usage=True
    )
    layers = model.gpt_neox.layers
    L = len(layers)
    log_raw = 0.0
    log_abs = 0.0
    g1_max_list = []
    g2_max_list = []
    per = []

    for i, layer in enumerate(layers):
        g1 = layer.input_layernorm.weight.data.float().to(DEVICE)
        g2 = layer.post_attention_layernorm.weight.data.float().to(DEVICE)
        W_qkv = layer.attention.query_key_value.weight.data.float().to(DEVICE)
        W_o = layer.attention.dense.weight.data.float().to(DEVICE)
        W_f1 = layer.mlp.dense_h_to_4h.weight.data.float().to(DEVICE)
        W_f2 = layer.mlp.dense_4h_to_h.weight.data.float().to(DEVICE)

        # raw spectral norms (no γ)
        s_qkv_raw = sigma(W_qkv)
        s_o = sigma(W_o)
        s_f1_raw = sigma(W_f1)
        s_f2 = sigma(W_f2)

        # with γ absorbed
        s_qkv_abs = sigma(W_qkv * g1.unsqueeze(0))
        s_f1_abs = sigma(W_f1 * g2.unsqueeze(0))

        g1_max = g1.abs().max().item()
        g2_max = g2.abs().max().item()
        g1_mean = g1.abs().mean().item()
        g2_mean = g2.abs().mean().item()
        g1_max_list.append(g1_max)
        g2_max_list.append(g2_max)

        per.append({
            "layer": i,
            "sigma_raw": {"qkv": s_qkv_raw, "o": s_o, "fc1": s_f1_raw, "fc2": s_f2},
            "sigma_abs": {"qkv_g1": s_qkv_abs, "o": s_o, "fc1_g2": s_f1_abs, "fc2": s_f2},
            "gamma1": {"max": g1_max, "mean": g1_mean},
            "gamma2": {"max": g2_max, "mean": g2_mean},
        })

        log_raw += math.log(s_qkv_raw) + math.log(s_o) + math.log(s_f1_raw) + math.log(s_f2)
        log_abs += math.log(s_qkv_abs) + math.log(s_o) + math.log(s_f1_abs) + math.log(s_f2)

        del W_qkv, W_o, W_f1, W_f2, g1, g2
        torch.cuda.empty_cache()

    M = 4 * L
    prod_raw = math.exp(log_raw)
    prod_abs = math.exp(log_abs)
    gm_raw = math.exp(log_raw / M)
    gm_abs = math.exp(log_abs / M)

    summary = {
        "num_layers": L, "M": M,
        "prod_raw_noLN": prod_raw, "gm_raw_noLN": gm_raw,
        "prod_abs_withLNgamma": prod_abs, "gm_abs_withLNgamma": gm_abs,
        "gamma1_max_over_layers": max(g1_max_list),
        "gamma2_max_over_layers": max(g2_max_list),
        "gamma1_mean_of_max": sum(g1_max_list) / L,
        "gamma2_mean_of_max": sum(g2_max_list) / L,
        "per_layer": per,
    }
    print(f"L={L}  M={M}", flush=True)
    print(f"  raw (no γ): Π σ = {prod_raw:.4g}, geomean = {gm_raw:.4f}", flush=True)
    print(f"  γ absorbed: Π σ = {prod_abs:.4g}, geomean = {gm_abs:.4f}", flush=True)
    print(f"  γ1 max over layers: {max(g1_max_list):.3f}   γ2 max: {max(g2_max_list):.3f}", flush=True)
    print(f"  γ1 mean of per-layer max: {summary['gamma1_mean_of_max']:.3f}   γ2: {summary['gamma2_mean_of_max']:.3f}", flush=True)
    print(f"  ⏱ {time.time()-t0:.1f}s", flush=True)

    del model
    gc.collect()
    torch.cuda.empty_cache()
    return summary


def main():
    results = {}
    if os.path.exists(RESULT):
        with open(RESULT) as f:
            results = json.load(f)
    for m in MODELS:
        key = m.split("/")[-1]
        if key in results:
            print(f"[skip] {key}", flush=True)
            continue
        try:
            r = analyze(m)
            results[key] = r
            with open(RESULT, "w") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
        except Exception as e:
            import traceback; traceback.print_exc()
            results[key] = {"error": str(e)}

    print("\n=== Diag Summary ===")
    print(f"{'model':<18}{'L':>4}{'M':>5}"
          f"{'Π σ (raw)':>14}{'gm (raw)':>10}"
          f"{'Π σ (γ)':>14}{'gm (γ)':>10}"
          f"{'γ1_max':>8}{'γ2_max':>8}")
    for m in MODELS:
        key = m.split("/")[-1]
        r = results.get(key, {})
        if "prod_raw_noLN" not in r: continue
        print(f"{key:<18}{r['num_layers']:>4}{r['M']:>5}"
              f"{r['prod_raw_noLN']:>14.4g}{r['gm_raw_noLN']:>10.4f}"
              f"{r['prod_abs_withLNgamma']:>14.4g}{r['gm_abs_withLNgamma']:>10.4f}"
              f"{r['gamma1_max_over_layers']:>8.3f}{r['gamma2_max_over_layers']:>8.3f}")


if __name__ == "__main__":
    main()
