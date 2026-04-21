"""Composite weight spectral norms, M=3L per-block.
Per block, 3 composites:
  σ_A = σ(W_Q · W_K^T)        # LN→Softmax
  σ_B = σ(W_O · W_V)          # Softmax→LN
  σ_C = σ(W_fc_out · W_fc_in) # LN→SiLU→LN (SiLU treated as Lip≤1)
For LLaMA SwiGLU use W_up for gap C composite (W_down · W_up)."""
import gc
import json
import math
import os
import time

import torch
from transformers import AutoModelForCausalLM

PYTHIA_MODELS = [
    "EleutherAI/pythia-70m",
    "EleutherAI/pythia-160m",
    "EleutherAI/pythia-410m",
    "EleutherAI/pythia-1b",
    "EleutherAI/pythia-1.4b",
    "EleutherAI/pythia-2.8b",
    "EleutherAI/pythia-6.9b",
    "EleutherAI/pythia-12b",
]
LLAMA_MODEL = "NousResearch/Llama-2-7b-hf"
RESULT = "/home/dgxspark/Desktop/A2S/spectral_3perblock.json"
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
def run_pythia(name):
    print(f"\n== {name} ==", flush=True)
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        name, torch_dtype=torch.float16, low_cpu_mem_usage=True
    )
    layers = model.gpt_neox.layers
    L = len(layers)
    print(f"loaded in {time.time()-t0:.1f}s  L={L}", flush=True)

    hidden = model.config.hidden_size
    per = []
    log_sum = 0.0
    flat = []
    t0 = time.time()
    for i, layer in enumerate(layers):
        W_qkv = layer.attention.query_key_value.weight.data.float().to(DEVICE)  # [3h, h]
        W_O = layer.attention.dense.weight.data.float().to(DEVICE)              # [h, h]
        W_fc_in = layer.mlp.dense_h_to_4h.weight.data.float().to(DEVICE)        # [4h, h]
        W_fc_out = layer.mlp.dense_4h_to_h.weight.data.float().to(DEVICE)       # [h, 4h]

        # Split W_qkv into Q, K, V
        W_Q = W_qkv[:hidden]
        W_K = W_qkv[hidden:2*hidden]
        W_V = W_qkv[2*hidden:3*hidden]

        # σ_A = σ(W_Q · W_K^T)
        M_A = W_Q @ W_K.t()
        s_A = sigma(M_A)
        # σ_B = σ(W_O · W_V)
        M_B = W_O @ W_V
        s_B = sigma(M_B)
        # σ_C = σ(W_fc_out · W_fc_in)
        M_C = W_fc_out @ W_fc_in
        s_C = sigma(M_C)

        per.append({"layer": i, "sigma_A": s_A, "sigma_B": s_B, "sigma_C": s_C})
        log_sum += math.log(s_A) + math.log(s_B) + math.log(s_C)
        flat.extend([s_A, s_B, s_C])

        del W_qkv, W_O, W_fc_in, W_fc_out, W_Q, W_K, W_V, M_A, M_B, M_C
        torch.cuda.empty_cache()
    print(f"spec iter in {time.time()-t0:.1f}s", flush=True)

    M = 3 * L
    prod = math.exp(log_sum)
    gm = math.exp(log_sum / M)
    below15 = [s for s in flat if s < 1.5]
    prod_below15 = math.prod(below15) if below15 else 0.0
    prod_A = math.prod(r["sigma_A"] for r in per)
    prod_B = math.prod(r["sigma_B"] for r in per)
    prod_C = math.prod(r["sigma_C"] for r in per)
    mn = min(flat); mx = max(flat); med = sorted(flat)[M // 2]

    print(f"L={L}  M=3L={M}", flush=True)
    print(f"  Π σ (all)       = {prod:.4g}  gm={gm:.4f}  median={med:.4f}  range [{mn:.4f}, {mx:.4f}]", flush=True)
    print(f"  Π σ (σ<1.5, {len(below15)}/{M}) = {prod_below15:.4g}", flush=True)
    print(f"  components: Π σ_A={prod_A:.3g}  Π σ_B={prod_B:.3g}  Π σ_C={prod_C:.3g}", flush=True)

    summary = {
        "num_layers": L, "M": M,
        "total_product": prod, "geomean": gm,
        "median_sigma": med, "min_sigma": mn, "max_sigma": mx,
        "num_below_1_5": len(below15), "product_below_1_5": prod_below15,
        "prod_A_QKcomposite": prod_A,
        "prod_B_VOcomposite": prod_B,
        "prod_C_MLPcomposite": prod_C,
        "gm_A": prod_A ** (1.0/L), "gm_B": prod_B ** (1.0/L), "gm_C": prod_C ** (1.0/L),
        "per_layer": per,
    }

    del model
    gc.collect()
    torch.cuda.empty_cache()
    return summary


@torch.no_grad()
def run_llama(name):
    print(f"\n== {name} ==", flush=True)
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        name, torch_dtype=torch.float16, low_cpu_mem_usage=True
    )
    layers = model.model.layers
    L = len(layers)
    print(f"loaded in {time.time()-t0:.1f}s  L={L}", flush=True)

    per = []
    log_sum = 0.0
    flat = []
    t0 = time.time()
    for i, layer in enumerate(layers):
        W_Q = layer.self_attn.q_proj.weight.data.float().to(DEVICE)
        W_K = layer.self_attn.k_proj.weight.data.float().to(DEVICE)
        W_V = layer.self_attn.v_proj.weight.data.float().to(DEVICE)
        W_O = layer.self_attn.o_proj.weight.data.float().to(DEVICE)
        W_gate = layer.mlp.gate_proj.weight.data.float().to(DEVICE)
        W_up   = layer.mlp.up_proj.weight.data.float().to(DEVICE)
        W_down = layer.mlp.down_proj.weight.data.float().to(DEVICE)

        M_A = W_Q @ W_K.t()
        s_A = sigma(M_A)
        M_B = W_O @ W_V
        s_B = sigma(M_B)
        # LLaMA: MLP is SwiGLU, gap C composite uses W_down · W_up (SiLU & gate treated as Lip-bounded)
        M_C = W_down @ W_up
        s_C = sigma(M_C)

        per.append({"layer": i, "sigma_A": s_A, "sigma_B": s_B, "sigma_C": s_C})
        log_sum += math.log(s_A) + math.log(s_B) + math.log(s_C)
        flat.extend([s_A, s_B, s_C])

        del W_Q, W_K, W_V, W_O, W_gate, W_up, W_down, M_A, M_B, M_C
        torch.cuda.empty_cache()
    print(f"spec iter in {time.time()-t0:.1f}s", flush=True)

    M = 3 * L
    prod = math.exp(log_sum)
    gm = math.exp(log_sum / M)
    below15 = [s for s in flat if s < 1.5]
    prod_below15 = math.prod(below15) if below15 else 0.0
    prod_A = math.prod(r["sigma_A"] for r in per)
    prod_B = math.prod(r["sigma_B"] for r in per)
    prod_C = math.prod(r["sigma_C"] for r in per)
    mn = min(flat); mx = max(flat); med = sorted(flat)[M // 2]

    print(f"L={L}  M=3L={M}", flush=True)
    print(f"  Π σ (all)       = {prod:.4g}  gm={gm:.4f}  median={med:.4f}  range [{mn:.4f}, {mx:.4f}]", flush=True)
    print(f"  Π σ (σ<1.5, {len(below15)}/{M}) = {prod_below15:.4g}", flush=True)
    print(f"  components: Π σ_A={prod_A:.3g}  Π σ_B={prod_B:.3g}  Π σ_C={prod_C:.3g}", flush=True)

    summary = {
        "num_layers": L, "M": M,
        "total_product": prod, "geomean": gm,
        "median_sigma": med, "min_sigma": mn, "max_sigma": mx,
        "num_below_1_5": len(below15), "product_below_1_5": prod_below15,
        "prod_A_QKcomposite": prod_A,
        "prod_B_VOcomposite": prod_B,
        "prod_C_MLPcomposite": prod_C,
        "gm_A": prod_A ** (1.0/L), "gm_B": prod_B ** (1.0/L), "gm_C": prod_C ** (1.0/L),
        "per_layer": per,
    }

    del model
    gc.collect()
    torch.cuda.empty_cache()
    return summary


def main():
    results = {}
    if os.path.exists(RESULT):
        with open(RESULT) as f:
            results = json.load(f)

    for m in PYTHIA_MODELS:
        k = m.split("/")[-1]
        if k in results and "total_product" in results[k]:
            print(f"[skip] {k}", flush=True); continue
        try:
            results[k] = run_pythia(m)
            with open(RESULT, "w") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
        except Exception as e:
            import traceback; traceback.print_exc()
            results[k] = {"error": str(e)}

    k = LLAMA_MODEL.split("/")[-1]
    if k not in results or "total_product" not in results[k]:
        try:
            results[k] = run_llama(LLAMA_MODEL)
            with open(RESULT, "w") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
        except Exception as e:
            import traceback; traceback.print_exc()
            results[k] = {"error": str(e)}

    print("\n=== Composite Weight Spectral (M=3L) ===")
    print(f"{'model':<20}{'L':>4}{'M':>5}{'Π σ':>14}{'gm':>10}{'med':>10}"
          f"{'Π σ<1.5':>14}{'#<1.5':>8}"
          f"{'gm_A':>8}{'gm_B':>8}{'gm_C':>8}")
    for m in list(PYTHIA_MODELS) + [LLAMA_MODEL]:
        k = m.split("/")[-1]
        r = results.get(k, {})
        if "total_product" not in r:
            continue
        print(f"{k:<20}{r['num_layers']:>4}{r['M']:>5}{r['total_product']:>14.4g}"
              f"{r['geomean']:>10.4f}{r['median_sigma']:>10.4f}"
              f"{r['product_below_1_5']:>14.4g}{r['num_below_1_5']:>8}"
              f"{r['gm_A']:>8.3f}{r['gm_B']:>8.3f}{r['gm_C']:>8.3f}")


if __name__ == "__main__":
    main()
