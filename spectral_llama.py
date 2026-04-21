"""Spectral norm product for LLaMA-2 7B: bare + γ-absorbed."""
import gc
import json
import math
import time

import torch
from transformers import AutoModelForCausalLM

MODEL = "NousResearch/Llama-2-7b-hf"
RESULT = "/home/dgxspark/Desktop/A2S/spectral_llama.json"
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
def main():
    print(f"== {MODEL} ==", flush=True)
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.float16, low_cpu_mem_usage=True
    )
    print(f"loaded in {time.time()-t0:.1f}s  L={len(model.model.layers)}", flush=True)

    layers = model.model.layers
    L = len(layers)
    log_raw = 0.0
    log_abs = 0.0
    per = []

    t0 = time.time()
    for i, layer in enumerate(layers):
        g1 = layer.input_layernorm.weight.data.float().to(DEVICE)            # [h]
        g2 = layer.post_attention_layernorm.weight.data.float().to(DEVICE)   # [h]
        Wq = layer.self_attn.q_proj.weight.data.float().to(DEVICE)
        Wk = layer.self_attn.k_proj.weight.data.float().to(DEVICE)
        Wv = layer.self_attn.v_proj.weight.data.float().to(DEVICE)
        Wo = layer.self_attn.o_proj.weight.data.float().to(DEVICE)
        Wg = layer.mlp.gate_proj.weight.data.float().to(DEVICE)
        Wu = layer.mlp.up_proj.weight.data.float().to(DEVICE)
        Wd = layer.mlp.down_proj.weight.data.float().to(DEVICE)

        raws = {
            "q": sigma(Wq), "k": sigma(Wk), "v": sigma(Wv), "o": sigma(Wo),
            "gate": sigma(Wg), "up": sigma(Wu), "down": sigma(Wd),
        }
        absb = {
            "q_g1":   sigma(Wq * g1.unsqueeze(0)),
            "k_g1":   sigma(Wk * g1.unsqueeze(0)),
            "v_g1":   sigma(Wv * g1.unsqueeze(0)),
            "o":      raws["o"],
            "gate_g2":sigma(Wg * g2.unsqueeze(0)),
            "up_g2":  sigma(Wu * g2.unsqueeze(0)),
            "down":   raws["down"],
        }
        per.append({"layer": i, "raw": raws, "abs": absb,
                    "gamma1_max": g1.abs().max().item(),
                    "gamma2_max": g2.abs().max().item()})
        for s in raws.values():
            log_raw += math.log(s)
        for s in absb.values():
            log_abs += math.log(s)

        del Wq, Wk, Wv, Wo, Wg, Wu, Wd, g1, g2
        torch.cuda.empty_cache()

    M = 7 * L
    prod_raw = math.exp(log_raw)
    prod_abs = math.exp(log_abs)
    gm_raw = math.exp(log_raw / M)
    gm_abs = math.exp(log_abs / M)

    gammas1 = [p["gamma1_max"] for p in per]
    gammas2 = [p["gamma2_max"] for p in per]

    summary = {
        "model": MODEL, "num_layers": L, "M_gaps": M,
        "prod_raw_noLN": prod_raw, "gm_raw_noLN": gm_raw,
        "prod_abs_withLNgamma": prod_abs, "gm_abs_withLNgamma": gm_abs,
        "gamma1_max_over_layers": max(gammas1),
        "gamma2_max_over_layers": max(gammas2),
        "gamma1_mean_of_max": sum(gammas1) / L,
        "gamma2_mean_of_max": sum(gammas2) / L,
        "per_layer": per,
    }
    print(f"L={L}  M={M}  ({time.time()-t0:.1f}s)", flush=True)
    print(f"  raw (no γ): Π σ = {prod_raw:.4g}, geomean = {gm_raw:.4f}", flush=True)
    print(f"  γ absorbed: Π σ = {prod_abs:.4g}, geomean = {gm_abs:.4f}", flush=True)
    print(f"  γ1 max over layers: {max(gammas1):.3f}  γ2 max: {max(gammas2):.3f}", flush=True)
    print(f"  γ1 mean of per-layer max: {summary['gamma1_mean_of_max']:.3f}  γ2: {summary['gamma2_mean_of_max']:.3f}", flush=True)

    with open(RESULT, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
