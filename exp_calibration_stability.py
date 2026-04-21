"""Calibration parameter (λ, μ) stability + clipping coverage.
Paper (LLaMA-2 7B, K=1024 calib):
  Mean relative deviation of λ, μ across {MMLU, C4, OpenWebText} < 4.2%
  Clipping coverage (fraction outside [μ, μ+(2^N-1)/λ]):
    SiLU 0.21%, Softmax 0.09%, LN 0.34% @ 1024 prompts
    SiLU 0.43%, Softmax 0.17%, LN 0.65% @ 256 prompts
"""
import gc
import json
import os
import time

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "NousResearch/Llama-2-7b-hf"
RESULT = "/home/dgxspark/Desktop/A2S/results_calibration.json"
DEVICE = "cuda"
TARGET_L = 16    # middle layer for probe


def make_batches(tok, texts, n_samples, seq_len=512):
    """Build tokenized batches from a list of strings until we have n_samples tokens."""
    all_ids = []
    total = 0
    for text in texts:
        if not text.strip():
            continue
        ids = tok(text, return_tensors="pt").input_ids[0]
        all_ids.append(ids)
        total += ids.numel()
        if total >= n_samples * seq_len:
            break
    all_ids = torch.cat(all_ids)[: n_samples * seq_len]
    return all_ids.view(n_samples, seq_len)


def load_dataset_text(name):
    """Return iterator of text samples from a named dataset."""
    try:
        if name == "wikitext":
            ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train", streaming=False)
            return [r["text"] for r in ds]
        if name == "c4":
            ds = load_dataset("allenai/c4", "en", split="train", streaming=True)
            return [next(iter(ds))["text"] for _ in range(500)]
        if name == "openwebtext":
            ds = load_dataset("stas/openwebtext-10k", split="train", streaming=False)
            return [r["text"] for r in ds][:500]
    except Exception as e:
        print(f"Could not load {name}: {e}. Falling back.")
        return None
    return None


@torch.no_grad()
def calibrate(model, tok, texts, n_prompts):
    """Run `n_prompts` sequences through model, capture activations of:
       - SiLU input (gate_proj output) at target layer
       - Softmax input (QK^T/sqrt(d)) at target layer
       - LN input at target layer
       Return (μ, λ) per module where λ = (2^16-1)/(max_abs × 2).
       Following paper's convention: λ = grid points per unit, μ = center."""
    import math
    ids = make_batches(tok, texts, n_prompts, seq_len=512).to(DEVICE)

    # capture with subsampling (cap memory)
    MAX_PER_CALL = 5000
    silu_buf, ln_buf, sm_buf = [], [], []
    layer = model.model.layers[TARGET_L]

    def _subsample(t):
        t = t.detach().float().flatten()
        n = t.numel()
        if n > MAX_PER_CALL:
            idx = torch.randperm(n, device=t.device)[:MAX_PER_CALL]
            t = t[idx]
        return t.cpu()

    def silu_hook(mod, args, kwargs, output):
        silu_buf.append(_subsample(output))

    def ln_prehook(mod, args, kwargs):
        x = args[0] if args else kwargs.get("hidden_states")
        ln_buf.append(_subsample(x))

    h1 = layer.mlp.gate_proj.register_forward_hook(silu_hook, with_kwargs=True)
    h2 = layer.post_attention_layernorm.register_forward_pre_hook(ln_prehook, with_kwargs=True)

    # softmax via QK
    q_store, k_store = [], []
    def q_hook(m, a, kw, o): q_store.append(o.detach())
    def k_hook(m, a, kw, o): k_store.append(o.detach())
    h3 = layer.self_attn.q_proj.register_forward_hook(q_hook, with_kwargs=True)
    h4 = layer.self_attn.k_proj.register_forward_hook(k_hook, with_kwargs=True)

    n_heads = model.config.num_attention_heads
    d_head = model.config.hidden_size // n_heads
    inv_s = 1.0 / math.sqrt(d_head)

    for i in range(n_prompts):
        q_store.clear(); k_store.clear()
        _ = model(ids[i:i+1], use_cache=False)
        q = q_store[0].view(1, -1, n_heads, d_head).transpose(1, 2)
        k = k_store[0].view(1, -1, n_heads, d_head).transpose(1, 2)
        scores = (q @ k.transpose(-1, -2)) * inv_s
        sm_buf.append(_subsample(scores))

    for h in (h1, h2, h3, h4):
        h.remove()

    def make_lambda_mu(samples_list, name):
        x = torch.cat(samples_list)
        x = x[torch.isfinite(x)]
        mu = x.mean().item()
        # Paper's (λ, μ): λ = (2^N - 1) / span; μ = low clip
        sd = x.std().item()
        lo = mu - 3.0 * sd   # ±3σ calibration
        hi = mu + 3.0 * sd
        lam = (2**16 - 1) / max(hi - lo, 1e-30)
        # Clipping coverage: fraction outside [lo, hi]
        out_frac = ((x < lo) | (x > hi)).float().mean().item()
        return {"module": name, "mu": mu, "sigma": sd, "low": lo, "high": hi,
                "lambda": lam, "out_of_range_frac": out_frac, "n_samples": x.numel()}

    results = {
        "SiLU": make_lambda_mu(silu_buf, "SiLU"),
        "LN":   make_lambda_mu(ln_buf, "LN"),
        "Softmax": make_lambda_mu(sm_buf, "Softmax"),
    }
    return results


def rel_dev(a, b):
    return abs(a - b) / max(abs(a), abs(b), 1e-30)


def main():
    print(f"Loading {MODEL}")
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.float16, low_cpu_mem_usage=True
    ).to(DEVICE)
    model.eval()

    results = {}

    # Primary calibration: WikiText (as MMLU-like proxy since MMLU requires dataset build)
    datasets_to_try = [("wikitext", "wikitext"), ("openwebtext", "openwebtext"), ("c4", "c4")]

    per_set = {}
    for label, name in datasets_to_try:
        texts = load_dataset_text(name)
        if texts is None:
            print(f"  skip {label}")
            continue
        for n_prompts in (256, 1024):
            key = f"{label}_{n_prompts}"
            t0 = time.time()
            try:
                per_set[key] = calibrate(model, tok, texts, n_prompts)
                print(f"  {key}: ok ({time.time()-t0:.1f}s)")
            except Exception as e:
                print(f"  {key}: error {e}")
                import traceback; traceback.print_exc()
    results["per_set"] = per_set

    # Compute mean relative deviation of (λ, μ) across sets at n_prompts=1024
    modules = ["SiLU", "LN", "Softmax"]
    sets_1024 = [k for k in per_set if k.endswith("1024")]
    if len(sets_1024) >= 2:
        ref = sets_1024[0]
        devs = {m: {"mu_rel_dev": [], "lambda_rel_dev": []} for m in modules}
        for other in sets_1024[1:]:
            for m in modules:
                if m in per_set[ref] and m in per_set[other]:
                    devs[m]["mu_rel_dev"].append(rel_dev(per_set[ref][m]["mu"], per_set[other][m]["mu"]))
                    devs[m]["lambda_rel_dev"].append(rel_dev(per_set[ref][m]["lambda"], per_set[other][m]["lambda"]))
        results["cross_set_rel_dev"] = {
            m: {"mu_mean_rel_dev": float(sum(devs[m]["mu_rel_dev"])/max(len(devs[m]["mu_rel_dev"]),1)),
                "lambda_mean_rel_dev": float(sum(devs[m]["lambda_rel_dev"])/max(len(devs[m]["lambda_rel_dev"]),1))}
            for m in modules
        }

    # Clipping coverage summary
    coverage = {m: {} for m in modules}
    for key, d in per_set.items():
        size = int(key.split("_")[-1])
        for m in modules:
            if m in d:
                coverage[m].setdefault(size, []).append(d[m]["out_of_range_frac"])
    results["clipping_coverage"] = {
        m: {size: float(sum(v)/len(v)) for size, v in sizes.items()}
        for m, sizes in coverage.items()
    }

    with open(RESULT, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {RESULT}")
    print(json.dumps(results.get("cross_set_rel_dev", {}), indent=2))
    print(json.dumps(results.get("clipping_coverage", {}), indent=2))


if __name__ == "__main__":
    main()
