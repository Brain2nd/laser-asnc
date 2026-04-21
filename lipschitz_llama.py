"""Empirical Lipschitz on LLaMA-2 7B (NousResearch mirror)."""
import gc
import json
import math
import os
import time

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "NousResearch/Llama-2-7b-hf"
RESULT = "/home/dgxspark/Desktop/A2S/lipschitz_llama.json"
DEVICE = "cuda"
SEQ_LEN = 512
BATCH = 2
NUM_PERTURB = 5
REL_EPS = 1e-3


def get_input_ids(tokenizer):
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(ds["text"])
    enc = tokenizer(text, return_tensors="pt").input_ids[0]
    total = BATCH * SEQ_LEN
    return enc[:total].view(BATCH, SEQ_LEN)


@torch.no_grad()
def measure(model, input_ids):
    layers = model.model.layers
    captured = {}
    hooks = []

    def make_hook(i):
        def hook(module, args, kwargs):
            captured[i] = (args, kwargs)
            return None
        return hook

    for i, layer in enumerate(layers):
        hooks.append(layer.register_forward_pre_hook(make_hook(i), with_kwargs=True))

    _ = model(input_ids.to(DEVICE), use_cache=False)
    for h in hooks:
        h.remove()

    per_layer = []
    for idx, layer in enumerate(layers):
        args, kwargs = captured[idx]
        x = args[0]
        rest = args[1:]
        kw = dict(kwargs)
        kw["use_cache"] = False

        out = layer(x, *rest, **kw)
        y = out[0] if isinstance(out, tuple) else out

        ratios = []
        x_norm = x.float().norm().item()
        for _ in range(NUM_PERTURB):
            eps = torch.randn_like(x)
            eps = eps * (REL_EPS * x_norm / (eps.float().norm().item() + 1e-30))
            out_p = layer(x + eps, *rest, **kw)
            y_p = out_p[0] if isinstance(out_p, tuple) else out_p
            dy = (y_p - y).float()
            r = dy.norm().item() / (eps.float().norm().item() + 1e-30)
            ratios.append(r)
        sigma_m = sum(ratios) / len(ratios)
        per_layer.append({"layer": idx, "sigma_emp": sigma_m, "ratios": ratios})
        captured[idx] = None
    return per_layer


@torch.no_grad()
def main():
    print(f"== {MODEL} ==", flush=True)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.float16, low_cpu_mem_usage=True
    ).to(DEVICE)
    model.eval()
    print(f"loaded in {time.time()-t0:.1f}s  L={len(model.model.layers)}", flush=True)

    input_ids = get_input_ids(tok)
    print(f"input shape: {tuple(input_ids.shape)}", flush=True)

    t0 = time.time()
    per_layer = measure(model, input_ids)
    elapsed = time.time() - t0

    sigmas = [r["sigma_emp"] for r in per_layer]
    L = len(sigmas)
    log_sum = sum(math.log(max(s, 1e-30)) for s in sigmas)
    prod = math.exp(log_sum)
    prod_wo_l0 = math.prod(sigmas[1:])
    gm = math.exp(log_sum / L)
    mn, mx = min(sigmas), max(sigmas)
    med = sorted(sigmas)[L // 2]

    print(f"L={L}  Π σ_emp (all)      = {prod:.4g}")
    print(f"         Π σ_emp (no L0)  = {prod_wo_l0:.4g}")
    print(f"         geomean          = {gm:.4f}")
    print(f"         range [{mn:.4f}, {mx:.4f}]  median {med:.4f}  ({elapsed:.1f}s)",
          flush=True)

    top = sorted(enumerate(sigmas), key=lambda z: -z[1])[:5]
    print(f"  top-5 σ layers: {top}", flush=True)

    summary = {
        "model": MODEL,
        "num_layers": L,
        "total_product": prod,
        "product_without_layer0": prod_wo_l0,
        "geomean_per_block": gm,
        "min_sigma": mn,
        "max_sigma": mx,
        "median_sigma": med,
        "sigma_layer0": sigmas[0],
        "per_layer": per_layer,
    }
    with open(RESULT, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"saved to {RESULT}", flush=True)


if __name__ == "__main__":
    main()
