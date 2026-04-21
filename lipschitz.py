"""Empirical Lipschitz per transformer block (with residual + LN).
For each block F_m, inject eps at x_m and measure ||F_m(x+eps) - F_m(x)||/||eps||.
Product across blocks = empirical error-amplification upper bound."""
import gc
import json
import math
import os
import time

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

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
RESULT = "/home/dgxspark/Desktop/A2S/lipschitz_results.json"
DEVICE = "cuda"
SEQ_LEN = 512
BATCH = 2
NUM_PERTURB = 5
REL_EPS = 1e-3  # ||eps||_F = REL_EPS * ||x||_F per block input


def get_input_ids(tokenizer):
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(ds["text"])
    enc = tokenizer(text, return_tensors="pt").input_ids[0]
    # Take BATCH contiguous chunks of SEQ_LEN
    total = BATCH * SEQ_LEN
    enc = enc[:total].view(BATCH, SEQ_LEN)
    return enc


@torch.no_grad()
def measure_block_lipschitz(model, input_ids):
    layers = model.gpt_neox.layers
    captured = {}
    hooks = []

    def make_hook(i):
        def hook(module, args, kwargs):
            captured[i] = (args, kwargs)
            return None
        return hook

    for i, layer in enumerate(layers):
        h = layer.register_forward_pre_hook(make_hook(i), with_kwargs=True)
        hooks.append(h)

    # Populate captured inputs
    _ = model(input_ids.to(DEVICE), use_cache=False)
    for h in hooks:
        h.remove()

    per_layer = []
    for idx, layer in enumerate(layers):
        args, kwargs = captured[idx]
        x = args[0]
        rest_args = args[1:]
        kwargs = dict(kwargs)
        kwargs["use_cache"] = False

        out = layer(x, *rest_args, **kwargs)
        y = out[0] if isinstance(out, tuple) else out

        ratios = []
        x_norm = x.float().norm().item()
        for _ in range(NUM_PERTURB):
            eps = torch.randn_like(x)
            scale = REL_EPS * x_norm / (eps.float().norm().item() + 1e-30)
            eps = eps * scale
            out_p = layer(x + eps, *rest_args, **kwargs)
            y_p = out_p[0] if isinstance(out_p, tuple) else out_p
            dy = (y_p - y).float()
            r = dy.norm().item() / (eps.float().norm().item() + 1e-30)
            ratios.append(r)

        sigma_m = sum(ratios) / len(ratios)
        per_layer.append({
            "layer": idx,
            "sigma_emp": sigma_m,
            "ratios": ratios,
        })
        # free captured input to save memory for big models
        captured[idx] = None

    return per_layer


@torch.no_grad()
def run_model(name, input_ids_cpu):
    print(f"\n== {name} ==", flush=True)
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        name, torch_dtype=torch.float16, low_cpu_mem_usage=True
    ).to(DEVICE)
    model.eval()
    print(f"loaded in {time.time()-t0:.1f}s", flush=True)

    t0 = time.time()
    per_layer = measure_block_lipschitz(model, input_ids_cpu)
    elapsed = time.time() - t0

    sigmas = [r["sigma_emp"] for r in per_layer]
    L = len(sigmas)
    log_sum = sum(math.log(max(s, 1e-30)) for s in sigmas)
    prod = math.exp(log_sum)
    gm = math.exp(log_sum / L)
    mn, mx = min(sigmas), max(sigmas)
    med = sorted(sigmas)[L // 2]

    print(f"L={L}  Π σ_emp = {prod:.4g}  geomean = {gm:.4f}  "
          f"range [{mn:.4f}, {mx:.4f}]  median {med:.4f}  ({elapsed:.1f}s)",
          flush=True)

    del model
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "num_layers": L,
        "total_product": prod,
        "log_total_product": log_sum,
        "geomean_per_block": gm,
        "min_sigma": mn,
        "max_sigma": mx,
        "median_sigma": med,
        "per_layer": per_layer,
    }


def load_results():
    if os.path.exists(RESULT):
        with open(RESULT) as f:
            return json.load(f)
    return {}


def save_results(r):
    with open(RESULT, "w") as f:
        json.dump(r, f, indent=2, ensure_ascii=False)


def main():
    tok = AutoTokenizer.from_pretrained(MODELS[0])
    input_ids = get_input_ids(tok)
    print(f"Input shape: {tuple(input_ids.shape)}", flush=True)

    results = load_results()
    for m in MODELS:
        key = m.split("/")[-1]
        if key in results and "total_product" in results[key]:
            print(f"[skip] {key}", flush=True)
            continue
        try:
            r = run_model(m, input_ids)
            results[key] = r
            save_results(results)
        except Exception as e:
            import traceback
            traceback.print_exc()
            results[key] = {"error": str(e)}
            save_results(results)

    print("\n=== Summary ===")
    print(f"{'model':<18}{'L':>4}{'Π σ_emp':>14}{'geomean':>12}"
          f"{'min':>10}{'median':>10}{'max':>10}")
    for m in MODELS:
        key = m.split("/")[-1]
        r = results.get(key, {})
        if "total_product" not in r:
            continue
        print(f"{key:<18}{r['num_layers']:>4}{r['total_product']:>14.4f}"
              f"{r['geomean_per_block']:>12.4f}"
              f"{r['min_sigma']:>10.4f}{r['median_sigma']:>10.4f}{r['max_sigma']:>10.4f}")


if __name__ == "__main__":
    main()
