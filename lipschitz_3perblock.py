"""Empirical Lipschitz per block, decomposed into 3 sub-σ (M=3L):
  σ_attn  = ||Δ attn_out|| / ||ε||
  σ_mlp   = ||Δ mlp_out||  / ||ε||
  σ_block = ||Δ block_out||/ ||ε||  (= the M=L measurement)"""
import gc
import json
import math
import os
import time

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

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
RESULT = "/home/dgxspark/Desktop/A2S/lipschitz_3perblock.json"
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
def measure_layers(model, input_ids, attn_attr, mlp_attr, layers_attr):
    """attn_attr/mlp_attr: attribute names on each layer for attention / mlp module.
       layers_attr: list of layers, accessed via model.<...>.layers"""
    # Get layers by attribute chain
    layers = layers_attr

    # Step 1: capture block inputs + kwargs via pre-hook
    captured_in = {}

    def make_pre_hook(i):
        def hook(module, args, kwargs):
            captured_in[i] = (args, dict(kwargs))
            return None
        return hook

    pre_hooks = [l.register_forward_pre_hook(make_pre_hook(i), with_kwargs=True)
                 for i, l in enumerate(layers)]
    _ = model(input_ids.to(DEVICE), use_cache=False)
    for h in pre_hooks:
        h.remove()

    per_layer = []
    for idx, layer in enumerate(layers):
        args, kwargs = captured_in[idx]
        x = args[0]
        rest = args[1:]
        kw = dict(kwargs)
        kw["use_cache"] = False

        # Register capture hooks on attention & mlp sub-modules
        capt = {}

        def mk_attn_hook():
            def hook(module, a, k, out):
                capt["attn"] = out[0] if isinstance(out, tuple) else out
            return hook

        def mk_mlp_hook():
            def hook(module, a, k, out):
                capt["mlp"] = out[0] if isinstance(out, tuple) else out
            return hook

        attn_mod = getattr(layer, attn_attr)
        mlp_mod = getattr(layer, mlp_attr)
        h_a = attn_mod.register_forward_hook(mk_attn_hook(), with_kwargs=True)
        h_m = mlp_mod.register_forward_hook(mk_mlp_hook(), with_kwargs=True)

        # Baseline forward
        capt.clear()
        out = layer(x, *rest, **kw)
        y_base = out[0] if isinstance(out, tuple) else out
        a_base = capt["attn"].clone()
        m_base = capt["mlp"].clone()

        r_attn = []
        r_mlp = []
        r_block = []
        x_norm = x.float().norm().item()
        for _ in range(NUM_PERTURB):
            eps = torch.randn_like(x)
            eps = eps * (REL_EPS * x_norm / (eps.float().norm().item() + 1e-30))
            capt.clear()
            out_p = layer(x + eps, *rest, **kw)
            y_p = out_p[0] if isinstance(out_p, tuple) else out_p
            a_p = capt["attn"]
            m_p = capt["mlp"]
            en = eps.float().norm().item() + 1e-30
            r_attn.append((a_p - a_base).float().norm().item() / en)
            r_mlp.append((m_p - m_base).float().norm().item() / en)
            r_block.append((y_p - y_base).float().norm().item() / en)

        h_a.remove(); h_m.remove()

        per_layer.append({
            "layer": idx,
            "sigma_attn": sum(r_attn) / len(r_attn),
            "sigma_mlp": sum(r_mlp) / len(r_mlp),
            "sigma_block": sum(r_block) / len(r_block),
        })
        captured_in[idx] = None

    return per_layer


def run(name, input_ids, is_llama=False):
    print(f"\n== {name} ==", flush=True)
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        name, torch_dtype=torch.float16, low_cpu_mem_usage=True
    ).to(DEVICE)
    model.eval()
    print(f"loaded in {time.time()-t0:.1f}s", flush=True)

    if is_llama:
        layers = model.model.layers
        attn_attr, mlp_attr = "self_attn", "mlp"
    else:
        layers = model.gpt_neox.layers
        attn_attr, mlp_attr = "attention", "mlp"

    t0 = time.time()
    per = measure_layers(model, input_ids, attn_attr, mlp_attr, layers)
    elapsed = time.time() - t0

    # Build flat list of 3L sigmas (interleaved per block)
    flat = []
    for row in per:
        flat.extend([row["sigma_attn"], row["sigma_mlp"], row["sigma_block"]])
    L = len(per)
    M = 3 * L
    log_sum = sum(math.log(max(s, 1e-30)) for s in flat)
    prod = math.exp(log_sum)
    gm = math.exp(log_sum / M)
    mn = min(flat); mx = max(flat); med = sorted(flat)[M // 2]
    below15 = [s for s in flat if s < 1.5]
    prod_below15 = math.prod(below15) if below15 else 0.0

    print(f"L={L}  M=3L={M}  ({elapsed:.1f}s)", flush=True)
    print(f"  Π σ (all 3L)          = {prod:.4g}", flush=True)
    print(f"  Π σ (σ<1.5 only, {len(below15)}/{M}) = {prod_below15:.4g}", flush=True)
    print(f"  geomean = {gm:.4f}  median = {med:.4f}  range [{mn:.4f}, {mx:.4f}]", flush=True)

    # component breakdown
    prod_attn = math.prod(r["sigma_attn"] for r in per)
    prod_mlp = math.prod(r["sigma_mlp"] for r in per)
    prod_block = math.prod(r["sigma_block"] for r in per)
    gm_attn = prod_attn ** (1.0 / L)
    gm_mlp = prod_mlp ** (1.0 / L)
    gm_block = prod_block ** (1.0 / L)
    print(f"  component: Π σ_attn={prod_attn:.3g} (gm {gm_attn:.3f})  "
          f"Π σ_mlp={prod_mlp:.3g} (gm {gm_mlp:.3f})  "
          f"Π σ_block={prod_block:.3g} (gm {gm_block:.3f})", flush=True)

    summary = {
        "num_layers": L, "M": M,
        "total_product": prod, "geomean": gm,
        "median_sigma": med, "min_sigma": mn, "max_sigma": mx,
        "num_below_1_5": len(below15), "product_below_1_5": prod_below15,
        "prod_attn": prod_attn, "prod_mlp": prod_mlp, "prod_block": prod_block,
        "gm_attn": gm_attn, "gm_mlp": gm_mlp, "gm_block": gm_block,
        "per_layer": per,
    }

    del model
    gc.collect()
    torch.cuda.empty_cache()
    return summary


def main():
    # Use Pythia tokenizer for Pythia, LLaMA tokenizer for LLaMA (same-family sharing)
    tok_py = AutoTokenizer.from_pretrained(PYTHIA_MODELS[0])
    ids_py = get_input_ids(tok_py)
    tok_llama = AutoTokenizer.from_pretrained(LLAMA_MODEL)
    ids_llama = get_input_ids(tok_llama)
    print(f"pythia input shape: {tuple(ids_py.shape)}", flush=True)
    print(f"llama  input shape: {tuple(ids_llama.shape)}", flush=True)

    results = {}
    if os.path.exists(RESULT):
        with open(RESULT) as f:
            results = json.load(f)

    for m in PYTHIA_MODELS:
        k = m.split("/")[-1]
        if k in results and "total_product" in results[k]:
            print(f"[skip] {k}", flush=True); continue
        try:
            results[k] = run(m, ids_py, is_llama=False)
            with open(RESULT, "w") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
        except Exception as e:
            import traceback; traceback.print_exc()
            results[k] = {"error": str(e)}
            with open(RESULT, "w") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

    # LLaMA-2
    k = LLAMA_MODEL.split("/")[-1]
    if k not in results or "total_product" not in results[k]:
        try:
            results[k] = run(LLAMA_MODEL, ids_llama, is_llama=True)
            with open(RESULT, "w") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
        except Exception as e:
            import traceback; traceback.print_exc()
            results[k] = {"error": str(e)}

    print("\n=== Summary (M=3L) ===")
    print(f"{'model':<20}{'L':>4}{'M':>5}{'Π σ':>14}{'gm':>10}{'med':>10}{'Π σ<1.5':>14}{'#<1.5':>8}")
    for m in list(PYTHIA_MODELS) + [LLAMA_MODEL]:
        k = m.split("/")[-1]
        r = results.get(k, {})
        if "total_product" not in r: continue
        print(f"{k:<20}{r['num_layers']:>4}{r['M']:>5}{r['total_product']:>14.4g}"
              f"{r['geomean']:>10.4f}{r['median_sigma']:>10.4f}"
              f"{r['product_below_1_5']:>14.4g}{r['num_below_1_5']:>8}")


if __name__ == "__main__":
    main()
