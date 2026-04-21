"""tab:comparison baselines on LLaMA-2 7B:
  - BSE + SiLU (standard, int16 weights only)     : 5.18 paper
  - BSE + ReLU (SiLU→ReLU substitution)           : 50.74 paper
  - BSE + Uniform (uniform K codec SiLU)          : 5.74 paper
  - BSE + ASNC (our main)                         : 5.58 paper  (==Full_SNN)
  - Rate + SiLU (weights rate-coded 16 levels)    : 103.5 paper
  - Rate + TB (rate + threshold balancing + ReLU) : 84.2 paper"""
import copy
import gc
import json
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

MODEL = "NousResearch/Llama-2-7b-hf"
RESULT = "/home/dgxspark/Desktop/A2S/results_baselines.json"
DEVICE = "cuda"
MAX_LEN = 2048
STRIDE = 512


@torch.no_grad()
def q_int16_pc(w):
    w32 = w.float()
    ma = w32.abs().amax(dim=1, keepdim=True).clamp_min(1e-30)
    scale = ma / 32767.0
    q = torch.round(w32 / scale).clamp(-32768, 32767)
    return (q * scale).to(w.dtype)


@torch.no_grad()
def q_rate(w, steps=16):
    w32 = w.float()
    lo = w32.amin(dim=1, keepdim=True)
    hi = w32.amax(dim=1, keepdim=True)
    span = (hi - lo).clamp_min(1e-30)
    p = ((w32 - lo) / span).clamp(0, 1)
    counts = torch.round(p * steps)
    return ((counts / steps) * span + lo).to(w.dtype)


@torch.no_grad()
def q_rate_tb(w, steps=16):
    """Rate coding with threshold-balancing: clip to 99% percentile (per row) to
    avoid outlier dominance of scale, then rate-quantize."""
    w32 = w.float()
    # Per-row 99% percentile range
    q_lo = torch.quantile(w32, 0.005, dim=1, keepdim=True)
    q_hi = torch.quantile(w32, 0.995, dim=1, keepdim=True)
    w_clip = w32.clamp(min=q_lo, max=q_hi)
    span = (q_hi - q_lo).clamp_min(1e-30)
    p = ((w_clip - q_lo) / span).clamp(0, 1)
    counts = torch.round(p * steps)
    return ((counts / steps) * span + q_lo).to(w.dtype)


def quantize_all_linears(model, quant_fn, skip_lm_head=False):
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear):
            if skip_lm_head and ("lm_head" in name or "embed" in name):
                continue
            m.weight.data.copy_(quant_fn(m.weight.data))


def uniform_silu_codec(K=32, xmin=-5.0, xmax=5.0):
    """Returns a callable that approximates SiLU via K uniform bins (dtype-preserving)."""
    edges = torch.linspace(xmin, xmax, K + 1, device=DEVICE, dtype=torch.float16)
    t = edges[1:-1]
    mids = (edges[:-1] + edges[1:]) / 2
    y = F.silu(mids)
    def codec(x):
        idx = torch.bucketize(x.contiguous(), t)
        return y[idx.clamp(0, K - 1)].to(x.dtype)
    return codec


def replace_silu(model, replacement):
    """Replace all SiLU/SwiGLU activation with given callable.
    LLaMA uses F.silu via LlamaMLP's act_fn which is nn.SiLU."""
    for m in model.modules():
        if type(m).__name__ == "LlamaMLP":
            # act_fn is nn.SiLU; wrap it
            class SiLUWrapper(nn.Module):
                def __init__(self, fn):
                    super().__init__()
                    self.fn = fn
                def forward(self, x):
                    return self.fn(x)
            m.act_fn = SiLUWrapper(replacement)


@torch.no_grad()
def compute_ppl(model, input_ids):
    seq_len = input_ids.size(1)
    nlls = []
    prev_end = 0
    total = 0
    for begin in tqdm(range(0, seq_len, STRIDE), desc="ppl", leave=False):
        end = min(begin + MAX_LEN, seq_len)
        trg_len = end - prev_end
        ids = input_ids[:, begin:end].to(DEVICE)
        tgt = ids.clone()
        tgt[:, :-trg_len] = -100
        out = model(ids, labels=tgt)
        t = trg_len - 1 if trg_len > 1 else 1
        nlls.append(out.loss * t)
        total += t
        prev_end = end
        if end == seq_len:
            break
    return torch.exp(torch.stack(nlls).sum() / total).item()


def run_config(name, load_fn, input_ids):
    if os.path.exists(RESULT):
        with open(RESULT) as f:
            r = json.load(f)
        if name in r:
            print(f"[skip] {name}: {r[name]}")
            return r
    else:
        r = {}
    t0 = time.time()
    model = load_fn()
    model.eval()
    ppl = compute_ppl(model, input_ids)
    dt = time.time() - t0
    print(f"{name:30s} PPL={ppl:.4f}  ({dt:.0f}s)", flush=True)
    r[name] = {"ppl": ppl, "time": dt}
    with open(RESULT, "w") as f:
        json.dump(r, f, indent=2)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return r


def load_base():
    return AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.float16, low_cpu_mem_usage=True
    ).to(DEVICE)


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(test["text"])
    enc = tok(text, return_tensors="pt").input_ids
    print(f"Tokens: {enc.size(1)}")

    # 1) BSE + SiLU (baseline int16 weights + exact SiLU)
    def cfg_bse_silu():
        m = load_base()
        quantize_all_linears(m, q_int16_pc)
        return m
    run_config("BSE_SiLU", cfg_bse_silu, enc)

    # 2) BSE + ReLU (int16 + SiLU replaced by ReLU)
    def cfg_bse_relu():
        m = load_base()
        quantize_all_linears(m, q_int16_pc)
        replace_silu(m, F.relu)
        return m
    run_config("BSE_ReLU", cfg_bse_relu, enc)

    # 3) BSE + Uniform (int16 + SiLU replaced by uniform K=32 piecewise)
    def cfg_bse_uniform():
        m = load_base()
        quantize_all_linears(m, q_int16_pc)
        codec = uniform_silu_codec(K=32, xmin=-8, xmax=8)
        replace_silu(m, codec)
        return m
    run_config("BSE_Uniform", cfg_bse_uniform, enc)

    # 4) Rate + SiLU (weight rate-coded, exact SiLU)
    def cfg_rate_silu():
        m = load_base()
        quantize_all_linears(m, q_rate)
        return m
    run_config("Rate_SiLU", cfg_rate_silu, enc)

    # 5) Rate + TB (rate with threshold-balancing clipping, skip lm_head)
    def cfg_rate_tb():
        m = load_base()
        quantize_all_linears(m, q_rate_tb, skip_lm_head=True)
        return m
    run_config("Rate_TB", cfg_rate_tb, enc)

    # Print summary
    print("\n=== Summary ===")
    with open(RESULT) as f:
        r = json.load(f)
    for k, v in r.items():
        print(f"  {k:30s} PPL={v['ppl']:.4f}")


if __name__ == "__main__":
    main()
