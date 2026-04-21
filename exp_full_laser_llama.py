"""Full-LASER end-to-end PPL on LLaMA-family models (BSE + ASNC + DCR).

Adapts exp_full_laser_pythia.py for:
  - LLaMA architecture (sequential residual, RMSNorm, SwiGLU)
  - Separate q/k/v/o projections
  - device_map="auto" for multi-GPU (for 70B)
"""
from __future__ import annotations
import argparse, gc, json, math, os, time
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

from asnc_modules import (
    ASNCActivation, ASNCSoftmax, ASNCLayerNorm,
    silu_fn, silu_fprime, bse_quantize_linears, int16_per_token_quant,
)

MAX_LEN = 2048
STRIDE = 512
K_ACTIVATION = 32
K_SOFTMAX = 64
K_LAYERNORM = 24
USE_LN_ASNC = True
USE_SOFTMAX_ASNC = True
USE_DCR = True
USE_ACTIVATION_ASNC = True
CALIB_BATCHES = 8
CALIB_SEQLEN = 1024


@torch.no_grad()
def capture_calibration(model, input_ids, first_device):
    """Capture pre-SiLU (gate_proj output), pre-RMSNorm inputs, and post-softmax
    samples per layer for LLaMA."""
    layers = model.model.layers
    L = len(layers)
    pre_silu = {i: [] for i in range(L)}
    ln_in = {i: [] for i in range(L)}     # input_layernorm input
    ln2_in = {i: [] for i in range(L)}    # post_attention_layernorm input
    post_softmax = {i: [] for i in range(L)}

    MAX_ROWS_PER_LAYER = 50_000

    def _subsample_rows(t, store, idx):
        t2 = t.reshape(-1, t.shape[-1]).detach().float().cpu()
        existing = sum(x.shape[0] for x in store[idx]) if store[idx] else 0
        need = max(0, MAX_ROWS_PER_LAYER - existing)
        if need > 0:
            store[idx].append(t2[:need])

    hooks = []
    for i, layer in enumerate(layers):
        def _silu(idx):
            def h(m, a, kw, out): _subsample_rows(out, pre_silu, idx)
            return h
        hooks.append(layer.mlp.gate_proj.register_forward_hook(_silu(i), with_kwargs=True))

        def _ln1(idx):
            def h(m, args, kwargs):
                x = args[0] if args else kwargs.get("hidden_states")
                _subsample_rows(x, ln_in, idx)
            return h
        hooks.append(layer.input_layernorm.register_forward_pre_hook(_ln1(i), with_kwargs=True))

        def _ln2(idx):
            def h(m, args, kwargs):
                x = args[0] if args else kwargs.get("hidden_states")
                _subsample_rows(x, ln2_in, idx)
            return h
        hooks.append(layer.post_attention_layernorm.register_forward_pre_hook(_ln2(i), with_kwargs=True))

    # Softmax spy for post-softmax capture
    orig_softmax = F.softmax
    current_attn = {"id": None}
    attn_hooks = []
    for i, layer in enumerate(layers):
        def _pre(idx):
            def h(m, a, kw):
                current_attn["id"] = idx
            return h
        def _post(idx):
            def h(m, a, kw, out):
                current_attn["id"] = None
            return h
        attn_hooks.append(layer.self_attn.register_forward_pre_hook(_pre(i), with_kwargs=True))
        attn_hooks.append(layer.self_attn.register_forward_hook(_post(i), with_kwargs=True))

    MAX_SOFTMAX_SAMPLES_PER_LAYER = 200_000

    def softmax_spy(*args, **kwargs):
        aid = current_attn["id"]
        out = orig_softmax(*args, **kwargs)
        if aid is not None:
            existing = sum(x.numel() for x in post_softmax[aid]) if post_softmax[aid] else 0
            need = max(0, MAX_SOFTMAX_SAMPLES_PER_LAYER - existing)
            if need > 0:
                flat = out.detach().float().flatten()
                if flat.numel() > need:
                    idx = torch.randperm(flat.numel(), device=flat.device)[:need]
                    flat = flat[idx]
                post_softmax[aid].append(flat.cpu())
        return out

    F.softmax = softmax_spy
    try:
        for b in range(CALIB_BATCHES):
            _ = model(input_ids[b:b+1].to(first_device), use_cache=False)
    finally:
        F.softmax = orig_softmax
        for h in hooks + attn_hooks:
            h.remove()

    # Fallback: compute Q·K^T via hooks if softmax spy missed (sdpa)
    if all(len(post_softmax[i]) == 0 for i in range(L)):
        print("  softmax spy missed; computing QK^T via q/k hooks", flush=True)
        q_cap = {i: [] for i in range(L)}
        k_cap = {i: [] for i in range(L)}
        hooks = []
        for i, layer in enumerate(layers):
            def _q(idx):
                def h(m, a, kw, out): q_cap[idx].append(out.detach().float().cpu())
                return h
            def _k(idx):
                def h(m, a, kw, out): k_cap[idx].append(out.detach().float().cpu())
                return h
            hooks.append(layer.self_attn.q_proj.register_forward_hook(_q(i), with_kwargs=True))
            hooks.append(layer.self_attn.k_proj.register_forward_hook(_k(i), with_kwargs=True))
        for b in range(CALIB_BATCHES):
            _ = model(input_ids[b:b+1].to(first_device), use_cache=False)
        for h in hooks:
            h.remove()
        n_heads = model.config.num_attention_heads
        n_kv = getattr(model.config, "num_key_value_heads", n_heads)
        hidden = model.config.hidden_size
        d_head = hidden // n_heads
        inv_s = 1.0 / math.sqrt(d_head)
        for i in range(L):
            for qv, kv in zip(q_cap[i], k_cap[i]):
                q = qv.view(1, -1, n_heads, d_head).transpose(1, 2)
                k = kv.view(1, -1, n_kv, d_head).transpose(1, 2)
                if n_kv != n_heads:
                    k = k.repeat_interleave(n_heads // n_kv, dim=1)
                scores = torch.matmul(q, k.transpose(-1, -2)) * inv_s
                w = F.softmax(scores, dim=-1)
                post_softmax[i].append(w.flatten().float())

    return {
        "pre_silu": {i: torch.cat(pre_silu[i]) for i in range(L) if pre_silu[i]},
        "ln_in":   {i: torch.cat(ln_in[i])   for i in range(L) if ln_in[i]},
        "ln2_in":  {i: torch.cat(ln2_in[i])  for i in range(L) if ln2_in[i]},
        "post_softmax": {i: torch.cat(post_softmax[i]) for i in range(L) if post_softmax[i]},
    }


def laser_eager_attention_forward(module, query, key, value, attention_mask,
                                  scaling, dropout=0.0, **kwargs):
    """LLaMA eager attention with DCR + ASNC Softmax. Supports GQA."""
    orig_dtype = query.dtype
    dcr_on = getattr(module, "_dcr_on", False)
    softmax_on = hasattr(module, "asnc_softmax") and module.asnc_softmax.fitted

    # Handle GQA: key/value may have fewer heads than query. Repeat them.
    n_rep = query.shape[1] // key.shape[1]
    if n_rep > 1:
        key = key.repeat_interleave(n_rep, dim=1)
        value = value.repeat_interleave(n_rep, dim=1)

    if dcr_on:
        q_q = int16_per_token_quant(query).float()
        k_q = int16_per_token_quant(key).float()
        v_q = int16_per_token_quant(value).float()
    else:
        q_q, k_q, v_q = query.float(), key.float(), value.float()

    attn_weights = torch.matmul(q_q, k_q.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal = attention_mask[:, :, :, : k_q.shape[-2]].float()
        attn_weights = attn_weights + causal

    if softmax_on:
        attn_weights = module.asnc_softmax(attn_weights, dim=-1)
    else:
        attn_weights = F.softmax(attn_weights, dim=-1)

    attn_weights = F.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, v_q)
    attn_output = attn_output.transpose(1, 2).contiguous().to(orig_dtype)
    return attn_output, attn_weights.to(orig_dtype)


def patch_llama_eager():
    from transformers.models.llama import modeling_llama as m
    m.eager_attention_forward = laser_eager_attention_forward
    try:
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
        ALL_ATTENTION_FUNCTIONS["eager"] = laser_eager_attention_forward
    except Exception:
        pass


def make_asnc_silu(K=32):
    return ASNCActivation(silu_fn, silu_fprime, K)


@torch.no_grad()
def install_asnc(model, calib):
    """Install ASNC codecs + DCR flags on LLaMA layers."""
    patch_llama_eager()
    layers = model.model.layers
    for i, layer in enumerate(layers):
        dev = next(layer.parameters()).device
        if USE_ACTIVATION_ASNC and i in calib["pre_silu"]:
            asnc = make_asnc_silu(K=K_ACTIVATION)
            asnc.fit(calib["pre_silu"][i])
            layer.mlp.act_fn = asnc.to(dev)
        if USE_LN_ASNC:
            if i in calib["ln_in"]:
                codec = ASNCLayerNorm(layer.input_layernorm, K=K_LAYERNORM)
                codec.fit(calib["ln_in"][i])
                layer.input_layernorm = codec.to(dev)
            if i in calib["ln2_in"]:
                codec = ASNCLayerNorm(layer.post_attention_layernorm, K=K_LAYERNORM)
                codec.fit(calib["ln2_in"][i])
                layer.post_attention_layernorm = codec.to(dev)
        if USE_SOFTMAX_ASNC and i in calib["post_softmax"]:
            asnc_sm = ASNCSoftmax(K=K_SOFTMAX)
            asnc_sm.fit(calib["post_softmax"][i])
            layer.self_attn.asnc_softmax = asnc_sm.to(dev)
        layer.self_attn._dcr_on = USE_DCR


@torch.no_grad()
def compute_ppl(model, input_ids, first_device):
    seq_len = input_ids.size(1)
    nlls = []
    prev_end = 0
    total = 0
    for begin in tqdm(range(0, seq_len, STRIDE), desc="ppl", leave=False):
        end = min(begin + MAX_LEN, seq_len)
        trg_len = end - prev_end
        ids = input_ids[:, begin:end].to(first_device)
        tgt = ids.clone(); tgt[:, :-trg_len] = -100
        out = model(ids, labels=tgt)
        t = trg_len - 1 if trg_len > 1 else 1
        nlls.append(out.loss.float() * t)
        total += t
        prev_end = end
        if end == seq_len:
            break
    return torch.exp(torch.stack(nlls).sum() / total).item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--result", required=True)
    parser.add_argument("--device_map", default="auto", help="'auto' for multi-GPU, 'cuda' for single")
    parser.add_argument("--max_memory_per_gpu", default="42GB")
    parser.add_argument("--fp16_ppl", type=float, default=None,
                        help="skip FP16 baseline if already known")
    args = parser.parse_args()

    print(f"Loading {args.model}", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model)

    kwargs = dict(torch_dtype=torch.float16, low_cpu_mem_usage=True,
                  attn_implementation="eager")
    if args.device_map == "auto":
        max_memory = {i: args.max_memory_per_gpu for i in range(torch.cuda.device_count())}
        kwargs.update(dict(device_map="auto", max_memory=max_memory))
        model = AutoModelForCausalLM.from_pretrained(args.model, **kwargs)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model, **kwargs).to(args.device_map)
    model.eval()
    L = len(model.model.layers)
    first_device = next(model.parameters()).device
    print(f"  L={L}, first_device={first_device}", flush=True)

    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    text_tr = "\n\n".join(ds["text"])
    enc_train = tok(text_tr, return_tensors="pt").input_ids[0]
    calib_total = CALIB_BATCHES * CALIB_SEQLEN
    calib_ids = enc_train[:calib_total].view(CALIB_BATCHES, CALIB_SEQLEN)

    test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text_te = "\n\n".join(test["text"])
    enc_te = tok(text_te, return_tensors="pt").input_ids
    print(f"  calib: {calib_total} tokens, test: {enc_te.numel()} tokens", flush=True)

    # Patch eager BEFORE FP16 baseline (needed for numerical safety on fp16 large models)
    patch_llama_eager()

    if args.fp16_ppl is not None:
        fp16_ppl = float(args.fp16_ppl); fp16_sec = 0.0
        print(f"[FP16] reusing supplied baseline FP16 PPL = {fp16_ppl:.4f}", flush=True)
    else:
        print("[FP16] running PPL...", flush=True)
        t0 = time.time()
        fp16_ppl = compute_ppl(model, enc_te, first_device)
        fp16_sec = time.time() - t0
        print(f"  FP16 PPL = {fp16_ppl:.4f}  ({fp16_sec:.0f}s)", flush=True)

    print("[Calibration] capturing...", flush=True)
    t0 = time.time()
    calib = capture_calibration(model, calib_ids, first_device)
    print(f"  captured in {time.time()-t0:.0f}s", flush=True)
    for k in calib:
        sizes = [calib[k][i].numel() for i in calib[k]]
        if sizes:
            print(f"    {k}: L={len(sizes)}, mean={sum(sizes)/len(sizes):.0f}", flush=True)

    print("[ASNC] installing...", flush=True)
    t0 = time.time()
    install_asnc(model, calib)
    print(f"  installed in {time.time()-t0:.0f}s", flush=True)

    print("[BSE] int16 per-channel weight quant...", flush=True)
    bse_quantize_linears(model)

    print("[Full LASER] running PPL...", flush=True)
    t0 = time.time()
    laser_ppl = compute_ppl(model, enc_te, first_device)
    laser_sec = time.time() - t0
    print(f"  Full LASER PPL = {laser_ppl:.4f}  ({laser_sec:.0f}s)", flush=True)

    dPPL = laser_ppl - fp16_ppl
    print(f"  ΔPPL = {dPPL:+.4f}", flush=True)

    result = {"model": args.model,
              "fp16_ppl": fp16_ppl, "fp16_sec": fp16_sec,
              "full_laser_ppl": laser_ppl, "laser_sec": laser_sec,
              "delta_ppl": dPPL,
              "K_activation": K_ACTIVATION, "K_softmax": K_SOFTMAX, "K_layernorm": K_LAYERNORM}
    with open(args.result, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved to {args.result}", flush=True)

    del model; gc.collect(); torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
