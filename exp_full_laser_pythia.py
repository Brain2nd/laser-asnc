"""Full-LASER end-to-end PPL on Pythia (BSE + ASNC + DCR).

Pipeline:
  1. Load model.
  2. Capture calibration activations per layer (pre-GeLU, pre-Softmax,
     post-LN outputs) from WikiText-2 train split.
  3. Fit ASNC codecs per module per layer:
       GeLU: K=32 Bennett-optimal
       Softmax: K=16 (on exp(x-max) space)
       LayerNorm: K=24 uniform
  4. Replace activations with ASNC; wrap LN with ASNCLayerNorm.
  5. Patch attention forward to use ASNCSoftmax + per-token INT16 Q/K/V (DCR).
  6. Apply BSE (per-channel INT16) on all Linear weights.
  7. Run FP16 baseline + Full-LASER PPL on WikiText-2 test.
"""
from __future__ import annotations
import argparse
import gc
import json
import math
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

from asnc_modules import (
    ASNCActivation, ASNCSoftmax, ASNCLayerNorm,
    make_asnc_gelu, bse_quantize_linears, int16_per_token_quant,
)


MAX_LEN = 2048
STRIDE = 512
DEVICE = "cuda"
K_ACTIVATION = 1024
K_SOFTMAX = 256
K_LAYERNORM = 1024
USE_LN_ASNC = True
USE_SOFTMAX_ASNC = True
USE_DCR = True
USE_ACTIVATION_ASNC = True
CALIB_BATCHES = 48
CALIB_SEQLEN = 1024


@torch.no_grad()
def capture_calibration(model, input_ids):
    """Run forward passes and capture pre-GeLU, LN output, and pre-softmax
    (QK^T/sqrt(d)) samples per layer."""
    layers = model.gpt_neox.layers
    L = len(layers)
    pre_gelu = {i: [] for i in range(L)}
    ln1_in = {i: [] for i in range(L)}
    ln2_in = {i: [] for i in range(L)}
    pre_softmax = {i: [] for i in range(L)}

    # Cap per-layer captured rows to ~50k to prevent CPU RAM OOM on large models.
    # Codec fitting needs row count ≫ K; 50k rows of [hidden]-vectors suffice.
    MAX_ROWS_PER_LAYER = 50_000

    def _subsample_rows(t, store, idx):
        # Flatten to [N, H] on CPU, append up to MAX_ROWS_PER_LAYER rows.
        t2 = t.reshape(-1, t.shape[-1]).detach().float().cpu()
        existing = sum(x.shape[0] for x in store[idx]) if store[idx] else 0
        need = max(0, MAX_ROWS_PER_LAYER - existing)
        if need > 0:
            store[idx].append(t2[:need])

    hooks = []
    for i, layer in enumerate(layers):
        def _pg(idx):
            def h(m, a, kw, out): _subsample_rows(out, pre_gelu, idx)
            return h
        hooks.append(layer.mlp.dense_h_to_4h.register_forward_hook(_pg(i), with_kwargs=True))

        def _ln1(idx):
            def h(m, args, kwargs):
                x = args[0] if args else kwargs.get("hidden_states")
                _subsample_rows(x, ln1_in, idx)
            return h
        def _ln2(idx):
            def h(m, args, kwargs):
                x = args[0] if args else kwargs.get("hidden_states")
                _subsample_rows(x, ln2_in, idx)
            return h
        hooks.append(layer.input_layernorm.register_forward_pre_hook(_ln1(i), with_kwargs=True))
        hooks.append(layer.post_attention_layernorm.register_forward_pre_hook(_ln2(i), with_kwargs=True))

        # pre-softmax: capture q_key_value output, split into Q/K, compute scores
        # GPTNeoX has a combined query_key_value projection
    # For pre-softmax, instead of hooking, we'll monkey-patch softmax calls
    # in this capture run.

    # Softmax spy: replace F.softmax to capture its input
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
        attn_hooks.append(layer.attention.register_forward_pre_hook(_pre(i), with_kwargs=True))
        attn_hooks.append(layer.attention.register_forward_hook(_post(i), with_kwargs=True))

    MAX_SOFTMAX_SAMPLES_PER_LAYER = 200_000

    def softmax_spy(*args, **kwargs):
        aid = current_attn["id"]
        out = orig_softmax(*args, **kwargs)
        if aid is not None:
            existing = sum(x.numel() for x in pre_softmax[aid]) if pre_softmax[aid] else 0
            need = max(0, MAX_SOFTMAX_SAMPLES_PER_LAYER - existing)
            if need > 0:
                flat = out.detach().float().flatten()
                if flat.numel() > need:
                    # random subsample on GPU (cheap), then to CPU
                    idx = torch.randperm(flat.numel(), device=flat.device)[:need]
                    flat = flat[idx]
                pre_softmax[aid].append(flat.cpu())
        return out

    F.softmax = softmax_spy
    try:
        for b in range(CALIB_BATCHES):
            _ = model(input_ids[b:b+1], use_cache=False)
    finally:
        F.softmax = orig_softmax
        for h in hooks + attn_hooks:
            h.remove()

    # If softmax spy didn't capture (newer transformers uses sdpa),
    # fall back to computing QK^T via q_proj/k_proj hooks.
    if all(len(pre_softmax[i]) == 0 for i in range(L)):
        print("  softmax spy did not capture (sdpa path); using QK^T reconstruction", flush=True)
        q_cap = {i: [] for i in range(L)}
        k_cap = {i: [] for i in range(L)}
        hooks = []
        for i, layer in enumerate(layers):
            # GPTNeoX has query_key_value combined projection
            def _qkv(idx):
                def h(m, a, kw, out):
                    # out shape [B, T, 3*H*D]. We'll split later.
                    q_cap[idx].append(out.detach().float().cpu())
                return h
            hooks.append(layer.attention.query_key_value.register_forward_hook(
                _qkv(i), with_kwargs=True))

        for b in range(CALIB_BATCHES):
            _ = model(input_ids[b:b+1], use_cache=False)
        for h in hooks:
            h.remove()

        n_heads = model.config.num_attention_heads
        hidden = model.config.hidden_size
        d_head = hidden // n_heads
        inv_s = 1.0 / math.sqrt(d_head)
        for i in range(L):
            for qkv in q_cap[i]:
                # qkv shape [1, T, 3*H*D]
                qkv = qkv.view(1, -1, n_heads, 3 * d_head)
                q, k, _ = qkv.split(d_head, dim=-1)
                q = q.transpose(1, 2)  # [1, H, T, d]
                k = k.transpose(1, 2)
                scores = torch.matmul(q, k.transpose(-1, -2)) * inv_s
                # subsample per layer
                flat = scores.float().flatten()
                if flat.numel() > 50_000:
                    idx = torch.randperm(flat.numel())[:50_000]
                    flat = flat[idx]
                pre_softmax[i].append(flat)

    return {
        "pre_gelu": {i: torch.cat(pre_gelu[i]) for i in range(L) if pre_gelu[i]},
        "ln1_in": {i: torch.cat(ln1_in[i]) for i in range(L) if ln1_in[i]},
        "ln2_in": {i: torch.cat(ln2_in[i]) for i in range(L) if ln2_in[i]},
        "pre_softmax": {i: torch.cat(pre_softmax[i]) for i in range(L) if pre_softmax[i]},
    }


def laser_eager_attention_forward(module, query, key, value, attention_mask,
                                  scaling, dropout=0.0, head_mask=None, **kwargs):
    orig_dtype = query.dtype
    dcr_on = getattr(module, "_dcr_on", False)
    softmax_on = hasattr(module, "asnc_softmax") and module.asnc_softmax.fitted

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

    if head_mask is not None:
        attn_weights = attn_weights * head_mask.float()
    attn_weights = F.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, v_q)
    attn_output = attn_output.transpose(1, 2).contiguous().to(orig_dtype)
    return attn_output, attn_weights.to(orig_dtype)


def patch_gpt_neox_eager():
    """Monkey-patch GPTNeoX's eager_attention_forward to our LASER version."""
    from transformers.models.gpt_neox import modeling_gpt_neox as m
    import transformers.modeling_utils
    m.eager_attention_forward = laser_eager_attention_forward
    # also patch in ALL_ATTENTION_FUNCTIONS if present
    try:
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
        ALL_ATTENTION_FUNCTIONS["eager"] = laser_eager_attention_forward
    except Exception:
        pass


@torch.no_grad()
def install_asnc(model, calib):
    """Replace activations / LNs / attention with ASNC + DCR versions, fitted
    on calibration samples."""
    layers = model.gpt_neox.layers
    for i, layer in enumerate(layers):
        # 1. GeLU (optional)
        if USE_ACTIVATION_ASNC:
            asnc = make_asnc_gelu(K=K_ACTIVATION)
            if i in calib["pre_gelu"]:
                asnc.fit(calib["pre_gelu"][i])
            layer.mlp.act = asnc.to(DEVICE)

        # 2. LN wrappers — optional (controlled by USE_LN_ASNC flag)
        if USE_LN_ASNC:
            ln1_codec = ASNCLayerNorm(layer.input_layernorm, K=K_LAYERNORM)
            if i in calib["ln1_in"]:
                ln1_codec.fit(calib["ln1_in"][i])
            layer.input_layernorm = ln1_codec.to(DEVICE)

            ln2_codec = ASNCLayerNorm(layer.post_attention_layernorm, K=K_LAYERNORM)
            if i in calib["ln2_in"]:
                ln2_codec.fit(calib["ln2_in"][i])
            layer.post_attention_layernorm = ln2_codec.to(DEVICE)

        # 3. Attention: attach ASNC softmax codec + DCR flag
        if USE_SOFTMAX_ASNC:
            asnc_sm = ASNCSoftmax(K=K_SOFTMAX)
            if i in calib["pre_softmax"]:
                asnc_sm.fit(calib["pre_softmax"][i])
            layer.attention.asnc_softmax = asnc_sm.to(DEVICE)
        layer.attention._dcr_on = USE_DCR

    # final LN at model level
    if hasattr(model.gpt_neox, "final_layer_norm"):
        fl = ASNCLayerNorm(model.gpt_neox.final_layer_norm, K=K_LAYERNORM)
        # We don't have samples for final LN; skip fit (pass-through)
        model.gpt_neox.final_layer_norm = fl.to(DEVICE)

    return model


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
    parser.add_argument("--model", required=True, help="e.g. EleutherAI/pythia-70m")
    parser.add_argument("--result", required=True, help="output JSON path")
    parser.add_argument("--fp16_ppl", type=float, default=None,
                        help="skip FP16 baseline if already known (reuse prior result)")
    args = parser.parse_args()

    MODEL = args.model
    RESULT = args.result

    print(f"Loading {MODEL}", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.float16, low_cpu_mem_usage=True,
        attn_implementation="eager",
    ).to(DEVICE)
    model.eval()
    # Use our fp32-intermediate eager for both baseline and LASER (avoids fp16
    # overflow in transformers' eager). LASER path activates only when
    # install_asnc attaches asnc_softmax to each attention module.
    patch_gpt_neox_eager()
    L = len(model.gpt_neox.layers)
    print(f"  L={L}", flush=True)

    # Calibration data (WikiText-2 train)
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    text = "\n\n".join(ds["text"])
    enc_train = tok(text, return_tensors="pt").input_ids[0]
    calib_total = CALIB_BATCHES * CALIB_SEQLEN
    calib_ids = enc_train[:calib_total].view(CALIB_BATCHES, CALIB_SEQLEN).to(DEVICE)

    # Test data
    test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text_te = "\n\n".join(test["text"])
    enc_te = tok(text_te, return_tensors="pt").input_ids
    print(f"  calib: {calib_total} tokens, test: {enc_te.numel()} tokens", flush=True)

    # -------- FP16 baseline --------
    if args.fp16_ppl is not None:
        fp16_ppl = float(args.fp16_ppl)
        fp16_sec = 0.0
        print(f"[FP16] reusing supplied baseline FP16 PPL = {fp16_ppl:.4f}", flush=True)
    else:
        print("[FP16] running PPL...", flush=True)
        t0 = time.time()
        fp16_ppl = compute_ppl(model, enc_te)
        fp16_sec = time.time() - t0
        print(f"  FP16 PPL = {fp16_ppl:.4f}  ({fp16_sec:.0f}s)", flush=True)

    # -------- Capture calibration --------
    print("[Calibration] capturing activations...", flush=True)
    t0 = time.time()
    try:
        calib = capture_calibration(model, calib_ids)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"CAPTURE ERROR: {e}", flush=True)
        raise
    print(f"  captured in {time.time()-t0:.0f}s", flush=True)
    for k in calib:
        sizes = [calib[k][i].numel() for i in calib[k]]
        if sizes:
            print(f"    {k}: L={len(sizes)}, mean_samples={sum(sizes)/len(sizes):.0f}")

    # -------- Install ASNC + DCR --------
    print("[ASNC] installing codecs...", flush=True)
    t0 = time.time()
    install_asnc(model, calib)
    print(f"  installed in {time.time()-t0:.0f}s", flush=True)

    # -------- BSE weight quant --------
    print("[BSE] quantizing Linear weights (per-channel INT16)...", flush=True)
    bse_quantize_linears(model)

    # -------- Full LASER PPL --------
    print("[Full LASER] running PPL...", flush=True)
    t0 = time.time()
    laser_ppl = compute_ppl(model, enc_te)
    laser_sec = time.time() - t0
    print(f"  Full LASER PPL = {laser_ppl:.4f}  ({laser_sec:.0f}s)", flush=True)

    dPPL = laser_ppl - fp16_ppl
    print(f"  ΔPPL = {dPPL:+.4f}", flush=True)

    result = {
        "model": MODEL,
        "fp16_ppl": fp16_ppl, "fp16_sec": fp16_sec,
        "full_laser_ppl": laser_ppl, "laser_sec": laser_sec,
        "delta_ppl": dPPL,
        "K_activation": K_ACTIVATION,
        "K_softmax": K_SOFTMAX,
        "K_layernorm": K_LAYERNORM,
    }
    with open(RESULT, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved to {RESULT}", flush=True)

    del model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
