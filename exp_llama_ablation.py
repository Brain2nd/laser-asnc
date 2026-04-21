"""LLaMA-2 7B comprehensive ablations:
- Layer-wise sensitivity: FFN only / Attn only / Embed only / LN only / Full SNN
- Progressive SNN-ization: FFN → +Act → +Attn → +Embed → Full
- Attention component: Q/K/V proj only / QK path only / V·O path only / Full attn
All use per-channel INT16 fake quantization."""
import copy
import json
import os
import time

import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

MODEL = "NousResearch/Llama-2-7b-hf"
MAX_LEN = 2048
STRIDE = 512
RESULT = "/home/dgxspark/Desktop/A2S/results_llama_ablation.json"
DEVICE = "cuda"


@torch.no_grad()
def q_linear(module: nn.Linear):
    w = module.weight.data
    orig = w.dtype
    w32 = w.float()
    max_abs = w32.abs().amax(dim=1, keepdim=True).clamp_min(1e-30)
    scale = max_abs / 32767.0
    q = torch.round(w32 / scale).clamp(-32768, 32767)
    dq = q * scale
    module.weight.data.copy_(dq.to(orig))


@torch.no_grad()
def q_modules(modules):
    for m in modules:
        if isinstance(m, nn.Linear):
            q_linear(m)


def restore_weights(model, saved):
    for name, buf in saved.items():
        mod = dict(model.named_modules())[name]
        mod.weight.data.copy_(buf)


def save_weights(model):
    return {name: m.weight.data.clone()
            for name, m in model.named_modules()
            if isinstance(m, nn.Linear)}


def ffn_linears(layer):
    return [layer.mlp.gate_proj, layer.mlp.up_proj, layer.mlp.down_proj]


def attn_linears(layer):
    return [layer.self_attn.q_proj, layer.self_attn.k_proj,
            layer.self_attn.v_proj, layer.self_attn.o_proj]


def qkv_linears(layer):
    return [layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj]


def o_linear(layer):
    return [layer.self_attn.o_proj]


@torch.no_grad()
def compute_ppl(model, input_ids, device=DEVICE):
    seq_len = input_ids.size(1)
    nlls = []
    prev_end = 0
    total_tokens = 0
    for begin in tqdm(range(0, seq_len, STRIDE), desc="ppl", leave=False):
        end = min(begin + MAX_LEN, seq_len)
        trg_len = end - prev_end
        ids = input_ids[:, begin:end].to(device)
        target_ids = ids.clone()
        target_ids[:, :-trg_len] = -100
        outputs = model(ids, labels=target_ids)
        t = (trg_len - 1 if trg_len > 1 else 1)
        nlls.append(outputs.loss * t)
        total_tokens += t
        prev_end = end
        if end == seq_len:
            break
    return torch.exp(torch.stack(nlls).sum() / total_tokens).item()


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(test["text"])
    enc = tok(text, return_tensors="pt").input_ids
    print(f"Total tokens: {enc.size(1)}", flush=True)

    print("Loading model...")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.float16, low_cpu_mem_usage=True
    ).to(DEVICE)
    model.eval()
    print(f"loaded in {time.time()-t0:.1f}s", flush=True)

    saved = save_weights(model)

    results = {}
    if os.path.exists(RESULT):
        with open(RESULT) as f:
            results = json.load(f)

    def run(name, targets_fn):
        if name in results:
            print(f"[skip] {name}: {results[name]}")
            return
        t0 = time.time()
        targets = targets_fn(model)
        q_modules(targets)
        ppl = compute_ppl(model, enc)
        elapsed = time.time() - t0
        results[name] = {"ppl": ppl, "time": elapsed, "n_modules": len(targets)}
        print(f"{name:40s} PPL={ppl:.4f}  ({elapsed:.0f}s, {len(targets)} mods)", flush=True)
        restore_weights(model, saved)
        with open(RESULT, "w") as f:
            json.dump(results, f, indent=2)

    # --- Baseline ---
    if "ANN_FP16" not in results:
        t0 = time.time()
        ppl = compute_ppl(model, enc)
        results["ANN_FP16"] = {"ppl": ppl, "time": time.time()-t0}
        print(f"{'ANN_FP16':40s} PPL={ppl:.4f}  ({time.time()-t0:.0f}s)", flush=True)
        with open(RESULT, "w") as f:
            json.dump(results, f, indent=2)

    # ==================== Layer-wise sensitivity ====================
    def all_ffn(m): return [x for l in m.model.layers for x in ffn_linears(l)]
    def all_attn(m): return [x for l in m.model.layers for x in attn_linears(l)]
    def embed_output(m): return [m.lm_head]  # embed_tokens is nn.Embedding, not nn.Linear
    def full_snn(m): return [mm for mm in m.modules() if isinstance(mm, nn.Linear)]

    run("layerwise_FFN_only", all_ffn)
    run("layerwise_Attn_only", all_attn)
    run("layerwise_Embed_Output_only", embed_output)
    # LayerNorm only: LN has no Linear layers; we quantize the gamma weights directly as placeholder
    if "layerwise_LN_only" not in results:
        t0 = time.time()
        ln_mods = [m for m in model.modules() if isinstance(m, nn.modules.normalization.LayerNorm) or 'RMSNorm' in type(m).__name__]
        # LN gamma is a Parameter (vector); quantize it to int16
        saved_ln = [(m, m.weight.data.clone()) for m in ln_mods]
        with torch.no_grad():
            for m in ln_mods:
                w = m.weight.data
                w32 = w.float()
                max_abs = w32.abs().max().clamp_min(1e-30)
                scale = max_abs / 32767.0
                q = torch.round(w32 / scale).clamp(-32768, 32767)
                m.weight.data.copy_((q * scale).to(w.dtype))
        ppl = compute_ppl(model, enc)
        # restore
        for m, b in saved_ln:
            m.weight.data.copy_(b)
        results["layerwise_LN_only"] = {"ppl": ppl, "time": time.time()-t0,
                                        "n_modules": len(ln_mods)}
        print(f"{'layerwise_LN_only':40s} PPL={ppl:.4f}  ({time.time()-t0:.0f}s)", flush=True)
        with open(RESULT, "w") as f:
            json.dump(results, f, indent=2)

    run("layerwise_Full_SNN", full_snn)

    # ==================== Progressive SNN-ization ====================
    # FFN -> +Activation -> +Attention -> +Embedding -> Full
    # For "+ Activation" we need to quantize activations (not weights).
    # Simplified: we simulate "activation quant" as INT16 rounding on activations via a hook.
    # For now: keep it as nested weight quant (FFN + LN for activation).

    def prog_ffn(m): return all_ffn(m)
    def prog_ffn_act(m):
        return all_ffn(m) + [mm for mm in m.modules()
                             if 'RMSNorm' in type(mm).__name__]
    def prog_ffn_act_attn(m): return prog_ffn_act(m) + all_attn(m)
    def prog_ffn_act_attn_emb(m): return prog_ffn_act_attn(m) + embed_output(m)
    def prog_full(m): return full_snn(m)

    run("progressive_FFN", prog_ffn)
    run("progressive_FFN_Act", prog_ffn_act)
    run("progressive_FFN_Act_Attn", prog_ffn_act_attn)
    run("progressive_FFN_Act_Attn_Emb", prog_ffn_act_attn_emb)
    run("progressive_Full", prog_full)

    # ==================== Attention component ablation ====================
    def all_qkv(m): return [x for l in m.model.layers for x in qkv_linears(l)]
    def all_o(m): return [x for l in m.model.layers for x in o_linear(l)]

    run("attn_QKV_only", all_qkv)
    run("attn_O_only", all_o)

    # For "QK dot product only" and "Attn-weighted V only":
    # In weight-quant framework, QK uses Q,K → we approximate by quantizing Q,K only
    # V·O path uses V, O → quantize V, O only
    def qk_only(m): return [x for l in m.model.layers for x in [l.self_attn.q_proj, l.self_attn.k_proj]]
    def vo_only(m): return [x for l in m.model.layers for x in [l.self_attn.v_proj, l.self_attn.o_proj]]

    run("attn_QK_only", qk_only)
    run("attn_VO_only", vo_only)
    run("attn_Full", all_attn)

    print("\n=== Summary ===")
    for k, v in results.items():
        if "ppl" in v:
            print(f"  {k:40s}  PPL={v['ppl']:.4f}")


if __name__ == "__main__":
    main()
