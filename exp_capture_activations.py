"""Capture LLaMA-2 7B activations for ASNC/KS/linear-fidelity analyses.
Saves:
  silu_input_L{i}.pt     (pre-SiLU = gate_proj output)
  ln2_input_L{i}.pt      (input to post_attention_layernorm)
  softmax_input_L{i}.pt  (pre-softmax logits, for i in TARGET_LAYERS)
  ffn_gate_L16.pt        (FFN gate_proj weight of layer 16)"""
import gc
import os
import time

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "NousResearch/Llama-2-7b-hf"
OUT_DIR = "/home/dgxspark/Desktop/A2S/activations"
os.makedirs(OUT_DIR, exist_ok=True)
DEVICE = "cuda"
TARGET_LAYERS = [4, 16, 28]
N_CHUNKS = 32
CHUNK_LEN = 1024
MAX_SAMPLES = 200_000


@torch.no_grad()
def main():
    print(f"Loading {MODEL}", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.float16, low_cpu_mem_usage=True
    ).to(DEVICE)
    model.eval()
    L = len(model.model.layers)
    print(f"loaded  L={L}", flush=True)

    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    text = "\n\n".join(ds["text"])
    enc = tok(text, return_tensors="pt").input_ids[0]
    ids = enc[:N_CHUNKS * CHUNK_LEN].view(N_CHUNKS, CHUNK_LEN).to(DEVICE)
    print(f"Using {N_CHUNKS}x{CHUNK_LEN} tokens", flush=True)

    # Buffers
    silu_buf = {i: [] for i in range(L)}
    ln2_buf = {i: [] for i in range(L)}

    hooks = []
    for i, layer in enumerate(model.model.layers):
        def mk_silu_hook(idx):
            def h(module, args, kwargs, output):
                silu_buf[idx].append(output.detach().to(torch.float32).flatten().cpu())
            return h
        hooks.append(
            layer.mlp.gate_proj.register_forward_hook(mk_silu_hook(i), with_kwargs=True)
        )

        def mk_ln2_prehook(idx):
            def h(module, args, kwargs):
                x = args[0] if args else kwargs.get("hidden_states")
                ln2_buf[idx].append(x.detach().to(torch.float32).flatten().cpu())
            return h
        hooks.append(
            layer.post_attention_layernorm.register_forward_pre_hook(
                mk_ln2_prehook(i), with_kwargs=True
            )
        )

    # Softmax spy
    import torch.nn.functional as F
    sm_buf = {i: [] for i in TARGET_LAYERS}
    attn_id = {id(model.model.layers[i].self_attn): i for i in TARGET_LAYERS}
    current = {"layer": None}

    attn_hooks = []
    for i in TARGET_LAYERS:
        am = model.model.layers[i].self_attn
        def mk_pre(idx):
            def h(module, args, kwargs):
                current["layer"] = idx
            return h
        def mk_post(idx):
            def h(module, args, kwargs, output):
                current["layer"] = None
            return h
        attn_hooks.append(am.register_forward_pre_hook(mk_pre(i), with_kwargs=True))
        attn_hooks.append(am.register_forward_hook(mk_post(i), with_kwargs=True))

    orig_softmax = F.softmax

    def sm_spy(*args, **kwargs):
        lid = current["layer"]
        if lid is not None and lid in TARGET_LAYERS:
            x = args[0] if args else kwargs.get("input")
            sm_buf[lid].append(x.detach().to(torch.float32).flatten().cpu())
        return orig_softmax(*args, **kwargs)

    F.softmax = sm_spy

    t0 = time.time()
    try:
        for b in range(N_CHUNKS):
            _ = model(ids[b:b+1], use_cache=False)
    finally:
        F.softmax = orig_softmax
        for h in hooks + attn_hooks:
            h.remove()
    print(f"Forward passes done in {time.time()-t0:.1f}s", flush=True)

    # Save
    for i in range(L):
        if silu_buf[i]:
            x = torch.cat(silu_buf[i])
            if len(x) > MAX_SAMPLES:
                x = x[torch.randperm(len(x))[:MAX_SAMPLES]]
            torch.save(x, os.path.join(OUT_DIR, f"silu_input_L{i}.pt"))
        if ln2_buf[i]:
            x = torch.cat(ln2_buf[i])
            if len(x) > MAX_SAMPLES:
                x = x[torch.randperm(len(x))[:MAX_SAMPLES]]
            torch.save(x, os.path.join(OUT_DIR, f"ln2_input_L{i}.pt"))
    for i in TARGET_LAYERS:
        if sm_buf[i]:
            x = torch.cat(sm_buf[i])
            if len(x) > MAX_SAMPLES:
                x = x[torch.randperm(len(x))[:MAX_SAMPLES]]
            torch.save(x, os.path.join(OUT_DIR, f"softmax_input_L{i}.pt"))
            print(f"  softmax_input_L{i}: shape={tuple(x.shape)}  range=[{x.min():.2f},{x.max():.2f}]")
        else:
            print(f"  WARN: no softmax samples for L{i}")

    # FFN weight for linear fidelity
    w16 = model.model.layers[16].mlp.gate_proj.weight.data.clone().cpu()
    torch.save(w16, os.path.join(OUT_DIR, "ffn_gate_L16.pt"))
    print(f"Saved ffn_gate_L16 shape={tuple(w16.shape)}")

    del model
    gc.collect()
    torch.cuda.empty_cache()
    print("Done.")


if __name__ == "__main__":
    main()
