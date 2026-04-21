"""Quick softmax-input capture: manually compute QK^T/sqrt(d_head) on the fly.
Targets LLaMA-2 7B layers 4, 16, 28."""
import gc
import math
import os
import time

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "NousResearch/Llama-2-7b-hf"
OUT_DIR = "/home/dgxspark/Desktop/A2S/activations"
DEVICE = "cuda"
TARGET_LAYERS = [4, 16, 28]
N_CHUNKS = 16
CHUNK_LEN = 512
MAX_SAMPLES = 200_000


@torch.no_grad()
def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.float16, low_cpu_mem_usage=True
    ).to(DEVICE)
    model.eval()
    print(f"loaded L={len(model.model.layers)}", flush=True)

    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    text = "\n\n".join(ds["text"])
    enc = tok(text, return_tensors="pt").input_ids[0]
    ids = enc[:N_CHUNKS * CHUNK_LEN].view(N_CHUNKS, CHUNK_LEN).to(DEVICE)

    # Capture q_proj/k_proj outputs for target layers and compute scores
    q_store = {i: [] for i in TARGET_LAYERS}
    k_store = {i: [] for i in TARGET_LAYERS}
    hooks = []
    for i in TARGET_LAYERS:
        attn = model.model.layers[i].self_attn
        def mk_q(idx):
            def h(m, a, kw, output):
                q_store[idx].append(output.detach())
            return h
        def mk_k(idx):
            def h(m, a, kw, output):
                k_store[idx].append(output.detach())
            return h
        hooks.append(attn.q_proj.register_forward_hook(mk_q(i), with_kwargs=True))
        hooks.append(attn.k_proj.register_forward_hook(mk_k(i), with_kwargs=True))

    n_heads = model.config.num_attention_heads
    n_kv = getattr(model.config, "num_key_value_heads", n_heads)
    d_head = model.config.hidden_size // n_heads
    scale = 1.0 / math.sqrt(d_head)
    print(f"n_heads={n_heads}, d_head={d_head}, scale=1/sqrt({d_head})={scale:.4f}", flush=True)

    score_samples = {i: [] for i in TARGET_LAYERS}
    t0 = time.time()
    for b in range(N_CHUNKS):
        # reset captures
        for i in TARGET_LAYERS:
            q_store[i].clear(); k_store[i].clear()
        _ = model(ids[b:b+1], use_cache=False)
        for i in TARGET_LAYERS:
            q = q_store[i][0]  # [1, T, n_heads*d_head]
            k = k_store[i][0]  # [1, T, n_kv*d_head]
            T = q.shape[1]
            q = q.view(1, T, n_heads, d_head).transpose(1, 2)  # [1, H, T, d]
            k = k.view(1, T, n_kv, d_head).transpose(1, 2)
            if n_kv != n_heads:
                # repeat KV heads for GQA (LLaMA-2 7B uses MHA so this is identity)
                k = k.repeat_interleave(n_heads // n_kv, dim=1)
            scores = torch.matmul(q, k.transpose(-1, -2)) * scale  # [1, H, T, T]
            # Sample randomly from scores (avoid huge memory)
            flat = scores.float().flatten()
            # Take random subsample of 50k per batch
            if flat.numel() > 50_000:
                idx = torch.randperm(flat.numel(), device=flat.device)[:50_000]
                flat = flat[idx]
            score_samples[i].append(flat.cpu())
    for h in hooks:
        h.remove()
    print(f"forward done in {time.time()-t0:.1f}s", flush=True)

    for i in TARGET_LAYERS:
        x = torch.cat(score_samples[i])
        if x.numel() > MAX_SAMPLES:
            x = x[torch.randperm(x.numel())[:MAX_SAMPLES]]
        torch.save(x, os.path.join(OUT_DIR, f"softmax_input_L{i}.pt"))
        print(f"L{i}: N={x.numel()}  range=[{x.min():.2f},{x.max():.2f}]  μ={x.mean():.3f}  σ={x.std():.3f}")

    del model; gc.collect(); torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
