"""LLaMA-2 70B FP16/INT16 per-channel PPL on WikiText-2.
Uses device_map='auto' to shard across 4 GPUs (~140GB FP16 weights)."""
import gc
import json
import os
import time

import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

MODEL = "NousResearch/Llama-2-70b-hf"
MAX_LEN = 2048
STRIDE = 512
RESULT = "/workspace/NeuronSpark-V1/results_llama70b.json"


@torch.no_grad()
def int16_fake_quantize(model):
    """Per-channel INT16 weight quant (BSE)."""
    for m in model.modules():
        if isinstance(m, nn.Linear):
            w = m.weight.data
            orig = w.dtype
            dev = w.device
            w32 = w.float()
            max_abs = w32.abs().amax(dim=1, keepdim=True).clamp_min(1e-30)
            scale = max_abs / 32767.0
            q = torch.round(w32 / scale).clamp(-32768, 32767)
            m.weight.data.copy_((q * scale).to(orig))
    return model


@torch.no_grad()
def compute_ppl(model, input_ids):
    seq_len = input_ids.size(1)
    # First device where embed lives
    try:
        first_dev = next(model.parameters()).device
    except StopIteration:
        first_dev = torch.device("cuda:0")
    nlls = []
    prev_end = 0
    total = 0
    for begin in tqdm(range(0, seq_len, STRIDE), desc="ppl"):
        end = min(begin + MAX_LEN, seq_len)
        trg_len = end - prev_end
        ids = input_ids[:, begin:end].to(first_dev)
        tgt = ids.clone()
        tgt[:, :-trg_len] = -100
        out = model(ids, labels=tgt)
        t = trg_len - 1 if trg_len > 1 else 1
        nlls.append(out.loss.float() * t)
        total += t
        prev_end = end
        if end == seq_len:
            break
    return torch.exp(torch.stack(nlls).sum() / total).item()


def main():
    print(f"Loading tokenizer {MODEL}", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL)
    test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(test["text"])
    enc = tok(text, return_tensors="pt").input_ids
    print(f"Total tokens: {enc.size(1)}", flush=True)

    print(f"Loading model {MODEL} with device_map=auto", flush=True)
    t0 = time.time()
    # Use max_memory to avoid running training GPUs into OOM
    max_memory = {i: "42GB" for i in range(torch.cuda.device_count())}
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.float16, low_cpu_mem_usage=True,
        device_map="auto", max_memory=max_memory,
    )
    model.eval()
    print(f"loaded in {time.time()-t0:.1f}s", flush=True)

    # FP16 PPL
    t0 = time.time()
    ppl_fp16 = compute_ppl(model, enc)
    t_fp16 = time.time() - t0
    print(f"FP16 ppl = {ppl_fp16:.4f}  ({t_fp16:.1f}s)", flush=True)

    # INT16 per-channel
    int16_fake_quantize(model)
    t0 = time.time()
    ppl_int16 = compute_ppl(model, enc)
    t_int16 = time.time() - t0
    print(f"INT16 ppl = {ppl_int16:.4f}  ({t_int16:.1f}s)", flush=True)

    dPPL = ppl_int16 - ppl_fp16
    print(f"ΔPPL = {dPPL:+.4f}", flush=True)

    result = {"model": MODEL, "fp16_ppl": ppl_fp16, "int16_ppl": ppl_int16,
              "delta_ppl": dPPL, "fp16_sec": t_fp16, "int16_sec": t_int16}
    with open(RESULT, "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
