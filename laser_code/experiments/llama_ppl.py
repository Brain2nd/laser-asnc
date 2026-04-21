import gc
import json
import math
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
RESULT_PATH = "/home/dgxspark/Desktop/A2S/results_llama.json"


@torch.no_grad()
def int16_fake_quantize(model, per_channel=True):
    """Per-channel INT16 weight quantization (BSE linear path)."""
    for module in model.modules():
        if isinstance(module, nn.Linear):
            w = module.weight.data
            orig_dtype = w.dtype
            w32 = w.float()
            if per_channel:
                max_abs = w32.abs().amax(dim=1, keepdim=True).clamp_min(1e-30)
            else:
                max_abs = w32.abs().max().clamp_min(1e-30)
            scale = max_abs / 32767.0
            q = torch.round(w32 / scale).clamp(-32768, 32767)
            dq = q * scale
            module.weight.data.copy_(dq.to(orig_dtype))
    return model


@torch.no_grad()
def compute_ppl(model, input_ids, device):
    seq_len = input_ids.size(1)
    nlls = []
    prev_end = 0
    total_tokens = 0
    pbar = tqdm(range(0, seq_len, STRIDE), desc="ppl")
    for begin in pbar:
        end = min(begin + MAX_LEN, seq_len)
        trg_len = end - prev_end
        ids = input_ids[:, begin:end].to(device)
        target_ids = ids.clone()
        target_ids[:, :-trg_len] = -100
        outputs = model(ids, labels=target_ids)
        neg_log_likelihood = outputs.loss * (trg_len - 1 if trg_len > 1 else 1)
        nlls.append(neg_log_likelihood)
        total_tokens += (trg_len - 1 if trg_len > 1 else 1)
        prev_end = end
        if end == seq_len:
            break
    ppl = torch.exp(torch.stack(nlls).sum() / total_tokens)
    return ppl.item()


def main():
    device = "cuda"
    print(f"Loading tokenizer {MODEL}", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL)
    test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(test["text"])
    enc = tok(text, return_tensors="pt").input_ids
    print(f"Total tokens: {enc.size(1)}", flush=True)

    print(f"Loading model {MODEL}", flush=True)
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.float16, low_cpu_mem_usage=True
    ).to(device)
    model.eval()
    print(f"loaded in {time.time()-t0:.1f}s", flush=True)

    t0 = time.time()
    ppl_fp16 = compute_ppl(model, enc, device)
    t_fp16 = time.time() - t0
    print(f"FP16 ppl={ppl_fp16:.4f}  ({t_fp16:.1f}s)", flush=True)

    int16_fake_quantize(model)
    t0 = time.time()
    ppl_int16 = compute_ppl(model, enc, device)
    t_int16 = time.time() - t0
    print(f"INT16 ppl={ppl_int16:.4f}  ({t_int16:.1f}s)", flush=True)

    print(f"ΔPPL = {ppl_int16 - ppl_fp16:+.4f}")

    result = {"model": MODEL, "fp16_ppl": ppl_fp16, "int16_ppl": ppl_int16,
              "delta_ppl": ppl_int16 - ppl_fp16,
              "fp16_sec": t_fp16, "int16_sec": t_int16}
    with open(RESULT_PATH, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
