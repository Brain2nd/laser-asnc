import argparse
import gc
import json
import os
import time

import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

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

MAX_LEN = 2048
STRIDE = 512
RESULT_PATH = "/home/dgxspark/Desktop/A2S/results.json"


@torch.no_grad()
def int16_fake_quantize(model):
    """Per-tensor symmetric int16 fake-quantization on all Linear weights."""
    for module in model.modules():
        if isinstance(module, nn.Linear):
            w = module.weight.data
            max_abs = w.abs().max()
            if max_abs == 0:
                continue
            scale = max_abs / 32767.0
            q = torch.round(w / scale).clamp(-32768, 32767).to(torch.int16)
            dq = q.to(w.dtype) * scale
            module.weight.data.copy_(dq)
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
        # HF's loss averages over (trg_len - 1) non-ignored positions.
        neg_log_likelihood = outputs.loss * (trg_len - 1 if trg_len > 1 else 1)
        nlls.append(neg_log_likelihood)
        total_tokens += (trg_len - 1 if trg_len > 1 else 1)
        prev_end = end
        if end == seq_len:
            break
    ppl = torch.exp(torch.stack(nlls).sum() / total_tokens)
    return ppl.item()


def load_results():
    if os.path.exists(RESULT_PATH):
        with open(RESULT_PATH) as f:
            return json.load(f)
    return {}


def save_results(results):
    with open(RESULT_PATH, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def run_one(model_name, input_ids_cpu, device):
    print(f"\n{'='*60}\n== {model_name}\n{'='*60}", flush=True)
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, low_cpu_mem_usage=True
    ).to(device)
    model.eval()
    print(f"[{model_name}] load in {time.time()-t0:.1f}s", flush=True)

    # FP16 ppl
    t0 = time.time()
    ppl_fp16 = compute_ppl(model, input_ids_cpu, device)
    t_fp16 = time.time() - t0
    print(f"[{model_name}] FP16 ppl={ppl_fp16:.4f} ({t_fp16:.1f}s)", flush=True)

    # INT16 fake quant
    int16_fake_quantize(model)
    t0 = time.time()
    ppl_int16 = compute_ppl(model, input_ids_cpu, device)
    t_int16 = time.time() - t0
    print(f"[{model_name}] INT16 ppl={ppl_int16:.4f} ({t_int16:.1f}s)", flush=True)

    del model
    gc.collect()
    torch.cuda.empty_cache()

    return {"fp16_ppl": ppl_fp16, "int16_ppl": ppl_int16,
            "fp16_sec": t_fp16, "int16_sec": t_int16}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", type=str, default=None, help="comma-sep subset")
    args = parser.parse_args()

    device = "cuda"
    print("Loading WikiText-2 test split…", flush=True)
    test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(test["text"])

    # tokenize once with the first model's tokenizer (all Pythia share tokenizer)
    tok = AutoTokenizer.from_pretrained(MODELS[0])
    enc = tok(text, return_tensors="pt")
    input_ids_cpu = enc.input_ids
    print(f"Total tokens: {input_ids_cpu.size(1)}", flush=True)

    targets = MODELS
    if args.only:
        want = set(args.only.split(","))
        targets = [m for m in MODELS if m.split("/")[-1] in want]

    results = load_results()
    for m in targets:
        key = m.split("/")[-1]
        if key in results and "int16_ppl" in results[key]:
            print(f"[skip] {key} already done: {results[key]}", flush=True)
            continue
        try:
            r = run_one(m, input_ids_cpu, device)
            results[key] = r
            save_results(results)
        except Exception as e:
            print(f"[error] {key}: {e}", flush=True)
            results[key] = {"error": str(e)}
            save_results(results)

    print("\n=== Final ===")
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
