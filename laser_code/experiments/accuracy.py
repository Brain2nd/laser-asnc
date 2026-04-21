"""5-model × 4-benchmark accuracy evaluation (MMLU/HellaSwag/ARC/TruthfulQA).
Runs ANN (FP16) and SNN (INT16 per-channel weight quant) for each.
Paper table: tab:scaling."""
import argparse
import gc
import json
import os
import time

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from lm_eval.models.huggingface import HFLM
from lm_eval import simple_evaluate

MODELS = {
    "phi-2":         "microsoft/phi-2",
    "llama-2-7b":    "NousResearch/Llama-2-7b-hf",
    "mistral-7b":    "mistralai/Mistral-7B-v0.1",
    # "mixtral-8x7b":  "mistralai/Mixtral-8x7B-v0.1",
    # "llama-2-70b":   "NousResearch/Llama-2-70b-hf",
}
TASKS = ["mmlu", "hellaswag", "arc_challenge", "truthfulqa_mc2"]
RESULT = "/home/dgxspark/Desktop/A2S/results_accuracy.json"
DEVICE = "cuda"
N_FEWSHOT = 0


@torch.no_grad()
def int16_quant(model):
    for m in model.modules():
        if isinstance(m, nn.Linear):
            w = m.weight.data
            orig = w.dtype
            w32 = w.float()
            ma = w32.abs().amax(dim=1, keepdim=True).clamp_min(1e-30)
            scale = ma / 32767.0
            q = torch.round(w32 / scale).clamp(-32768, 32767)
            m.weight.data.copy_((q * scale).to(orig))


def eval_model(name, path, mode):
    print(f"\n=== {name} [{mode}] ===", flush=True)
    t0 = time.time()
    kwargs = dict(torch_dtype=torch.float16, low_cpu_mem_usage=True, device_map=DEVICE)
    model = AutoModelForCausalLM.from_pretrained(path, **kwargs)
    tok = AutoTokenizer.from_pretrained(path)
    if mode == "SNN":
        int16_quant(model)
    model.eval()
    print(f"[{name}/{mode}] loaded/quantized in {time.time()-t0:.1f}s")

    hflm = HFLM(pretrained=model, tokenizer=tok, batch_size=4)
    results = {}
    for task in TASKS:
        t0 = time.time()
        try:
            out = simple_evaluate(model=hflm, tasks=[task], num_fewshot=N_FEWSHOT,
                                  limit=None)
            # Extract primary metric
            r = out["results"].get(task, {})
            # Choose accuracy-like metric
            metric_keys = ["acc_norm,none", "acc,none", "mc2,none", "mc1,none"]
            acc = None
            for k in metric_keys:
                if k in r:
                    acc = r[k]
                    break
            if acc is None and r:
                # fallback: first numeric
                for k, v in r.items():
                    if isinstance(v, (int, float)):
                        acc = v; break
            results[task] = {"acc": acc, "time": time.time()-t0, "all_metrics": r}
            print(f"  {task}: {acc}  ({time.time()-t0:.0f}s)", flush=True)
        except Exception as e:
            import traceback; traceback.print_exc()
            results[task] = {"error": str(e)}

    del model, hflm
    gc.collect()
    torch.cuda.empty_cache()
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", type=str, default=None, help="comma-sep model keys")
    parser.add_argument("--mode", choices=["ANN", "SNN", "both"], default="both")
    args = parser.parse_args()

    results = {}
    if os.path.exists(RESULT):
        with open(RESULT) as f:
            results = json.load(f)

    targets = list(MODELS.keys())
    if args.only:
        want = set(args.only.split(","))
        targets = [k for k in targets if k in want]

    for name in targets:
        path = MODELS[name]
        results.setdefault(name, {})
        modes = ["ANN", "SNN"] if args.mode == "both" else [args.mode]
        for mode in modes:
            if mode in results[name] and all(t in results[name][mode] for t in TASKS):
                print(f"[skip] {name}/{mode}")
                continue
            results[name][mode] = eval_model(name, path, mode)
            with open(RESULT, "w") as f:
                json.dump(results, f, indent=2, default=str)

    print("\n=== Summary ===")
    for name in targets:
        print(f"\n{name}:")
        for mode in ["ANN", "SNN"]:
            if mode not in results.get(name, {}):
                continue
            d = results[name][mode]
            row = {t: d.get(t, {}).get("acc") for t in TASKS}
            print(f"  {mode}: " + "  ".join(f"{t}={row[t]}" for t in TASKS))


if __name__ == "__main__":
    main()
