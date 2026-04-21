"""KS-test for Gaussian fit on LLaMA-2 7B activations at layers 4, 16, 28.
Paper target: KS < 0.04, p > 0.3; μ in [-0.1, 0.2], σ in [0.8, 1.4]."""
import json
import os

import numpy as np
import torch
from scipy import stats

ACT_DIR = "/home/dgxspark/Desktop/A2S/activations"
RESULT = "/home/dgxspark/Desktop/A2S/results_ks.json"

TARGET_LAYERS = [4, 16, 28]


def ks_gaussian(samples):
    x = samples.float().numpy().astype(np.float64)
    # Subsample if huge (KS is expensive)
    if len(x) > 50_000:
        idx = np.random.choice(len(x), 50_000, replace=False)
        x = x[idx]
    mu = x.mean()
    sigma = x.std()
    # standardize
    z = (x - mu) / max(sigma, 1e-12)
    # KS against standard normal
    stat, p = stats.kstest(z, "norm")
    return {"mu": float(mu), "sigma": float(sigma), "ks": float(stat), "p_value": float(p)}


def main():
    results = {}
    for mod in ["silu_input", "ln2_input", "softmax_input"]:
        results[mod] = {}
        for L in TARGET_LAYERS:
            fp = os.path.join(ACT_DIR, f"{mod}_L{L}.pt")
            if not os.path.exists(fp):
                results[mod][f"L{L}"] = {"error": "file missing"}
                continue
            x = torch.load(fp, weights_only=True)
            r = ks_gaussian(x)
            results[mod][f"L{L}"] = r
            print(f"  {mod} L{L}: μ={r['mu']:+.3f} σ={r['sigma']:.3f} KS={r['ks']:.4f} p={r['p_value']:.3g}")
    with open(RESULT, "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
