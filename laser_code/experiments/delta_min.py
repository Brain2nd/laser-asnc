"""δ_min and P[|ŷ - y| > δ_min/2] for ASNC-fitted codecs.
Paper (LLaMA-2 7B):
  SiLU (K=32): δ_min=0.028, Pr=0.8%
  Softmax (K=16): δ_min=0.047, Pr=0.3%
  LayerNorm (K=24): δ_min=0.011, Pr=1.2%
"""
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

ACT_DIR = "/home/dgxspark/Desktop/A2S/activations"
RESULT = "/home/dgxspark/Desktop/A2S/results_delta_min.json"


def bennett_codec(x, f, K, fprime_fn=None, n_hist=4096):
    x = x.to(torch.float64).numpy()
    xmin, xmax = float(x.min()), float(x.max())
    hist, edges = np.histogram(x, bins=n_hist, range=(xmin, xmax))
    p = hist.astype(np.float64)
    p = p / max(p.sum(), 1) / ((xmax - xmin) / n_hist)
    centers = 0.5 * (edges[:-1] + edges[1:])
    if fprime_fn is not None:
        fp = fprime_fn(torch.from_numpy(centers).to(torch.float32)).to(torch.float64).abs().numpy()
    else:
        h = (xmax - xmin) / n_hist
        ct = torch.from_numpy(centers).to(torch.float32)
        f_hi = f(ct + h / 2).to(torch.float64).numpy()
        f_lo = f(ct - h / 2).to(torch.float64).numpy()
        fp = np.abs((f_hi - f_lo) / h)
    lam = (p ** (1.0 / 3.0)) * (fp ** (2.0 / 3.0) + 1e-30)
    lam[p == 0] = 0.0
    cum = np.cumsum(lam); cum /= max(cum[-1], 1e-30)
    thresholds = np.zeros(K - 1)
    for k in range(1, K):
        idx = int(np.searchsorted(cum, k / K))
        idx = max(0, min(n_hist - 1, idx))
        thresholds[k - 1] = centers[idx]
    thresholds = np.maximum.accumulate(thresholds + 1e-15 * np.arange(K - 1))

    xt = torch.from_numpy(x)
    fx = f(xt.to(torch.float32)).to(torch.float64)
    t = torch.from_numpy(thresholds).to(torch.float32)
    boundaries = torch.cat([torch.tensor([xmin - 1.0]), t.to(torch.float64), torch.tensor([xmax + 1.0])])
    y = torch.zeros(K, dtype=torch.float64)
    for k in range(K):
        m = (xt >= boundaries[k]) & (xt < boundaries[k + 1])
        if m.any():
            y[k] = fx[m].mean()
        else:
            mid = 0.5 * (boundaries[k] + boundaries[k + 1])
            y[k] = f(torch.tensor([float(mid)], dtype=torch.float32)).double().item()
    return t, y.to(torch.float32)


def apply_codec(x, t, y):
    idx = torch.bucketize(x, t)
    return y[idx]


def silu_fn(x): return F.silu(x)
def silu_fprime(x):
    s = torch.sigmoid(x); return s * (1.0 + x * (1.0 - s))

_MX = [0.0]
def exp_fn(x): return torch.exp(x - _MX[0])
def exp_fprime(x): return torch.exp(x - _MX[0])


def layernorm_scalar(x):
    """Scalar surrogate for LayerNorm gain (identity on pre-norm input)."""
    return x
def layernorm_fprime(x):
    return torch.ones_like(x)


def load_clip(path, q_lo=0.001, q_hi=0.999, max_n=100_000):
    x = torch.load(path, weights_only=True)
    x = x[torch.isfinite(x)]
    lo, hi = x.quantile(q_lo), x.quantile(q_hi)
    x = x[(x > lo) & (x < hi)]
    if x.numel() > max_n:
        x = x[torch.randperm(x.numel())[:max_n]]
    return x


def delta_min_and_exceed(x, t, y, f):
    # δ_min: min spacing of input-space thresholds (paper's definition)
    delta_min_input = (t[1:] - t[:-1]).min().item() if t.numel() > 1 else 0.0
    y_sorted, _ = y.sort()
    delta_min_output = (y_sorted[1:] - y_sorted[:-1]).min().item() if y.numel() > 1 else 0.0
    # Paper: P[|ASNC_out - f(x)| > δ_min_input/2]
    fx = f(x)
    yhat = apply_codec(x, t, y)
    err = (yhat - fx).abs()
    rate_exceed = (err > delta_min_input / 2).float().mean().item()
    return {"delta_min_input": delta_min_input,
            "delta_min_output": delta_min_output,
            "exceed_rate": rate_exceed}


def main():
    torch.manual_seed(0)
    results = {}

    silu = load_clip(os.path.join(ACT_DIR, "silu_input_L16.pt"))
    print(f"SiLU N={len(silu)}  range=[{silu.min():.2f},{silu.max():.2f}]")
    t, y = bennett_codec(silu, silu_fn, K=32, fprime_fn=silu_fprime)
    r = delta_min_and_exceed(silu, t, y, silu_fn)
    r["K"] = 32
    results["SiLU"] = r
    print(f"  SiLU K=32: δ_min_input={r['delta_min_input']:.4f}  "
          f"δ_min_output={r['delta_min_output']:.4f}  Pr[>δ/2]={r['exceed_rate']*100:.2f}%")

    sm = load_clip(os.path.join(ACT_DIR, "softmax_input_L16.pt"))
    print(f"Softmax-input N={len(sm)}  range=[{sm.min():.2f},{sm.max():.2f}]")
    _MX[0] = float(sm.max())
    t, y = bennett_codec(sm, exp_fn, K=16, fprime_fn=exp_fprime)
    r = delta_min_and_exceed(sm, t, y, exp_fn)
    r["K"] = 16
    results["Softmax"] = r
    print(f"  Softmax K=16: δ_min_input={r['delta_min_input']:.4f}  "
          f"δ_min_output={r['delta_min_output']:.4f}  Pr[>δ/2]={r['exceed_rate']*100:.2f}%")

    ln = load_clip(os.path.join(ACT_DIR, "ln2_input_L16.pt"))
    print(f"LN-input N={len(ln)}  range=[{ln.min():.2f},{ln.max():.2f}]")
    # For f=identity, Bennett reduces to λ ∝ p^{1/3} which over-concentrates in
    # dense regions; uniform-in-x is the MSE-optimal under bounded x for linear f.
    # Paper's LN K=24 uses uniform partition → min spacing = (xmax-xmin)/K.
    lo, hi = ln.min().item(), ln.max().item()
    K_ln = 24
    edges_ln = torch.linspace(lo, hi, K_ln + 1)
    t = edges_ln[1:-1]
    y = 0.5 * (edges_ln[:-1] + edges_ln[1:])  # midpoint reconstruction
    r = delta_min_and_exceed(ln, t, y, layernorm_scalar)
    r["K"] = K_ln
    results["LayerNorm"] = r
    print(f"  LN K=24 (uniform): δ_min_input={r['delta_min_input']:.4f}  "
          f"δ_min_output={r['delta_min_output']:.4f}  Pr[>δ/2]={r['exceed_rate']*100:.2f}%")

    with open(RESULT, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {RESULT}")


if __name__ == "__main__":
    main()
