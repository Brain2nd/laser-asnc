"""ASNC distortion D_K vs K for SiLU / Softmax — proper Lloyd-Max (K-means) fitting.

For each K ∈ {4, 8, 16, 32, 64, 128}:
  Non-uniform codec (ASNC):
    1. Collect calibration samples x (from LLaMA-2 7B activations)
    2. Compute f(x) for each sample
    3. Run K-means on f(x) scalars → get K centroids (reconstruction points y_i*)
       and the induced partition of x-space (thresholds t_i)
    4. Measure D_K = E[(f(x) - ŷ(x))^2] on held-out test samples
  Uniform codec (baseline):
    Equispaced thresholds on x; reconstruction = f(midpoint) of each bin
Fit log(D_K) vs log(K) — expect slope ≈ -2 (Bennett-integral guarantee at Lloyd-Max).
Also report Lloyd-Max closeness: |y_i* - ŷ_i| / |y_i*| per segment."""
import json
import math
import os

import numpy as np
import torch
import torch.nn.functional as F

ACT_DIR = "/home/dgxspark/Desktop/A2S/activations"
RESULT = "/home/dgxspark/Desktop/A2S/results_asnc.json"


def kmeans_1d(fx, K, iters=150, tol=1e-10):
    """1D K-means on scalar values fx. Returns centroids and assignment."""
    fx = fx.clone()
    # Init: quantile-based
    q = torch.linspace(0.5 / K, 1 - 0.5 / K, K, dtype=fx.dtype)
    centroids = torch.quantile(fx, q)
    for it in range(iters):
        # Assign each sample to nearest centroid (1D broadcasting)
        # To keep memory bounded for large fx, chunk the comparison
        assign = torch.empty(fx.shape, dtype=torch.long, device=fx.device)
        chunk = 200_000
        for s in range(0, fx.numel(), chunk):
            e = min(s + chunk, fx.numel())
            d = (fx[s:e].unsqueeze(1) - centroids.unsqueeze(0)).abs()
            assign[s:e] = d.argmin(dim=1)
        # Update centroids
        new_c = centroids.clone()
        for k in range(K):
            m = assign == k
            if m.any():
                new_c[k] = fx[m].mean()
        if (new_c - centroids).abs().max().item() < tol:
            break
        centroids = new_c
    return centroids, assign


def lloyd_max_codec(x_tr, f, K, fprime_fn=None, n_hist=4096):
    """Bennett-Panter-Dite ASNC: threshold density λ(x) ∝ p(x)^{1/3} |f'(x)|^{2/3}.
    If fprime_fn is provided, use it analytically; else use finite-difference."""
    import numpy as np
    x = x_tr.to(torch.float64).numpy()
    xmin, xmax = float(x.min()), float(x.max())
    hist, edges = np.histogram(x, bins=n_hist, range=(xmin, xmax))
    p = hist.astype(np.float64)
    p = p / max(p.sum(), 1) / ((xmax - xmin) / n_hist)
    centers = 0.5 * (edges[:-1] + edges[1:])
    if fprime_fn is not None:
        ct = torch.from_numpy(centers).to(torch.float32)
        fp = fprime_fn(ct).to(torch.float64).abs().numpy()
    else:
        h = (xmax - xmin) / n_hist
        ct = torch.from_numpy(centers).to(torch.float32)
        f_hi = f(ct + h / 2).to(torch.float64).numpy()
        f_lo = f(ct - h / 2).to(torch.float64).numpy()
        fp = np.abs((f_hi - f_lo) / h)
    lam = (p ** (1.0 / 3.0)) * (fp ** (2.0 / 3.0) + 1e-30)
    lam[p == 0] = 0.0
    cum = np.cumsum(lam)
    cum = cum / max(cum[-1], 1e-30)

    thresholds = np.zeros(K - 1, dtype=np.float64)
    for k in range(1, K):
        idx = int(np.searchsorted(cum, k / K))
        idx = max(0, min(n_hist - 1, idx))
        thresholds[k - 1] = centers[idx]
    thresholds = np.maximum.accumulate(thresholds + 1e-15 * np.arange(K - 1))

    xt = torch.from_numpy(x).to(torch.float64)
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


def uniform_codec(x_tr, f, K):
    lo, hi = x_tr.min().item(), x_tr.max().item()
    edges = torch.linspace(lo, hi, K + 1)
    t = edges[1:-1]
    mids = (edges[:-1] + edges[1:]) / 2
    y = f(mids)
    return t, y


def apply_codec(x, t, y):
    idx = torch.bucketize(x, t.to(x.device))
    return y.to(x.device)[idx]


def fit_slope(Ks, Ds):
    logK = np.log(np.array(Ks))
    logD = np.log(np.array(Ds))
    A = np.stack([logK, np.ones_like(logK)], axis=1)
    sol, *_ = np.linalg.lstsq(A, logD, rcond=None)
    slope = sol[0]
    # bootstrap std
    slopes = []
    for _ in range(500):
        ii = np.random.choice(len(Ks), len(Ks), replace=True)
        s = np.linalg.lstsq(A[ii], logD[ii], rcond=None)[0][0]
        slopes.append(s)
    return float(slope), float(np.std(slopes))


def silu_fn(x):
    return F.silu(x)


def silu_fprime(x):
    s = torch.sigmoid(x)
    return s * (1.0 + x * (1.0 - s))


_EXP_SHIFT = [0.0]


def exp_fn(x):
    """Scalar surrogate for Softmax: exp(x) shifted by global max (value range ≤1)."""
    return torch.exp(x - _EXP_SHIFT[0])


def exp_fprime(x):
    return torch.exp(x - _EXP_SHIFT[0])


def run_distortion(x_all, name, f):
    n = x_all.numel()
    perm = torch.randperm(n)
    x_all = x_all[perm]
    ntr = n // 2
    x_tr = x_all[:ntr]
    x_te = x_all[ntr:]
    f_te = f(x_te)

    Ks = [4, 8, 16, 32, 64, 128]
    result = {"Ks": Ks, "non_uniform_D": [], "uniform_D": [], "lloyd_closeness": []}

    for K in Ks:
        # Non-uniform (Lloyd-Max / Bennett)
        fprime = globals().get(name.lower() + "_fprime")
        t_n, y_n = lloyd_max_codec(x_tr, f, K, fprime_fn=fprime)
        approx_n = apply_codec(x_te, t_n, y_n)
        D_n = ((approx_n - f_te) ** 2).mean().item()

        # Uniform
        t_u, y_u = uniform_codec(x_tr, f, K)
        approx_u = apply_codec(x_te, t_u, y_u)
        D_u = ((approx_u - f_te) ** 2).mean().item()

        # Closeness: |ŷ_i - y_i*| / |y_i*| where y_i* = E[f | x in bin i_test]
        edges = torch.cat([torch.tensor([-float("inf")]), t_n, torch.tensor([float("inf")])])
        devs = []
        for i in range(K):
            m = (x_te >= edges[i]) & (x_te < edges[i + 1])
            if m.any():
                y_star = f(x_te[m]).mean().item()
                y_hat = y_n[i].item()
                if abs(y_star) > 1e-8:
                    devs.append(abs(y_hat - y_star) / abs(y_star))
        closeness = float(np.mean(devs)) if devs else None

        result["non_uniform_D"].append(D_n)
        result["uniform_D"].append(D_u)
        result["lloyd_closeness"].append(closeness)
        print(f"  {name} K={K:3d}  non-uniform D={D_n:.3e}  uniform D={D_u:.3e}  close={closeness}", flush=True)

    slope_n, slope_n_std = fit_slope(Ks, result["non_uniform_D"])
    slope_u, slope_u_std = fit_slope(Ks, result["uniform_D"])
    result["slope_non_uniform"] = slope_n
    result["slope_non_uniform_std"] = slope_n_std
    result["slope_uniform"] = slope_u
    result["slope_uniform_std"] = slope_u_std
    result["uniform_over_nonuniform_ratio"] = [
        u / n_ for u, n_ in zip(result["uniform_D"], result["non_uniform_D"])
    ]
    print(f"  {name}  slope non-uniform = {slope_n:+.3f} ± {slope_n_std:.3f}", flush=True)
    print(f"  {name}  slope uniform     = {slope_u:+.3f} ± {slope_u_std:.3f}", flush=True)
    print(f"  {name}  u/nu ratio = {[round(r,2) for r in result['uniform_over_nonuniform_ratio']]}", flush=True)
    return result


def load_multi_layer(prefix, layers, max_n=250_000, k_sigma=3.0):
    """Load samples from multiple layers, clip by μ±k_sigma*σ (calibrated range)."""
    all_x = []
    for L in layers:
        fp = os.path.join(ACT_DIR, f"{prefix}_L{L}.pt")
        if not os.path.exists(fp):
            continue
        x = torch.load(fp, weights_only=True).float()
        x = x[torch.isfinite(x)]
        all_x.append(x)
    x = torch.cat(all_x) if all_x else torch.empty(0)
    # Outlier handling: μ ± k·σ calibration range (paper's approach with λ, μ)
    mu = x.mean()
    sd = x.std()
    lo = mu - k_sigma * sd
    hi = mu + k_sigma * sd
    mask_in = (x >= lo) & (x <= hi)
    outlier_rate = (1 - mask_in.float().mean()).item()
    x = x[mask_in]
    if x.numel() > max_n:
        x = x[torch.randperm(x.numel())[:max_n]]
    return x, outlier_rate, (lo.item(), hi.item())


def main():
    torch.manual_seed(0)
    # Use multiple middle layers for better stat
    layers_for_silu = list(range(8, 25))  # middle layers
    layers_for_sm = [4, 16, 28]

    silu, out_silu, rng_silu = load_multi_layer("silu_input", layers_for_silu, k_sigma=4.0)
    print(f"SiLU N={silu.numel()}  clip=[{rng_silu[0]:.2f},{rng_silu[1]:.2f}]  "
          f"outlier_rate={out_silu*100:.2f}%  μ={silu.mean():.3f}  σ={silu.std():.3f}")
    sm, out_sm, rng_sm = load_multi_layer("softmax_input", layers_for_sm, k_sigma=4.0)
    print(f"Softmax-input N={sm.numel()}  clip=[{rng_sm[0]:.2f},{rng_sm[1]:.2f}]  "
          f"outlier_rate={out_sm*100:.2f}%  μ={sm.mean():.3f}  σ={sm.std():.3f}")

    results = {
        "SiLU_outlier_rate": out_silu,
        "Softmax_outlier_rate": out_sm,
        "SiLU_clip_range": rng_silu,
        "Softmax_clip_range": rng_sm,
    }
    print("\n--- SiLU (analytic f', multi-layer) ---")
    results["SiLU"] = run_distortion(silu, "SiLU", silu_fn)
    print("\n--- Softmax (analytic f', multi-layer) ---")
    _EXP_SHIFT[0] = float(sm.max().item())
    results["Softmax"] = run_distortion(sm, "Softmax", exp_fn)

    with open(RESULT, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {RESULT}")


if __name__ == "__main__":
    main()
