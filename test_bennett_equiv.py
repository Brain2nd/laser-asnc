"""Unit-test: bennett_thresholds (numpy, per-channel) vs
bennett_thresholds_batched (torch GPU).

Critical cases:
  - normal Gaussian channels (equivalence should hold)
  - outlier channels with q_lo/q_hi masking (reproduces LLM.int8 regime)

The outlier-channel case is what broke the 6.9b/12b/70B runs:
old numpy FILTERED outliers, new batched was CLAMPING them -> edge-bin mass
spike -> Bennett thresholds collapse to boundary. Fix: mask instead of clamp.
"""
import math, time, sys
import numpy as np
import torch

sys.path.insert(0, "/home/dgxspark/Desktop/A2S")
from asnc_modules import bennett_thresholds, bennett_thresholds_batched

torch.manual_seed(0); np.random.seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def identity(x): return x
def ones_like(x): return torch.ones_like(x)


def ref_fit_per_channel(X_np, K, use_quantile_filter=False, q_lo=0.001, q_hi=0.999):
    """Numpy per-channel fit replicating the OLD behaviour."""
    N, H = X_np.shape
    t_ref = np.zeros((H, K - 1), dtype=np.float32)
    y_ref = np.zeros((H, K), dtype=np.float32)
    for h in range(H):
        col = X_np[:, h].astype(np.float64)
        col = col[np.isfinite(col)]
        if use_quantile_filter and col.size > 200:
            lo, hi = np.quantile(col, q_lo), np.quantile(col, q_hi)
            col = col[(col > lo) & (col < hi)]
        if col.size < K:
            continue
        t, y = bennett_thresholds(col, identity, ones_like, K)
        t_ref[h] = t
        y_ref[h] = y
    return t_ref, y_ref


def run_case(name, X, K, quantile_filter=False, q_lo_q=0.001, q_hi_q=0.999):
    N, H = X.shape
    print(f"\n=== {name}: N={N}, H={H}, K={K}, filter={quantile_filter} ===")
    X_cpu = X.float().cpu()
    X_np = X_cpu.numpy()

    # reference: numpy per-channel with optional quantile filter
    t_ref, y_ref = ref_fit_per_channel(X_np, K, quantile_filter, q_lo_q, q_hi_q)

    # batched: use quantile OVERRIDE (mask-based filter)
    Xg = X_cpu.to(device)
    if quantile_filter:
        q_lo = torch.quantile(Xg, q_lo_q, dim=0)
        q_hi = torch.quantile(Xg, q_hi_q, dim=0)
        t_b, y_b = bennett_thresholds_batched(Xg, identity, ones_like, K,
                                               n_hist=4096, device=device,
                                               xmin_override=q_lo,
                                               xmax_override=q_hi)
    else:
        t_b, y_b = bennett_thresholds_batched(Xg, identity, ones_like, K,
                                               n_hist=4096, device=device)

    t_diff = np.abs(t_ref - t_b.cpu().numpy())
    y_diff = np.abs(y_ref - y_b.cpu().numpy())
    print(f"  |Δt| max={t_diff.max():.4f}  mean={t_diff.mean():.4f}")
    print(f"  |Δy| max={y_diff.max():.4f}  mean={y_diff.mean():.4f}")

    # end-to-end reconstruction MSE on ALL samples (outliers included)
    t_ref_g = torch.from_numpy(t_ref).to(device)
    y_ref_g = torch.from_numpy(y_ref).to(device)
    assign_n = torch.searchsorted(t_ref_g, Xg.t().contiguous()).clamp_(0, K - 1)
    y_rec_n = y_ref_g.gather(1, assign_n)
    mse_n = (y_rec_n - Xg.t()).pow(2).mean().item()

    assign_b = torch.searchsorted(t_b, Xg.t().contiguous()).clamp_(0, K - 1)
    y_rec_b = y_b.gather(1, assign_b)
    mse_b = (y_rec_b - Xg.t()).pow(2).mean().item()

    print(f"  recon MSE  numpy={mse_n:.6e}  batched={mse_b:.6e}  ratio={mse_b/max(mse_n,1e-30):.3f}")
    return t_ref, t_b.cpu().numpy()


# Case 1: well-behaved Gaussian
X1 = torch.randn(10000, 512)
run_case("Gaussian H=512, no filter", X1, K=24)
run_case("Gaussian H=512, quantile filter", X1, K=24, quantile_filter=True)

# Case 2: outlier channels (LLM.int8 regime) -- THIS IS THE CRITICAL CASE
X2 = torch.randn(10000, 4096)
outlier_idx = torch.randperm(4096)[:40]  # 1% outlier channels
# typical channel: N(0,1); outlier channel: mostly N(0,1) with 1% huge spikes
for h in outlier_idx:
    spike_mask = torch.rand(10000) < 0.01
    X2[spike_mask, h] = torch.randn(spike_mask.sum()) * 100.0
run_case("Outlier H=4096 @1% channels with 1% ×100 spikes, no filter", X2, K=24)
run_case("Outlier H=4096 @1% channels with 1% ×100 spikes, QUANTILE FILTER", X2, K=24, quantile_filter=True)

# Case 3: H=16384 (mimic pythia-6.9b pre-GeLU)
X3 = torch.randn(20000, 16384)
# inject 0.5% outlier channels with 2% heavy tails
outlier_idx3 = torch.randperm(16384)[:80]
for h in outlier_idx3:
    spike_mask = torch.rand(20000) < 0.02
    X3[spike_mask, h] = torch.randn(spike_mask.sum()) * 50.0
run_case("H=16384 with outlier channels, QUANTILE FILTER", X3, K=32, quantile_filter=True)

print("\nDone.")
