"""Verify the three-zone Bennett codec substantially improves reconstruction
on outlier-feature channels vs single-zone Bennett (both quantile-masked)."""
import sys, torch, numpy as np
sys.path.insert(0, "/home/dgxspark/Desktop/A2S")
from asnc_modules import bennett_thresholds_batched, bennett_three_zone

torch.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def identity(x): return x
def ones_like(x): return torch.ones_like(x)


def eval_codec(thresholds, y, x, K):
    assign = torch.searchsorted(thresholds, x.t().contiguous()).clamp_(0, K - 1)
    rec = y.gather(1, assign)
    return (rec - x.t()).pow(2).mean().item()


# Build a realistic pythia-6.9b-style outlier channel distribution.
# 4096 channels, ~95% N(0, 1), ~5% outlier channels with heavy ~N(0, 20) spikes.
N, H = 20000, 4096
X = torch.randn(N, H) * 1.0
outlier_idx = torch.randperm(H)[:200]  # 5% outlier channels
for h in outlier_idx:
    spike = torch.rand(N) < 0.05  # 5% of samples per outlier channel are huge
    X[spike, h] = torch.randn(spike.sum()) * 30.0
Xg = X.to(device)

# Single-zone Bennett (current, quantile-masked, K=24)
K = 24
q_lo = torch.quantile(Xg, 0.001, dim=0)
q_hi = torch.quantile(Xg, 0.999, dim=0)
t1, y1 = bennett_thresholds_batched(Xg, identity, ones_like, K=K, n_hist=4096,
                                      device=device, xmin_override=q_lo, xmax_override=q_hi)
mse_single = eval_codec(t1, y1, Xg, K)

# Three-zone Bennett (K_main=16, K_tail=4 → total 24)
t3, y3 = bennett_three_zone(Xg, identity, ones_like, K_main=16, K_tail=4, device=device)
mse_three = eval_codec(t3, y3, Xg, K=24)

print(f"Single-zone (K=24):       recon MSE = {mse_single:.6e}")
print(f"Three-zone (16+4+4=24):    recon MSE = {mse_three:.6e}")
print(f"Reduction: {mse_single/mse_three:.2f}× better on outliers" if mse_three > 0 else "")

# Breakdown: MSE on bulk channels vs outlier channels
bulk_mask = torch.ones(H, dtype=torch.bool); bulk_mask[outlier_idx] = False
bulk_idx = torch.nonzero(bulk_mask).squeeze().to(device)
outlier_idx_g = outlier_idx.to(device)

def ch_mse(thresholds, y, x, K, sel):
    x_s = x[:, sel]
    t_s = thresholds[sel]
    y_s = y[sel]
    assign = torch.searchsorted(t_s, x_s.t().contiguous()).clamp_(0, K - 1)
    rec = y_s.gather(1, assign)
    return (rec - x_s.t()).pow(2).mean().item()

print(f"\nPer-group MSE:")
print(f"                     bulk channels       outlier channels")
print(f"Single-zone:        {ch_mse(t1, y1, Xg, K, bulk_idx):.4e}          {ch_mse(t1, y1, Xg, K, outlier_idx_g):.4e}")
print(f"Three-zone:         {ch_mse(t3, y3, Xg, 24, bulk_idx):.4e}          {ch_mse(t3, y3, Xg, 24, outlier_idx_g):.4e}")
