"""Compare K=24 single-zone vs K_main=24 + K_tail=4 three-zone (total 32) on:
  - normal Gaussian channels (small-model regime)
  - outlier-feature channels (6.9B+ regime)
Three-zone with MORE bins must not be worse than single-zone on bulk, and
must be much better on outliers."""
import sys, torch
sys.path.insert(0, "/home/dgxspark/Desktop/A2S")
from asnc_modules import bennett_thresholds_batched, bennett_three_zone

torch.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def identity(x): return x
def ones_like(x): return torch.ones_like(x)

def eval_codec(t, y, x, K):
    assign = torch.searchsorted(t, x.t().contiguous()).clamp_(0, K - 1)
    return (y.gather(1, assign) - x.t()).pow(2).mean().item()


# All-Gaussian (small-model regime)
N, H = 50000, 512
Xg = torch.randn(N, H).to(device)
q_lo = torch.quantile(Xg, 0.0005, dim=0)
q_hi = torch.quantile(Xg, 0.9995, dim=0)
t1, y1 = bennett_thresholds_batched(Xg, identity, ones_like, K=24, n_hist=4096,
                                      device=device, xmin_override=q_lo, xmax_override=q_hi)
t3, y3 = bennett_three_zone(Xg, identity, ones_like, K_main=24, K_tail=4, device=device,
                              q_main_lo=0.0005, q_main_hi=0.9995)
print("Gaussian, no outliers")
print(f"  Single K=24:              MSE = {eval_codec(t1, y1, Xg, K=24):.6e}")
print(f"  Three-zone 24+4+4 (K=32): MSE = {eval_codec(t3, y3, Xg, K=32):.6e}")

# Mixed: bulk + outlier channels
X = torch.randn(N, H).to(device)
outlier_idx = torch.randperm(H)[:20]
for h in outlier_idx:
    spike = torch.rand(N, device=device) < 0.02
    X[spike, h] = torch.randn(int(spike.sum().item()), device=device) * 40.0
q_lo = torch.quantile(X, 0.0005, dim=0)
q_hi = torch.quantile(X, 0.9995, dim=0)
t1, y1 = bennett_thresholds_batched(X, identity, ones_like, K=24, n_hist=4096,
                                      device=device, xmin_override=q_lo, xmax_override=q_hi)
t3, y3 = bennett_three_zone(X, identity, ones_like, K_main=24, K_tail=4, device=device,
                              q_main_lo=0.0005, q_main_hi=0.9995)
print("\nMixed bulk + outlier (20/512 outlier channels w/ 2% ×40 spikes)")
print(f"  Single K=24:              MSE = {eval_codec(t1, y1, X, K=24):.6e}")
print(f"  Three-zone 24+4+4 (K=32): MSE = {eval_codec(t3, y3, X, K=32):.6e}")
