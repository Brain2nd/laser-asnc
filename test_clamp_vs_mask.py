"""Direct compare CLAMP-based fit vs MASK-based fit on a Gaussian channel."""
import sys, torch
sys.path.insert(0, "/home/dgxspark/Desktop/A2S")
from asnc_modules import bennett_thresholds_batched

torch.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def identity(x): return x
def ones_like(x): return torch.ones_like(x)

N, H = 50000, 512
X = torch.randn(N, H).to(device)

# Method A: CLAMP (old)
q_lo = torch.quantile(X, 0.001, dim=0)
q_hi = torch.quantile(X, 0.999, dim=0)
X_clamp = torch.clamp(X, q_lo, q_hi)
# In old code, bennett was called with no override, so it computed xmin/xmax from the clamped x
t_a, y_a = bennett_thresholds_batched(X_clamp, identity, ones_like, K=24, n_hist=4096, device=device)

# Method B: MASK (new)
t_b, y_b = bennett_thresholds_batched(X, identity, ones_like, K=24, n_hist=4096, device=device,
                                        xmin_override=q_lo, xmax_override=q_hi)

def eval_codec(t, y, x, K):
    assign = torch.searchsorted(t, x.t().contiguous()).clamp_(0, K - 1)
    return (y.gather(1, assign) - x.t()).pow(2).mean().item()

print(f"CLAMP-based fit:  recon MSE = {eval_codec(t_a, y_a, X, K=24):.6e}")
print(f"MASK-based fit:   recon MSE = {eval_codec(t_b, y_b, X, K=24):.6e}")
print(f"|t_a - t_b| max = {(t_a - t_b).abs().max().item():.4f}  mean = {(t_a - t_b).abs().mean().item():.4f}")
print(f"|y_a - y_b| max = {(y_a - y_b).abs().max().item():.4f}  mean = {(y_a - y_b).abs().mean().item():.4f}")

# Critical check: what do the edge reconstructions look like?
print(f"\nEdge bin reconstructions (channel 0):")
print(f"  CLAMP:  y[0] = {y_a[0, 0].item():.4f}  y[K-1] = {y_a[0, -1].item():.4f}")
print(f"  MASK:   y[0] = {y_b[0, 0].item():.4f}  y[K-1] = {y_b[0, -1].item():.4f}")
print(f"  q_lo[0] = {q_lo[0].item():.4f}  q_hi[0] = {q_hi[0].item():.4f}")

# Also check inference behavior on out-of-range samples
out_of_range = torch.tensor([[-5.0, -4.0, 4.0, 5.0]]).to(device).expand(4, H).contiguous()
print(f"\nReconstruction of out-of-range inputs [-5, -4, +4, +5] (channel 0):")
t_a_ch = t_a[0:1]; y_a_ch = y_a[0:1]
assign_a = torch.searchsorted(t_a_ch, torch.tensor([[-5.0, -4.0, 4.0, 5.0]]).to(device)).clamp_(0, 23)
t_b_ch = t_b[0:1]; y_b_ch = y_b[0:1]
assign_b = torch.searchsorted(t_b_ch, torch.tensor([[-5.0, -4.0, 4.0, 5.0]]).to(device)).clamp_(0, 23)
print(f"  CLAMP: {y_a_ch.gather(1, assign_a).flatten().tolist()}")
print(f"  MASK:  {y_b_ch.gather(1, assign_b).flatten().tolist()}")
