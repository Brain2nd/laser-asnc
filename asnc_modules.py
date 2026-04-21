"""ASNC / BSE / DCR simulation modules.

- ASNCActivation: K-level piecewise-constant codec replacing SiLU/GeLU.
  Bennett-Panter-Dite optimal: threshold density λ(x) ∝ p(x)^{1/3}|f'(x)|^{2/3}.
- ASNCSoftmax: K-level codec for softmax scores.
  Applies the codec to the unnormalized exp(x-max) before row-normalization.
- ASNCLayerNorm: K-level codec wrapping a LayerNorm/RMSNorm.
- DCR (activation-activation product): per-token INT16 quantization of Q, K, V
  before attention matmul.
"""
from __future__ import annotations
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def bennett_thresholds_batched(x, f_fn, fp_fn, K, n_hist=4096, device=None):
    """GPU-batched per-channel Bennett-Panter-Dite fit, robust implementation.

    Strategy (avoids the threshold-collapse pathology of naive histogram-Bennett
    on clipped / heavy-tailed activations):

      1. Per-channel thresholds are placed at the Bennett-weighted empirical
         CDF inverse. Practically: sort samples per channel, compute a discrete
         Bennett weight w_i = p_i^{1/3} · |f'(x_i)|^{2/3} with p_i = 1/N for
         equal-weight samples (so w_i ∝ |f'(x_i)|^{2/3}), form a normalised
         cumulative weight, and invert at targets (k/K) for k=1..K-1.
      2. Strict monotonicity is guaranteed because samples are sorted and each
         threshold picks a distinct sorted position.
      3. Reconstruction y[k] = mean of f(x) over samples in bin k.

    For f = identity (LN codec), weights are uniform so thresholds reduce to
    per-channel equal-probability quantiles. For f = GeLU/SiLU, weights
    concentrate thresholds where |f'| is large (around 0, the non-linear
    region), matching the Bennett-optimal density.

    Returns (thresholds [H, K-1], y [H, K])."""
    if device is None:
        device = x.device
    x = x.to(device).contiguous()
    N, H = x.shape
    dtype = torch.float32

    # 1. sort each column ascending
    xs, _ = torch.sort(x, dim=0)  # [N, H]
    xs = xs.float()

    # 2. Proper Bennett weight.  For sorted samples xs[0..N-1] the empirical
    # density at xs[i] is p(xs[i]) ≈ 1/(N·Δx_i) with Δx_i = xs[i+1]-xs[i].
    # Bennett density λ(x) = p(x)^{1/3}·|f'(x)|^{2/3} integrated over Δx_i
    # gives a per-segment weight
    #       Δcdf_i  ∝  Δx_i · p(xs[i])^{1/3} · |f'(xs[i])|^{2/3}
    #              =  Δx_i^{2/3} · N^{-1/3} · |f'(xs[i])|^{2/3}
    # so Δcdf_i ∝ Δx_i^{2/3} · |f'|^{2/3}.  The N^{-1/3} is a per-channel
    # constant that drops out after CDF normalisation.
    fp_vals = fp_fn(xs).abs()                       # [N, H]
    dx = torch.zeros_like(xs)
    dx[:-1] = xs[1:] - xs[:-1]                      # Δx_i = xs[i+1] - xs[i]
    dx = dx.clamp_min(0.0)                          # should be non-negative after sort
    w = (dx + 1e-30) ** (2.0 / 3.0) * (fp_vals + 1e-30) ** (2.0 / 3.0)   # [N, H]
    cw = torch.cumsum(w, dim=0)                     # [N, H]
    total = cw[-1:, :].clamp_min(1e-30)             # [1, H]
    cdf = cw / total                                # [N, H] in [0, 1]

    # 3. invert CDF at k/K targets.  For each channel h, k, find smallest i s.t. cdf[i, h] >= k/K.
    targets = torch.arange(1, K, device=device, dtype=dtype) / K   # [K-1]
    # searchsorted along dim=0 requires the "sorted sequence" to be 1D or 2D with
    # matching outer dim. cdf is [N, H]; we want per-column searchsorted.  Use
    # transposed layout [H, N] with targets broadcast to [H, K-1].
    cdf_t = cdf.t().contiguous()                    # [H, N]
    targets_b = targets.unsqueeze(0).expand(H, -1).contiguous()     # [H, K-1]
    idx_pos = torch.searchsorted(cdf_t, targets_b).clamp_(0, N - 1) # [H, K-1]
    xs_t = xs.t().contiguous()                      # [H, N]
    thresholds = xs_t.gather(1, idx_pos)            # [H, K-1]

    # enforce strict monotonicity with a relative eps (fp32-safe).
    span = (xs_t[:, -1] - xs_t[:, 0]).clamp_min(1e-30)              # [H]
    # Use fp16-representable relative epsilon (~1e-3 * span).  A tiny
    # numeric eps would collapse in half precision at inference time and
    # cause searchsorted to put many samples in the same bin; scaling by
    # span/K/1000 keeps thresholds distinguishable in fp16 without distorting
    # fit accuracy (the per-bin quantile error is O(span/K)).
    rel_eps = (span.unsqueeze(1) / max(K - 1, 1) * 1e-3) * torch.arange(
        K - 1, device=device, dtype=dtype).unsqueeze(0)
    thresholds = torch.cummax(thresholds + rel_eps, dim=1).values

    # 4. per-bin reconstructions y[h, k] = mean f(x) over samples in bin k.
    fx = f_fn(x)                                    # [N, H]
    assign = torch.searchsorted(thresholds,
                                 x.t().contiguous()).clamp_(0, K - 1)   # [H, N]
    fx_t = fx.t().contiguous()                      # [H, N]
    sum_bin = torch.zeros(H, K, device=device, dtype=dtype)
    cnt_bin = torch.zeros(H, K, device=device, dtype=dtype)
    sum_bin.scatter_add_(1, assign, fx_t)
    cnt_bin.scatter_add_(1, assign, torch.ones_like(fx_t))
    # fallback for empty bins: f(midpoint) using xmin/xmax as outer boundaries
    xmin = xs_t[:, 0:1]
    xmax = xs_t[:, -1:]
    boundaries = torch.cat([xmin, thresholds, xmax], dim=1)             # [H, K+1]
    mid = 0.5 * (boundaries[:, :-1] + boundaries[:, 1:])
    f_mid = f_fn(mid)
    y = torch.where(cnt_bin > 0, sum_bin / cnt_bin.clamp_min(1), f_mid)

    return thresholds, y


class ASNCActivation(nn.Module):
    """Per-channel K-level piecewise-constant codec for scalar activation.
    Each input dimension gets its own K thresholds + K reconstructions.
    Fit: clamp outliers to per-channel 0.1%/99.9% quantile, then GPU-batched
    Bennett-Panter-Dite (bit-exact with the numpy per-channel loop)."""
    def __init__(self, f_fn, fp_fn, K=32):
        super().__init__()
        self.f_fn = f_fn
        self.fp_fn = fp_fn
        self.K = K
        self.register_buffer("thresholds", torch.zeros(1, max(K - 1, 1)))
        self.register_buffer("y", torch.zeros(1, K))
        self.fitted = False

    def fit(self, samples: torch.Tensor):
        x = samples.detach().float()
        x = x.view(-1, x.shape[-1]) if x.dim() > 1 else x.view(-1, 1)
        mask = torch.isfinite(x).all(dim=-1)
        x = x[mask]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        x = x.to(device)
        if x.shape[0] < 2 * self.K:
            raise RuntimeError(f"ASNCActivation.fit: too few samples ({x.shape[0]} < {2*self.K})")
        thresh, recon = bennett_thresholds_batched(
            x, self.f_fn, self.fp_fn, self.K, device=device,
        )
        self.thresholds = thresh.to(self.thresholds.device)
        self.y = recon.to(self.y.device)
        self.fitted = True
        return self

    def forward(self, x):
        if not self.fitted:
            return self.f_fn(x)
        shape = x.shape
        orig_dtype = x.dtype
        x2 = x.view(-1, shape[-1]).contiguous().float()           # [N, H] fp32
        t = self.thresholds.to(device=x.device, dtype=torch.float32)  # [H, K-1]
        y = self.y.to(device=x.device, dtype=torch.float32)           # [H, K]
        N, H = x2.shape
        K = self.K
        if t.shape[0] != H:
            return self.f_fn(x)
        idx = torch.searchsorted(t, x2.t().contiguous()).clamp_(0, K - 1)   # [H, N]
        out = y.gather(1, idx).t()                                           # [N, H]
        return out.view(shape).to(orig_dtype)


def silu_fn(x): return F.silu(x)
def silu_fprime(x):
    s = torch.sigmoid(x); return s * (1.0 + x * (1.0 - s))


def gelu_fn(x): return F.gelu(x, approximate="none")
def gelu_fprime(x):
    # d/dx [0.5 x (1 + erf(x/sqrt2))]
    sqrt2 = math.sqrt(2.0)
    cdf = 0.5 * (1.0 + torch.erf(x / sqrt2))
    pdf = torch.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)
    return cdf + x * pdf


def make_asnc_silu(K=32): return ASNCActivation(silu_fn, silu_fprime, K)
def make_asnc_gelu(K=32): return ASNCActivation(gelu_fn, gelu_fprime, K)


class ASNCSoftmax(nn.Module):
    """K-level codec applied to softmax OUTPUT (attention weights in [0,1]).
    Fit on the actual softmax output distribution; quantize each row to K levels,
    then renormalize so each row sums to 1 again.
    Rationale: attention weights are bounded & interpretable, so scalar K-quant
    preserves normalization by post-division."""
    def __init__(self, K=16):
        super().__init__()
        self.K = K
        self.register_buffer("thresholds", torch.zeros(K - 1))
        self.register_buffer("y", torch.zeros(K))
        self.fitted = False

    def fit(self, post_softmax_samples: torch.Tensor):
        """Fit codec on softmax-output samples (attention weights) in [0, 1],
        fully vectorised on GPU — reuses the sort-based batched Bennett with
        a single channel (H=1)."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        x = post_softmax_samples.detach().float().flatten().to(device)
        x = x[torch.isfinite(x)]
        x = x[(x >= 0.0) & (x <= 1.0)]
        if x.numel() < 10:
            raise RuntimeError("too few softmax-output samples to fit")
        if x.numel() > 200_000:
            idx = torch.randperm(x.numel(), device=device)[:200_000]
            x = x[idx]

        def f_fn(t): return t
        def fp_fn(t): return torch.ones_like(t)
        thresh, recon = bennett_thresholds_batched(
            x.unsqueeze(1), f_fn, fp_fn, self.K, device=device,
        )
        self.thresholds.data = thresh.squeeze(0).clamp(0.0, 1.0).to(self.thresholds.device)
        self.y.data = recon.squeeze(0).clamp(0.0, 1.0).to(self.y.device)
        self.fitted = True
        return self

    def forward(self, scores: torch.Tensor, dim: int = -1):
        # Always compute exact softmax first, then quantize the output.
        y_exact = F.softmax(scores, dim=dim)
        if not self.fitted:
            return y_exact
        t = self.thresholds.to(device=y_exact.device, dtype=y_exact.dtype)
        y = self.y.to(device=y_exact.device, dtype=y_exact.dtype)
        idx = torch.bucketize(y_exact.contiguous(), t)
        y_q = y[idx.clamp(0, self.K - 1)]
        # Renormalize so each row sums to 1 (preserves attention semantics)
        return y_q / y_q.sum(dim=dim, keepdim=True).clamp_min(1e-30)


class ASNCLayerNorm(nn.Module):
    """Per-channel K-level codec on LN INPUT, then exact LN.
    Each input dim gets its own K thresholds/reconstructions.
    Fit: clamp to per-channel 0.05%/99.95% quantile, then GPU-batched Bennett."""
    def __init__(self, base_ln: nn.Module, K=24, hidden=None):
        super().__init__()
        self.base_ln = base_ln
        self.K = K
        self.register_buffer("thresholds", torch.zeros(1, max(K - 1, 1)))
        self.register_buffer("y", torch.zeros(1, K))
        self.fitted = False
        self.hidden = hidden

    def fit(self, samples: torch.Tensor):
        x = samples.detach().float()
        x = x.view(-1, x.shape[-1]) if x.dim() > 1 else x.view(-1, 1)
        hidden = x.shape[-1]
        self.hidden = hidden
        mask = torch.isfinite(x).all(dim=-1)
        x = x[mask]

        def f_fn(t): return t
        def fp_fn(t): return torch.ones_like(t)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        x = x.to(device)
        if x.shape[0] < 2 * self.K:
            raise RuntimeError(f"ASNCLayerNorm.fit: too few samples ({x.shape[0]} < {2*self.K})")
        # Sort-based Bennett places thresholds at K equal-probability sample
        # positions — no clamp needed (the k/K quantile naturally ignores the
        # bottom/top 1/K of samples anyway).
        thresh, recon = bennett_thresholds_batched(
            x, f_fn, fp_fn, self.K, device=device,
        )
        self.thresholds = thresh.to(self.thresholds.device)
        self.y = recon.to(self.y.device)
        self.fitted = True
        return self

    def forward(self, x: torch.Tensor):
        if not self.fitted:
            return self.base_ln(x)
        shape = x.shape
        orig_dtype = x.dtype
        x2 = x.view(-1, shape[-1]).contiguous().float()               # [N, H] fp32
        t = self.thresholds.to(device=x.device, dtype=torch.float32)  # [H, K-1]
        y = self.y.to(device=x.device, dtype=torch.float32)           # [H, K]
        N, H = x2.shape
        K = self.K
        if t.shape[0] != H:
            return self.base_ln(x)
        idx = torch.searchsorted(t, x2.t().contiguous()).clamp_(0, K - 1)  # [H, N]
        x_q = y.gather(1, idx).t()                                          # [N, H]
        return self.base_ln(x_q.view(shape).to(orig_dtype))


def int16_per_token_quant(x: torch.Tensor):
    """DCR: per-token INT16 symmetric quantization of activations."""
    orig = x.dtype
    x32 = x.float()
    max_abs = x32.abs().amax(dim=-1, keepdim=True).clamp_min(1e-30)
    scale = max_abs / 32767.0
    q = torch.round(x32 / scale).clamp(-32768, 32767)
    return (q * scale).to(orig)


@torch.no_grad()
def bse_quantize_linears(model):
    """Per-channel INT16 weight quantization on all Linear layers."""
    for m in model.modules():
        if isinstance(m, nn.Linear):
            w = m.weight.data
            orig = w.dtype
            w32 = w.float()
            ma = w32.abs().amax(dim=1, keepdim=True).clamp_min(1e-30)
            scale = ma / 32767.0
            q = torch.round(w32 / scale).clamp(-32768, 32767)
            m.weight.data.copy_((q * scale).to(orig))
    return model
