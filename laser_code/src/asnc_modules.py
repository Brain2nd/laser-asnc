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

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def bennett_thresholds(x_np, f_fn, fp_fn, K, n_hist=4096):
    """Bennett-Panter-Dite: λ(x) ∝ p(x)^{1/3}|f'(x)|^{2/3}.
    Returns (K-1,) thresholds in input-space and (K,) reconstruction y[k] = E[f|bin_k]."""
    xmin, xmax = float(x_np.min()), float(x_np.max())
    hist, edges = np.histogram(x_np, bins=n_hist, range=(xmin, xmax))
    p = hist.astype(np.float64)
    p = p / max(p.sum(), 1) / ((xmax - xmin) / n_hist)
    centers = 0.5 * (edges[:-1] + edges[1:])
    ct = torch.from_numpy(centers).float()
    fp = fp_fn(ct).double().abs().numpy()
    lam = (p ** (1/3)) * (fp ** (2/3) + 1e-30)
    lam[p == 0] = 0.0
    cum = np.cumsum(lam); cum /= max(cum[-1], 1e-30)
    thresholds = np.zeros(K - 1, dtype=np.float64)
    for k in range(1, K):
        i = int(np.searchsorted(cum, k / K))
        thresholds[k - 1] = centers[max(0, min(n_hist - 1, i))]
    thresholds = np.maximum.accumulate(thresholds + 1e-15 * np.arange(K - 1))

    # Reconstructions
    xt = torch.from_numpy(x_np).double()
    fx = f_fn(xt.float()).double()
    t_t = torch.from_numpy(thresholds).double()
    boundaries = torch.cat([torch.tensor([xmin - 1.0]), t_t, torch.tensor([xmax + 1.0])])
    y = torch.zeros(K, dtype=torch.float64)
    for k in range(K):
        m = (xt >= boundaries[k]) & (xt < boundaries[k + 1])
        if m.any():
            y[k] = fx[m].mean()
        else:
            mid = 0.5 * (boundaries[k] + boundaries[k + 1])
            y[k] = f_fn(torch.tensor([float(mid)], dtype=torch.float32)).double().item()
    return thresholds.astype(np.float32), y.float().numpy()


class ASNCActivation(nn.Module):
    """Per-channel K-level piecewise-constant codec for scalar activation.
    Each input dimension gets its own K thresholds + K reconstructions."""
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
        H = x.shape[-1]
        mask = torch.isfinite(x).all(dim=-1)
        x = x[mask]
        if x.shape[0] > 50_000:
            idx = torch.randperm(x.shape[0])[:50_000]
            x = x[idx]
        x_np = x.cpu().numpy()

        thresh = torch.zeros(H, self.K - 1)
        recon = torch.zeros(H, self.K)
        for ch in range(H):
            col = x_np[:, ch]
            col = col[np.isfinite(col)]
            if col.size < 2 * self.K:
                # degenerate: use uniform
                lo, hi = float(col.min()) if col.size else -1.0, float(col.max()) if col.size else 1.0
                edges = np.linspace(lo, hi, self.K + 1)
                thresh[ch] = torch.from_numpy(edges[1:-1].astype(np.float32))
                mids = 0.5 * (edges[:-1] + edges[1:])
                recon[ch] = self.f_fn(torch.from_numpy(mids.astype(np.float32)))
                continue
            lo_q, hi_q = np.quantile(col, 0.001), np.quantile(col, 0.999)
            col_clip = col[(col > lo_q) & (col < hi_q)]
            if col_clip.size < self.K:
                col_clip = col
            t_np, y_np = bennett_thresholds(col_clip.astype(np.float64), self.f_fn, self.fp_fn, self.K)
            thresh[ch] = torch.from_numpy(t_np)
            recon[ch] = torch.from_numpy(y_np)
        self.thresholds = thresh.to(self.thresholds.device)
        self.y = recon.to(self.y.device)
        self.fitted = True
        return self

    def forward(self, x):
        if not self.fitted:
            return self.f_fn(x)
        shape = x.shape
        x2 = x.view(-1, shape[-1]).contiguous()  # [N, H]
        t = self.thresholds.to(device=x.device, dtype=x2.dtype)  # [H, K-1]
        y = self.y.to(device=x.device, dtype=x2.dtype)           # [H, K]
        N, H = x2.shape
        K = self.K
        if t.shape[0] != H:
            return self.f_fn(x)
        # Batched per-channel search via torch.searchsorted (single fused kernel).
        # sorted_sequence: t [H, K-1], values: x2.t() [H, N] → idx [H, N]
        idx = torch.searchsorted(t, x2.t().contiguous()).clamp_(0, K - 1).t()  # [N, H]
        # Gather per-channel reconstructions
        out = y.gather(1, idx.t()).t()  # [N, H]
        return out.view(shape)


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
        """Fit codec on softmax output (attention weight) samples in [0, 1]."""
        x = post_softmax_samples.detach().float().flatten().cpu().numpy()
        x = x[np.isfinite(x)]
        x = x[(x >= 0.0) & (x <= 1.0)]
        if x.size < 10:
            raise RuntimeError("too few softmax-output samples to fit")
        # Attention weights are heavy-tailed toward 0. Use plain Bennett with
        # f=identity (we just want to minimize output MSE).
        if x.size > 200_000:
            x = np.random.choice(x, 200_000, replace=False)

        def f_fn(t): return t
        def fp_fn(t): return torch.ones_like(t)
        t_np, y_np = bennett_thresholds(x.astype(np.float64), f_fn, fp_fn, self.K)
        # Clip reconstructions to [0, 1]
        y_np = np.clip(y_np, 0.0, 1.0)
        self.thresholds.data = torch.from_numpy(t_np)
        self.y.data = torch.from_numpy(y_np)
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
    Each of `hidden` input dimensions gets its own K thresholds/reconstructions,
    fit from calibration samples of that specific channel. This preserves per-
    channel activation statistics (different channels have very different
    distributions in LLMs)."""
    def __init__(self, base_ln: nn.Module, K=24, hidden=None):
        super().__init__()
        self.base_ln = base_ln
        self.K = K
        self.register_buffer("thresholds", torch.zeros(1, max(K - 1, 1)))
        self.register_buffer("y", torch.zeros(1, K))
        self.fitted = False
        self.hidden = hidden

    def fit(self, samples: torch.Tensor):
        """samples: [..., hidden] shaped LN INPUT samples (flattened batches allowed)."""
        x = samples.detach().float()
        x = x.view(-1, x.shape[-1]) if x.dim() > 1 else x.view(-1, 1)
        hidden = x.shape[-1]
        self.hidden = hidden
        # filter NaN/Inf
        mask = torch.isfinite(x).all(dim=-1)
        x = x[mask]
        # subsample per channel
        if x.shape[0] > 50_000:
            idx = torch.randperm(x.shape[0])[:50_000]
            x = x[idx]

        def f_fn(t): return t
        def fp_fn(t): return torch.ones_like(t)

        thresh = torch.zeros(hidden, self.K - 1)
        recon = torch.zeros(hidden, self.K)
        x_np = x.cpu().numpy()
        for ch in range(hidden):
            col = x_np[:, ch]
            col = col[np.isfinite(col)]
            if col.size < 2 * self.K:
                # too few samples: degenerate to single bin
                thresh[ch] = torch.linspace(float(col.min()) if col.size else -1.0,
                                           float(col.max()) if col.size else 1.0,
                                           self.K + 1)[1:-1]
                recon[ch] = 0.5 * (torch.linspace(float(col.min()) if col.size else -1.0,
                                                  float(col.max()) if col.size else 1.0,
                                                  self.K + 1)[:-1] +
                                   torch.linspace(float(col.min()) if col.size else -1.0,
                                                  float(col.max()) if col.size else 1.0,
                                                  self.K + 1)[1:])
                continue
            lo_q, hi_q = np.quantile(col, 0.0005), np.quantile(col, 0.9995)
            col = col[(col > lo_q) & (col < hi_q)]
            if col.size < self.K:
                continue
            t_np, y_np = bennett_thresholds(col.astype(np.float64), f_fn, fp_fn, self.K)
            thresh[ch] = torch.from_numpy(t_np)
            recon[ch] = torch.from_numpy(y_np)
        self.thresholds = thresh.to(self.thresholds.device)
        self.y = recon.to(self.y.device)
        self.fitted = True
        return self

    def forward(self, x: torch.Tensor):
        if not self.fitted:
            return self.base_ln(x)
        shape = x.shape
        x2 = x.view(-1, shape[-1]).contiguous()  # [N, H]
        t = self.thresholds.to(device=x.device, dtype=x2.dtype)  # [H, K-1]
        y = self.y.to(device=x.device, dtype=x2.dtype)           # [H, K]
        N, H = x2.shape
        K = self.K
        if t.shape[0] != H:
            return self.base_ln(x)
        idx = torch.searchsorted(t, x2.t().contiguous()).clamp_(0, K - 1).t()  # [N, H]
        x_q = y.gather(1, idx.t()).t()  # [N, H]
        return self.base_ln(x_q.view(shape))


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
