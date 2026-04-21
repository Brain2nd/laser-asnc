"""Unit tests: verify each ASNC codec approximates its target function accurately.
Measures MSE between codec output and exact function on real LLaMA-2 / Pythia
activations. If codec MSE is high, the codec is buggy — NOT the integration."""
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from asnc_modules import (
    make_asnc_gelu, ASNCSoftmax, ASNCLayerNorm,
)

ACT_DIR = "/home/dgxspark/Desktop/A2S/activations"


def test_gelu(K=32, layer=16):
    """Test GeLU ASNC codec: does it approximate GeLU(x) accurately?"""
    from asnc_modules import gelu_fn
    path = os.path.join(ACT_DIR, f"silu_input_L{layer}.pt")  # SiLU captured
    if not os.path.exists(path):
        print(f"  [skip gelu, missing {path}]")
        return
    x = torch.load(path, weights_only=True).float()
    # clip
    lo, hi = x.quantile(0.001), x.quantile(0.999)
    x = x[(x > lo) & (x < hi)]
    # fit on half, test on half
    n = x.numel()
    perm = torch.randperm(n)
    x_tr = x[perm[:n // 2]]
    x_te = x[perm[n // 2:]]
    codec = make_asnc_gelu(K=K)
    codec.fit(x_tr)
    y_codec = codec(x_te)
    y_exact = gelu_fn(x_te)
    # Paper uses SiLU for LLaMA, so use that target for this captured data
    y_silu_exact = F.silu(x_te)
    mse_gelu = ((y_codec - y_exact) ** 2).mean().item()
    mse_silu_ref = ((y_codec - y_silu_exact) ** 2).mean().item()
    var_target = y_exact.var().item()
    rel_mse = mse_gelu / max(var_target, 1e-30)
    print(f"  GeLU K={K} on L{layer} SiLU-captured inputs ({n} samples):")
    print(f"    MSE vs GeLU exact = {mse_gelu:.3e}  rel = {rel_mse:.3%}")


def test_softmax(K=16, layer=16, seq_len=512):
    path = os.path.join(ACT_DIR, f"softmax_input_L{layer}.pt")
    if not os.path.exists(path):
        print(f"  [skip softmax, missing {path}]")
        return
    scores_flat = torch.load(path, weights_only=True).float()
    scores_flat = scores_flat[torch.isfinite(scores_flat)]
    scores_flat = scores_flat[scores_flat > -1e4]
    n_fit = scores_flat.numel() // 2
    scores_te = scores_flat[torch.randperm(scores_flat.numel())[:n_fit]]
    # Build rows for softmax evaluation
    rows = scores_te.numel() // seq_len
    scores_mat = scores_te[:rows * seq_len].view(rows, seq_len)
    exact = F.softmax(scores_mat, dim=-1)

    # Fit codec on half of rows' softmax output
    fit_half = exact[:rows // 2].flatten()
    codec = ASNCSoftmax(K=K)
    codec.fit(fit_half)

    # Evaluate on remaining rows
    eval_scores = scores_mat[rows // 2:]
    exact_eval = F.softmax(eval_scores, dim=-1)
    codec_out = codec(eval_scores, dim=-1)
    mse = ((codec_out - exact_eval) ** 2).mean().item()
    max_abs_err = (codec_out - exact_eval).abs().max().item()
    kl = (exact_eval * (torch.log(exact_eval.clamp_min(1e-30))
                       - torch.log(codec_out.clamp_min(1e-30)))).sum(-1).mean().item()
    print(f"  Softmax K={K} on L{layer}:")
    print(f"    MSE (codec vs exact softmax) = {mse:.3e}")
    print(f"    max |err| = {max_abs_err:.3e}")
    print(f"    mean KL(exact || codec) = {kl:.3e}")


def test_layernorm(K=24, layer=16):
    """Test LN ASNC: apply codec to LN output, compare to exact LN output."""
    path = os.path.join(ACT_DIR, f"ln2_input_L{layer}.pt")
    if not os.path.exists(path):
        print(f"  [skip LN, missing {path}]")
        return
    pre_ln = torch.load(path, weights_only=True).float()
    # Simulate LN
    class DummyLN(torch.nn.Module):
        def forward(self, x):
            return (x - x.mean()) / (x.std() + 1e-6)
    base = DummyLN()
    # reshape: we need per-row tokens. The capture flattened, so we just treat
    # all scalars as independent samples — not ideal.
    pre_ln = pre_ln[torch.isfinite(pre_ln)]
    lo, hi = pre_ln.quantile(0.001), pre_ln.quantile(0.999)
    pre_ln = pre_ln[(pre_ln > lo) & (pre_ln < hi)]
    n = pre_ln.numel()
    x_tr = pre_ln[torch.randperm(n)[:100_000]]

    # For scalar LN surrogate (identity post-norm): test codec on x_tr itself
    codec = ASNCLayerNorm(base, K=K)
    # Fit on samples (treating them as LN output under identity LN surrogate)
    codec.fit(x_tr)
    t = codec.thresholds
    y = codec.y
    # Apply codec directly
    x_te = pre_ln[torch.randperm(n)[:100_000]]
    idx = torch.bucketize(x_te.contiguous(), t)
    y_q = y[idx.clamp(0, K - 1)]
    mse = ((y_q - x_te) ** 2).mean().item()
    var = x_te.var().item()
    rel = mse / max(var, 1e-30)
    print(f"  LN K={K} codec on L{layer} (scalar test):")
    print(f"    MSE = {mse:.3e}  rel = {rel:.3%}  thresholds range [{t.min():.3f}, {t.max():.3f}]")


def main():
    print("=== ASNC codec unit tests (LLaMA-2 7B captured activations) ===")
    test_gelu(K=32, layer=16)
    test_softmax(K=16, layer=16)
    test_layernorm(K=24, layer=16)
    # Also larger K to see convergence
    print("\n=== Scaling tests ===")
    for K in (16, 32, 64, 128):
        test_gelu(K=K, layer=16)
    for K in (8, 16, 32, 64):
        test_softmax(K=K, layer=16)
    for K in (16, 24, 48, 96):
        test_layernorm(K=K, layer=16)


if __name__ == "__main__":
    main()
