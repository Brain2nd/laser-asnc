"""Encoding fidelity: BSE vs Rate vs TTFS reconstruction MSE.
Paper targets:
  BSE@FP16:  MSE ~ 1e-9
  BSE@FP32:  MSE ~ 1e-14 (bounded by FP32 precision)
  Rate @16 steps: MSE 10^4-10^5 (on unnormalized activations)
  TTFS@16 steps: similarly large
  TTFS requires 1024 steps to reach FP16 precision (64x latency)
"""
import json
import math
import torch


def bse_encode_decode(x, n_bits=16):
    """INT-N symmetric per-batch quant of FP values. Returns reconstruction."""
    max_abs = x.abs().max().clamp_min(1e-30)
    n_max = 2 ** (n_bits - 1) - 1
    scale = max_abs / n_max
    q = torch.round(x / scale).clamp(-2**(n_bits-1), n_max)
    return q * scale


def rate_encode_decode(x, steps=16):
    """Rate coding with `steps` time steps. Maps x linearly to [0, steps]
    spike counts (using clipping at max/min of input)."""
    x_min, x_max = x.min(), x.max()
    span = (x_max - x_min).clamp_min(1e-30)
    p = ((x - x_min) / span).clamp(0, 1)
    # Each of `steps` time steps: bernoulli(p). Output = count/steps * span + x_min
    # Deterministic: round to nearest of {0, 1, ..., steps}
    counts = torch.round(p * steps)
    return (counts / steps) * span + x_min


def ttfs_encode_decode(x, steps=16):
    """Time-to-first-spike: value v in [0, 1] encoded as spike time t = (1-v)*steps.
    Effective precision: log2(steps+1) bits. Min 0 at v=1, max `steps` at v=0."""
    x_min, x_max = x.min(), x.max()
    span = (x_max - x_min).clamp_min(1e-30)
    v = ((x - x_min) / span).clamp(0, 1)
    t = torch.round((1 - v) * steps)
    v_rec = 1 - t / steps
    return v_rec * span + x_min


def main():
    torch.manual_seed(0)
    N = 10_000
    results = {}

    # Paper protocol: 10,000 random FP values in [-1, 1] (uniform) per precision.
    # BSE@FP16: encode/decode via 16-bit INT quant (16 timesteps).
    x_fp16 = (torch.rand(N) * 2 - 1).to(torch.float16).float()
    results["BSE@FP16_MSE"] = ((bse_encode_decode(x_fp16, n_bits=16) - x_fp16) ** 2).mean().item()

    # BSE@FP32: 32-bit INT quant, limited by FP32 precision.
    # Simulate FP32 precision floor via random roundoff of last bit.
    x_fp32 = (torch.rand(N, dtype=torch.float64) * 2 - 1)
    x_fp32_casted = x_fp32.to(torch.float32)
    # BSE round-trip in fp32 (int32 quant is effectively noop relative to fp32 precision)
    bse_out = bse_encode_decode(x_fp32_casted, n_bits=32).double()
    results["BSE@FP32_MSE"] = ((bse_out - x_fp32) ** 2).mean().item()

    # Rate / TTFS at 16 steps on FP16-representable signal of magnitude ~100
    # (paper's 10^4-10^5 suggests activations are unnormalized; use std~10)
    x = (torch.randn(N) * 10.0).to(torch.float32)
    mse_rate16 = ((rate_encode_decode(x, steps=16) - x) ** 2).mean().item()
    mse_ttfs16 = ((ttfs_encode_decode(x, steps=16) - x) ** 2).mean().item()
    results["Rate@16steps_MSE"] = mse_rate16
    results["TTFS@16steps_MSE"] = mse_ttfs16

    # TTFS@1024 steps — paper says 1024 steps needed for FP16 precision
    mse_ttfs1024 = ((ttfs_encode_decode(x, steps=1024) - x) ** 2).mean().item()
    results["TTFS@1024steps_MSE"] = mse_ttfs1024

    # Also on unit variance for comparison
    x1 = torch.randn(N).to(torch.float32)
    results["Rate@16steps_MSE_unitvar"] = ((rate_encode_decode(x1, 16) - x1) ** 2).mean().item()
    results["TTFS@16steps_MSE_unitvar"] = ((ttfs_encode_decode(x1, 16) - x1) ** 2).mean().item()

    print(json.dumps(results, indent=2))
    with open("/home/dgxspark/Desktop/A2S/results_encoding.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
