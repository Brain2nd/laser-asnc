"""Demo: BSE (IF-neuron) encoding ≡ INT16 fake-quantization, bit-exact.

Why this matters
----------------
In all end-to-end experiments we simulate BSE with per-channel INT16 symmetric
quantization. The math is the same: BSE encodes a calibrated-range FP value as
N=16 binary spikes that map 1:1 to the binary representation of its INT16
quantized index, and an IF (Integrate-and-Fire) neuron reconstructs the
dequantized FP value by summing spike contributions with power-of-two synaptic
weights. That sum is exactly the INT16-dequantize of the original value.

Demo shows:
  for a batch of 10,000 FP32 values x:
    q   = round((x − μ) / λ)            (INT16 code)
    x̂_int16 = q · λ + μ                 (INT16 dequant baseline)
    spk = bin(q, N=16)                  (BSE spike train, MSB-first)
    x̂_bse   = IF(spk)                   (IF-neuron reconstruction)
  assert x̂_int16 ≡ x̂_bse  bit-for-bit
"""
import numpy as np
import torch


def int16_fake_quant(x, v_min, v_max, N=16):
    """Symmetric INT-N quantize-dequantize. Returns (q, x_hat, scale, offset)."""
    span = (v_max - v_min)
    scale = torch.tensor(span / (2 ** N - 1), dtype=torch.float64)  # λ
    offset = torch.tensor(float(v_min), dtype=torch.float64)        # μ
    q = torch.round((x - offset) / scale).clamp(0, 2 ** N - 1).to(torch.int64)
    x_hat = q.to(torch.float64) * scale + offset
    return q, x_hat, scale, offset


def bse_encode(q, N=16):
    """Encode INT-N codes as N binary spikes, MSB first.
    Returns spikes of shape [..., N] (uint8)."""
    masks = 2 ** torch.arange(N - 1, -1, -1, dtype=torch.int64, device=q.device)  # [2^{N-1},...,2^0]
    spikes = ((q.unsqueeze(-1) & masks) > 0).to(torch.uint8)
    return spikes                       # [..., N]


class IFNeuron:
    """Integrate-and-fire decoder for BSE.
    Membrane potential V accumulates spike × synaptic weight for each of N
    time steps. Synaptic weight at time step n = 2^{N-1-n} · λ (MSB-first).
    After N steps, V = q · λ. The final FP value is V + μ.
    """
    def __init__(self, scale, offset, N=16):
        self.scale = scale
        self.offset = offset
        self.N = N
        self.w = 2.0 ** torch.arange(N - 1, -1, -1, dtype=torch.float64, device=scale.device if torch.is_tensor(scale) else "cpu")

    def decode(self, spikes):
        """spikes: [..., N] → reconstruction [...]"""
        V = (spikes.to(torch.float64) * self.w).sum(dim=-1)
        return V * self.scale + self.offset


def bse_linear_via_if(x_spikes, W):
    """Illustrative: BSE linear layer is identical to INT16 matmul.
    x_spikes: [B, in, N], W: [out, in].
    For each time step n, the 'soma' receives the spike pattern scaled by 2^{N-1-n}.
    Accumulated over N steps, the result is Σ_i W_{o,i} · q_i · λ per output o.
    """
    B, in_d, N = x_spikes.shape
    out_d = W.shape[0]
    w_time = 2.0 ** torch.arange(N - 1, -1, -1, dtype=torch.float64, device=x_spikes.device)
    # q = Σ_n 2^{N-1-n} · spike_n (reconstruct INT codes)
    q = (x_spikes.to(torch.float64) * w_time).sum(dim=-1)            # [B, in]
    # Linear: y = q · W^T
    y_q = q @ W.to(torch.float64).t()                                # [B, out]
    return y_q  # still in INT scale; multiply by λ and add μ·W.sum outside if needed


def main():
    torch.manual_seed(0)
    N = 16
    n_samples = 10_000

    # 1. Random FP32 values in a calibrated range
    v_min, v_max = -3.0, 3.0
    x = torch.rand(n_samples, dtype=torch.float64) * (v_max - v_min) + v_min

    # 2. INT16 quantize / dequantize
    scale = torch.tensor((v_max - v_min) / (2 ** N - 1), dtype=torch.float64)
    offset = torch.tensor(v_min, dtype=torch.float64)
    q, x_hat_int16, scale_, offset_ = int16_fake_quant(x, v_min, v_max, N=N)

    # 3. BSE spike encoding
    spikes = bse_encode(q, N=N)

    # 4. IF-neuron reconstruction
    if_neuron = IFNeuron(scale_, offset_, N=N)
    x_hat_bse = if_neuron.decode(spikes)

    # 5. Bit-level equivalence check
    diff = (x_hat_int16 - x_hat_bse).abs().max().item()
    mse  = ((x_hat_int16 - x_hat_bse) ** 2).mean().item()
    exact = torch.equal(x_hat_int16, x_hat_bse)
    print("=" * 60)
    print("BSE (IF-neuron) vs INT16 quantization — equivalence demo")
    print("=" * 60)
    print(f"N samples            : {n_samples}")
    print(f"Precision N          : {N} bits")
    print(f"Range                : [{v_min}, {v_max}]")
    print(f"Scale λ              : {scale_.item():.6e}")
    print(f"Offset μ             : {offset_.item():.6e}")
    print(f"max |Δ| (int16 vs BSE): {diff:.3e}")
    print(f"MSE                  : {mse:.3e}")
    print(f"torch.equal (bit-exact): {exact}")
    assert exact, "BSE and INT16 must agree bit-for-bit!"
    print()

    # 6. Linear-layer equivalence: INT16 matmul ≡ sum of IF outputs
    in_d, out_d = 512, 128
    W = torch.randn(out_d, in_d, dtype=torch.float64) * 0.02
    xb = torch.rand(4, in_d, dtype=torch.float64) * (v_max - v_min) + v_min
    q_b, xb_hat, _, _ = int16_fake_quant(xb, v_min, v_max, N=N)
    sp_b = bse_encode(q_b, N=N)

    # INT16 linear: y = x̂ · W^T where x̂ = q·λ + μ
    y_int16 = xb_hat @ W.t()

    # BSE linear via per-time-step IF accumulation
    y_bse_q = bse_linear_via_if(sp_b, W)           # integer scale
    y_bse = y_bse_q * scale_ + offset_ * W.sum(dim=-1).to(torch.float64)

    diff_lin = (y_int16 - y_bse).abs().max().item()
    exact_lin = torch.allclose(y_int16, y_bse, atol=1e-12, rtol=0)
    print(f"Linear layer ({in_d}→{out_d}):")
    print(f"  max |Δ| (int16 matmul vs BSE soma): {diff_lin:.3e}")
    print(f"  allclose (atol=1e-12)               : {exact_lin}")
    assert exact_lin, "BSE soma must equal INT16 matmul up to FP round-off"
    print()
    print("CONCLUSION")
    print("-" * 60)
    print("BSE (IF-neuron spike encode/decode) and INT16 symmetric quantization")
    print("are MATHEMATICALLY IDENTICAL. Using INT16 fake-quant in experiments is")
    print("a pure speed optimisation; the bit pattern of every reconstructed value")
    print("is byte-for-byte equal to what an IF neuron would produce.")


if __name__ == "__main__":
    main()
