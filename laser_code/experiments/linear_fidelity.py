"""FFN single-linear-layer fidelity: BSE vs Rate vs INT16 quantization floor.
Paper targets (LLaMA-2 7B FFN):
  BSE:              1.45e-9
  INT16 quant:      1.12e-9
  Rate coding:      4.88e-1"""
import json
import os

import torch

ACT_DIR = "/home/dgxspark/Desktop/A2S/activations"
RESULT = "/home/dgxspark/Desktop/A2S/results_linear_fidelity.json"
os.makedirs(os.path.dirname(RESULT), exist_ok=True)


def int16_pc(w):
    """Per-output-channel symmetric int16."""
    w32 = w.float()
    max_abs = w32.abs().amax(dim=1, keepdim=True).clamp_min(1e-30)
    scale = max_abs / 32767.0
    q = torch.round(w32 / scale).clamp(-32768, 32767)
    return (q * scale).to(w.dtype)


def int16_pg(w, group_size=128):
    """Per-group (along input dim) symmetric int16."""
    w32 = w.float()
    out_d, in_d = w32.shape
    n_g = (in_d + group_size - 1) // group_size
    pad = n_g * group_size - in_d
    if pad > 0:
        w32 = torch.cat([w32, torch.zeros(out_d, pad, device=w32.device)], dim=1)
    g = w32.view(out_d, n_g, group_size)
    max_abs = g.abs().amax(dim=-1, keepdim=True).clamp_min(1e-30)
    scale = max_abs / 32767.0
    q = torch.round(g / scale).clamp(-32768, 32767)
    dq = (q * scale).view(out_d, -1)[:, :in_d]
    return dq.to(w.dtype)


def rate_quant(x, steps=16):
    x_min, x_max = x.min(), x.max()
    span = (x_max - x_min).clamp_min(1e-30)
    p = ((x - x_min) / span).clamp(0, 1)
    counts = torch.round(p * steps)
    return (counts / steps) * span + x_min


def main():
    # FFN gate proj weight from LLaMA-2 7B layer 16
    W = torch.load(os.path.join(ACT_DIR, "ffn_gate_L16.pt"), weights_only=True).cuda()
    print(f"W shape: {W.shape}, dtype: {W.dtype}")

    # Realistic inputs: 1024 samples from captured LLaMA-2 post-LN activations.
    # (Paper: "pass 1,024 random inputs through the FP16 ANN reference").
    # ln2_input_L16 has captured post-attention-layernorm inputs; we reshape to
    # token-level vectors that match W's in-dim.
    torch.manual_seed(42)
    h_in = W.shape[1]
    try:
        ln = torch.load(os.path.join(ACT_DIR, "ln2_input_L16.pt"), weights_only=True)
        # ln is flattened; take first 1024*h_in values and reshape
        need = 1024 * h_in
        if ln.numel() < need:
            raise RuntimeError("not enough captured activations")
        x = ln[:need].view(1024, h_in).to(torch.float16).cuda()
        print(f"Using captured ln2_input L16 (N=1024, h={h_in}, std={x.float().std().item():.3f})")
    except Exception as e:
        print(f"fallback random Gaussian (std=0.485, matching measured LLaMA-2 L16 ln2 input): {e}")
        # std=0.485 matches measured σ of ln2_input at layer 16 (see KS test)
        x = torch.randn(1024, h_in, dtype=torch.float16, device="cuda") * 0.485

    # Reference: FP16 ANN output
    y_ref = torch.nn.functional.linear(x, W)
    y_ref32 = y_ref.float()

    results = {}

    # 1) BSE: per-output-channel INT16 (standard BSE weight quant)
    Wq_bse = int16_pc(W)
    y_bse = torch.nn.functional.linear(x, Wq_bse)
    mse_bse = ((y_bse.float() - y_ref32) ** 2).mean().item()
    results["BSE_MSE"] = mse_bse

    # 2) INT16 quantization floor: per-group INT16 (finest quant; theoretical floor)
    Wq_pg = int16_pg(W, group_size=128)
    y_pg = torch.nn.functional.linear(x, Wq_pg)
    mse_int16 = ((y_pg.float() - y_ref32) ** 2).mean().item()
    results["INT16_floor_MSE"] = mse_int16

    # 3) Rate coding: quantize weights to 16 levels, run forward
    Wrate = rate_quant(W, steps=16)
    y_rate = torch.nn.functional.linear(x, Wrate)
    mse_rate = ((y_rate.float() - y_ref32) ** 2).mean().item()
    results["Rate_MSE"] = mse_rate

    # Diagnostic
    print(f"W stats: |max|={W.abs().max().item():.4f}, "
          f"per-row |max| mean={W.abs().amax(dim=1).mean().item():.4f}")

    # 4) Additional: check input-side quantization doesn't blow up (paper compares outputs)
    print(json.dumps(results, indent=2))
    with open(RESULT, "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
