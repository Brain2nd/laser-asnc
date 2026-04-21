"""Minimal reproducer: load pythia-70m, install ONLY ASNCLayerNorm, measure PPL.
Print layer-0 codec thresholds/y before and after to catch drift."""
import sys, torch, torch.nn.functional as F
sys.path.insert(0, "/home/dgxspark/Desktop/A2S")
from transformers import AutoModelForCausalLM, AutoTokenizer
from asnc_modules import ASNCLayerNorm

torch.manual_seed(0)
device = torch.device("cuda")

print("loading pythia-70m", flush=True)
tok = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m")
model = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-70m", torch_dtype=torch.float16).to(device).eval()

# Capture ln1_in for layer 0 only
ln1_in = []
layer0 = model.gpt_neox.layers[0]
def pre_hook(m, args, kwargs):
    x = args[0] if args else kwargs.get("hidden_states")
    ln1_in.append(x.detach().float().reshape(-1, x.shape[-1]).cpu())
h = layer0.input_layernorm.register_forward_pre_hook(pre_hook, with_kwargs=True)
ids = tok("The quick brown fox jumps over the lazy dog. " * 200, return_tensors="pt").input_ids[:, :1024].to(device)
with torch.no_grad():
    for _ in range(8):
        model(ids)
h.remove()
ln = torch.cat(ln1_in)[:50000]
print(f"captured {ln.shape}")

# Install ASNCLayerNorm on layer 0 only
codec = ASNCLayerNorm(layer0.input_layernorm, K=24)
codec.fit(ln)
layer0.input_layernorm = codec.to(device)

# Diagnostic on channel 0
t0 = codec.thresholds[0].cpu().tolist()
y0 = codec.y[0].cpu().tolist()
print(f"ch0 thresholds (fp32): {t0[:3]} ... {t0[-3:]}")
print(f"ch0 thresholds (fp16): {torch.tensor(t0).half().tolist()[:3]} ... {torch.tensor(t0).half().tolist()[-3:]}")
print(f"ch0 y (fp32):         {y0[:3]} ... {y0[-3:]}")

# Check for unique fp16 thresholds per channel
t_fp16 = codec.thresholds.to(torch.float16).to(torch.float32)
unique_per_ch = torch.tensor([len(set(row.tolist())) for row in t_fp16])
print(f"unique fp16 thresholds per channel: min={unique_per_ch.min().item()}, mean={unique_per_ch.float().mean().item():.2f}, max={unique_per_ch.max().item()}, K-1={codec.K-1}")
n_collapsed = (unique_per_ch < codec.K - 1).sum().item()
print(f"channels with any fp16 collapse: {n_collapsed} / {len(unique_per_ch)}")
