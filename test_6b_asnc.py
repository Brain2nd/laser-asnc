"""Component-level ASNC quality test on pythia-6.9b.
Captures ONE layer's activations (no full-model calib accumulation — avoids
the CPU-RAM blowup that killed the full pipeline), fits each ASNC codec,
and measures fitted-sample reconstruction MSE.

Target: MSE ≤ 1e-6 for each non-linearity. If reached, ASNC is fine
and any pipeline failure is elsewhere."""
import sys, os, torch, torch.nn.functional as F
sys.path.insert(0, "/workspace/laser-asnc")
from transformers import AutoModelForCausalLM, AutoTokenizer
from asnc_modules import make_asnc_gelu, ASNCLayerNorm, ASNCSoftmax

torch.manual_seed(0)
device = torch.device("cuda")
MODEL = "EleutherAI/pythia-6.9b"
LAYER_IDX = 15   # middle layer
K_ACT, K_LN, K_SM = 1024, 1024, 256

print(f"loading {MODEL}", flush=True)
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16).to(device).eval()
layer = model.gpt_neox.layers[LAYER_IDX]
H = model.config.hidden_size
print(f"  hidden={H}, layers={model.config.num_hidden_layers}", flush=True)

# Capture ONE layer only (no accumulation across layers) — stays on CPU as we go
pre_gelu, ln1_in, post_softmax = [], [], []

def hook_pg(m, a, out):
    pre_gelu.append(out.detach().float().reshape(-1, out.shape[-1]).cpu())

def hook_ln(m, args, kwargs):
    x = args[0] if args else kwargs.get("hidden_states")
    ln1_in.append(x.detach().float().reshape(-1, x.shape[-1]).cpu())

# post-softmax via monkey-patch F.softmax during attention of this layer only
in_attn = [False]
orig_softmax = F.softmax
def sm_spy(*args, **kwargs):
    out = orig_softmax(*args, **kwargs)
    if in_attn[0]:
        flat = out.detach().float().flatten()
        if flat.numel() > 50000:
            idx = torch.randperm(flat.numel(), device=flat.device)[:50000]
            flat = flat[idx]
        post_softmax.append(flat.cpu())
    return out
def pre_attn(m, a, kw): in_attn[0] = True
def post_attn(m, a, kw, out): in_attn[0] = False

h1 = layer.mlp.dense_h_to_4h.register_forward_hook(hook_pg)
h2 = layer.input_layernorm.register_forward_pre_hook(hook_ln, with_kwargs=True)
h3 = layer.attention.register_forward_pre_hook(pre_attn, with_kwargs=True)
h4 = layer.attention.register_forward_hook(post_attn, with_kwargs=True)
F.softmax = sm_spy

ids = tok("The quick brown fox jumps over the lazy dog. " * 400, return_tensors="pt").input_ids[:, :1024].to(device)
print("running 8 forward passes for capture...", flush=True)
with torch.no_grad():
    for b in range(8):
        _ = model(ids)
F.softmax = orig_softmax
for h in [h1, h2, h3, h4]: h.remove()

pg = torch.cat(pre_gelu)[:8192]
ln = torch.cat(ln1_in)[:8192]
sm = torch.cat(post_softmax)[:200000] if post_softmax else None
print(f"  pre_gelu  {pg.shape}  range [{pg.min():.3f}, {pg.max():.3f}]", flush=True)
print(f"  ln1_in    {ln.shape}  range [{ln.min():.3f}, {ln.max():.3f}]", flush=True)
print(f"  softmax   {sm.shape if sm is not None else None}", flush=True)

# Free the big model before fitting (fit moves data to GPU)
del model, layer
torch.cuda.empty_cache()

# ---- fit + measure MSE ----
def mse(a, b): return (a.float() - b.float()).pow(2).mean().item()

# 1. GeLU
print(f"\nASNCActivation GeLU (K={K_ACT})", flush=True)
asnc_g = make_asnc_gelu(K=K_ACT).to(device)
asnc_g.fit(pg)
with torch.no_grad():
    pg_d = pg.to(device)
    exact = F.gelu(pg_d, approximate="none")
    rec = asnc_g(pg_d)
    print(f"  MSE = {mse(rec, exact):.3e}   max|err| = {(rec-exact).abs().max().item():.4f}", flush=True)

# 2. LN — wrap a fresh LayerNorm with same H
print(f"\nASNCLayerNorm (K={K_LN})", flush=True)
base_ln = torch.nn.LayerNorm(ln.shape[-1]).to(device).half()
asnc_ln = ASNCLayerNorm(base_ln, K=K_LN).to(device)
asnc_ln.fit(ln)
with torch.no_grad():
    ln_d = ln.to(device).half()
    exact = base_ln(ln_d)
    rec = asnc_ln(ln_d)
    print(f"  LN-full MSE = {mse(rec, exact):.3e}   max|err| = {(rec-exact).abs().max().item():.4f}", flush=True)
    # Also input-quant MSE alone
    shape = ln_d.shape
    x2 = ln_d.view(-1, shape[-1]).contiguous().float()
    t = asnc_ln.thresholds.float()
    y = asnc_ln.y.float()
    idx = torch.searchsorted(t, x2.t().contiguous()).clamp_(0, asnc_ln.K-1)
    x_q = y.gather(1, idx).t().view(shape)
    print(f"  LN input-quant MSE = {mse(x_q, ln_d.float()):.3e}   max|err| = {(x_q-ln_d.float()).abs().max().item():.4f}", flush=True)

# 3. Softmax
if sm is not None:
    print(f"\nASNCSoftmax (K={K_SM})", flush=True)
    asnc_sm = ASNCSoftmax(K=K_SM).to(device)
    asnc_sm.fit(sm)
    with torch.no_grad():
        sm_d = sm.to(device)
        t = asnc_sm.thresholds.float()
        y = asnc_sm.y.float()
        idx = torch.bucketize(sm_d, t).clamp(0, asnc_sm.K-1)
        rec = y[idx]
        print(f"  MSE = {mse(rec, sm_d):.3e}   max|err| = {(rec-sm_d).abs().max().item():.4f}", flush=True)

print("\ndone", flush=True)
