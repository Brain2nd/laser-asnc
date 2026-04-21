"""Softmax ASNC unit test on pythia-6.9b with eager attention forced,
so F.softmax is invoked and our spy captures post-softmax attention weights.
Measures codec reconstruction MSE on the captured distribution."""
import sys, torch, torch.nn.functional as F
sys.path.insert(0, "/workspace/laser-asnc")
from transformers import AutoModelForCausalLM, AutoTokenizer
from asnc_modules import ASNCSoftmax

torch.manual_seed(0)
device = torch.device("cuda")
MODEL = "EleutherAI/pythia-6.9b"
LAYER_IDX = 15
K_SM = 256

print(f"loading {MODEL} (eager attention)", flush=True)
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(
    MODEL, torch_dtype=torch.float16, attn_implementation="eager",
).to(device).eval()
print(f"  attn impl: {model.config._attn_implementation}", flush=True)

# Spy post-softmax attention weights at layer LAYER_IDX only
in_attn = [False]
post_softmax = []

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

layer = model.gpt_neox.layers[LAYER_IDX]
h1 = layer.attention.register_forward_pre_hook(pre_attn, with_kwargs=True)
h2 = layer.attention.register_forward_hook(post_attn, with_kwargs=True)
F.softmax = sm_spy

ids = tok("The quick brown fox jumps over the lazy dog. " * 400, return_tensors="pt").input_ids[:, :1024].to(device)
print("running 8 forward passes", flush=True)
with torch.no_grad():
    for b in range(8):
        _ = model(ids)
F.softmax = orig_softmax
h1.remove(); h2.remove()

print(f"captured {len(post_softmax)} chunks, total samples = {sum(x.numel() for x in post_softmax)}", flush=True)
if not post_softmax:
    print("NO samples captured — eager path didn't trigger F.softmax", flush=True)
    sys.exit(1)

sm = torch.cat(post_softmax)
print(f"  shape {sm.shape}  range [{sm.min():.6e}, {sm.max():.6e}]  mean {sm.mean():.4e}", flush=True)

# free model
del model, layer
torch.cuda.empty_cache()

# Fit ASNCSoftmax
asnc_sm = ASNCSoftmax(K=K_SM).to(device)
asnc_sm.fit(sm)
with torch.no_grad():
    sm_d = sm.to(device)
    t = asnc_sm.thresholds.float()
    y = asnc_sm.y.float()
    idx = torch.bucketize(sm_d, t).clamp(0, asnc_sm.K - 1)
    rec = y[idx]
    mse = (rec - sm_d).pow(2).mean().item()
    maxerr = (rec - sm_d).abs().max().item()
print(f"\nASNCSoftmax (K={K_SM}):", flush=True)
print(f"  MSE = {mse:.3e}   max|err| = {maxerr:.4e}", flush=True)
print(f"  thresholds[:5]: {t[:5].tolist()}", flush=True)
print(f"  thresholds[-5:]: {t[-5:].tolist()}", flush=True)
