"""Load pythia-70m, capture pre-GeLU and pre-LN activations for one layer,
fit ASNCActivation / ASNCLayerNorm, then measure reconstruction quality on
the SAME calibration samples. If MSE is low, the codec is fine — regression
comes from distribution shift at inference. If MSE is high, codec is broken."""
import sys, torch, torch.nn.functional as F
sys.path.insert(0, "/home/dgxspark/Desktop/A2S")
from transformers import AutoModelForCausalLM, AutoTokenizer
from asnc_modules import make_asnc_gelu, ASNCLayerNorm, gelu_fn

torch.manual_seed(0)
device = torch.device("cuda")

print("loading pythia-70m", flush=True)
tok = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m")
model = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-70m", torch_dtype=torch.float16).to(device).eval()

# capture pre-GeLU and ln1_in at layer 0
pre_gelu, ln1_in = [], []
layer0 = model.gpt_neox.layers[0]
h1 = layer0.mlp.dense_h_to_4h.register_forward_hook(
    lambda m, a, out: pre_gelu.append(out.detach().float().reshape(-1, out.shape[-1]).cpu())
)
def pre_hook(m, args, kwargs):
    x = args[0] if args else kwargs.get("hidden_states")
    ln1_in.append(x.detach().float().reshape(-1, x.shape[-1]).cpu())
h2 = layer0.input_layernorm.register_forward_pre_hook(pre_hook, with_kwargs=True)

# calibration sweep
ids = tok("The quick brown fox jumps over the lazy dog. " * 200, return_tensors="pt").input_ids[:, :1024].to(device)
with torch.no_grad():
    for _ in range(8):
        model(ids)
h1.remove(); h2.remove()

pg = torch.cat(pre_gelu)[:50000]
ln = torch.cat(ln1_in)[:50000]
print(f"pre_gelu: {pg.shape}, dtype={pg.dtype}")
print(f"ln1_in:   {ln.shape}, dtype={ln.dtype}")

# fit ASNCActivation on pre_gelu (GeLU approximation)
asnc = make_asnc_gelu(K=32)
asnc.fit(pg).to(device)
with torch.no_grad():
    pg_d = pg.to(device).half()
    gelu_exact = F.gelu(pg_d, approximate="none")
    gelu_asnc = asnc(pg_d)
    mse_gelu = (gelu_exact.float() - gelu_asnc.float()).pow(2).mean().item()
    max_gelu = (gelu_exact.float() - gelu_asnc.float()).abs().max().item()
print(f"GeLU codec MSE = {mse_gelu:.6e}  max|err| = {max_gelu:.4f}")

# fit ASNCLayerNorm on ln1_in
base_ln = torch.nn.LayerNorm(ln.shape[-1]).to(device).half()
# copy pythia LN params
base_ln.weight.data.copy_(layer0.input_layernorm.weight.data)
base_ln.bias.data.copy_(layer0.input_layernorm.bias.data)
asnc_ln = ASNCLayerNorm(base_ln, K=24).to(device)
asnc_ln.fit(ln)
with torch.no_grad():
    ln_d = ln.to(device).half()
    ln_exact = base_ln(ln_d)
    ln_asnc = asnc_ln(ln_d)
    mse_ln = (ln_exact.float() - ln_asnc.float()).pow(2).mean().item()
    max_ln = (ln_exact.float() - ln_asnc.float()).abs().max().item()
print(f"LN (ASNC input quant + exact LN) MSE = {mse_ln:.6e}  max|err| = {max_ln:.4f}")

# also check input quant error directly
with torch.no_grad():
    # manually run codec without final LN
    shape = ln_d.shape
    x2 = ln_d.view(-1, shape[-1]).contiguous()
    t = asnc_ln.thresholds.to(device=device, dtype=x2.dtype)
    y = asnc_ln.y.to(device=device, dtype=x2.dtype)
    idx = torch.searchsorted(t, x2.t().contiguous()).clamp_(0, asnc_ln.K - 1).t()
    x_q = y.gather(1, idx.t()).t().view(shape)
    input_mse = (ln_d.float() - x_q.float()).pow(2).mean().item()
    input_max = (ln_d.float() - x_q.float()).abs().max().item()
print(f"LN input-quant MSE = {input_mse:.6e}  max|err| = {input_max:.4f}")
print(f"    input range: [{ln_d.min().item():.3f}, {ln_d.max().item():.3f}]")
print(f"    thresholds[0]: {asnc_ln.thresholds[0].cpu().float().tolist()[:5]} ... {asnc_ln.thresholds[0].cpu().float().tolist()[-5:]}")
print(f"    y[0]:          {asnc_ln.y[0].cpu().float().tolist()[:5]} ... {asnc_ln.y[0].cpu().float().tolist()[-5:]}")
