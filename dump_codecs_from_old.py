"""Run the OLD exp_full_laser_pythia flow to fit codecs in-place, then dump the
fitted thresholds/y to a state dict that eval_ppl.py can load. Used to isolate
whether the save/load path is broken vs the fit path."""
import argparse, sys, torch, torch.nn.functional as F
sys.path.insert(0, "/workspace/laser-asnc")
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from asnc_modules import (
    ASNCActivation, ASNCLayerNorm, ASNCSoftmax,
    make_asnc_gelu, bse_quantize_linears, int16_per_token_quant,
)
# reuse the old script's functions by importing
sys.path.insert(0, ".")
from exp_full_laser_pythia import (
    capture_calibration, laser_eager_attention_forward, patch_gpt_neox_eager,
    install_asnc,
)
import exp_full_laser_pythia as E

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--out", required=True)
args = ap.parse_args()

device = torch.device("cuda")
tok = AutoTokenizer.from_pretrained(args.model)
model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16).to(device).eval()
ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
text = "\n\n".join(ds["text"])
enc_train = tok(text, return_tensors="pt").input_ids[0]
calib_total = E.CALIB_BATCHES * E.CALIB_SEQLEN
calib_ids = enc_train[:calib_total].view(E.CALIB_BATCHES, E.CALIB_SEQLEN).to(device)

# old calibration + fit
calib = capture_calibration(model, calib_ids)
install_asnc(model, calib)

# Extract codec state
state = {"pre_gelu": {}, "ln1_in": {}, "ln2_in": {}, "post_softmax": {}, "meta": {}}
layers = model.gpt_neox.layers
for i, layer in enumerate(layers):
    act = layer.mlp.act
    if isinstance(act, ASNCActivation) and act.fitted:
        state["pre_gelu"][i] = (act.thresholds.detach().cpu(), act.y.detach().cpu())
    ln1 = layer.input_layernorm
    if isinstance(ln1, ASNCLayerNorm) and ln1.fitted:
        state["ln1_in"][i] = (ln1.thresholds.detach().cpu(), ln1.y.detach().cpu())
    ln2 = layer.post_attention_layernorm
    if isinstance(ln2, ASNCLayerNorm) and ln2.fitted:
        state["ln2_in"][i] = (ln2.thresholds.detach().cpu(), ln2.y.detach().cpu())
    if hasattr(layer.attention, "asnc_softmax") and layer.attention.asnc_softmax.fitted:
        sm = layer.attention.asnc_softmax
        state["post_softmax"][i] = (sm.thresholds.detach().cpu(), sm.y.detach().cpu())

state["meta"] = dict(K_act=E.K_ACTIVATION, K_ln=E.K_LAYERNORM, K_sm=E.K_SOFTMAX,
                     hidden=model.config.hidden_size, num_layers=len(layers),
                     model=type(model).__name__)

torch.save(state, args.out)
print(f"saved {args.out}")
for k, v in state.items():
    if k == "meta": continue
    print(f"  {k}: {len(v)} layers, first-layer shapes="
          f"{tuple(v[next(iter(v))][0].shape) if v else None}")
