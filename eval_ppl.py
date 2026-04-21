"""PPL evaluation pipeline that LOADS pre-fitted codec state and only does
forward + PPL. No calibration inside the pipeline — zero CPU RAM accumulation.

Run:
  python eval_ppl.py \\
      --model EleutherAI/pythia-6.9b \\
      --codecs codecs_pythia-6.9b.pt \\
      --fp16_ppl 8.2290 \\
      --result results_full_laser_pythia-6.9b.json
"""
import argparse, json, math, time, torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm.auto import tqdm
from asnc_modules import (
    make_asnc_gelu, ASNCLayerNorm, ASNCSoftmax,
    bse_quantize_linears, int16_per_token_quant,
)

MAX_LEN = 2048
STRIDE = 512
DEVICE = "cuda"


def laser_eager_attention_forward(module, query, key, value, attention_mask,
                                  scaling, dropout=0.0, head_mask=None, **kwargs):
    orig_dtype = query.dtype
    dcr_on = getattr(module, "_dcr_on", False)
    softmax_on = hasattr(module, "asnc_softmax") and module.asnc_softmax.fitted

    if dcr_on:
        q_q = int16_per_token_quant(query).float()
        k_q = int16_per_token_quant(key).float()
        v_q = int16_per_token_quant(value).float()
    else:
        q_q, k_q, v_q = query.float(), key.float(), value.float()

    attn_weights = torch.matmul(q_q, k_q.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal = attention_mask[:, :, :, : k_q.shape[-2]].float()
        attn_weights = attn_weights + causal

    if softmax_on:
        attn_weights = module.asnc_softmax(attn_weights, dim=-1)
    else:
        attn_weights = F.softmax(attn_weights, dim=-1)

    if head_mask is not None:
        attn_weights = attn_weights * head_mask.float()
    attn_weights = F.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, v_q)
    attn_output = attn_output.transpose(1, 2).contiguous().to(orig_dtype)
    return attn_output, attn_weights.to(orig_dtype)


def patch_gpt_neox_eager():
    from transformers.models.gpt_neox import modeling_gpt_neox as m
    m.eager_attention_forward = laser_eager_attention_forward
    try:
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
        ALL_ATTENTION_FUNCTIONS["eager"] = laser_eager_attention_forward
    except Exception:
        pass


def _load_asnc_act(state_layer, K, K_tail=0):
    t, y = state_layer
    m = make_asnc_gelu(K=K, K_tail=K_tail)
    m.thresholds = t.clone()
    m.y = y.clone()
    m.fitted = True
    return m


def _load_asnc_ln(base_ln, state_layer, K, K_tail=0):
    t, y = state_layer
    m = ASNCLayerNorm(base_ln, K=K, K_tail=K_tail)
    m.thresholds = t.clone()
    m.y = y.clone()
    m.fitted = True
    m.hidden = t.shape[0]
    return m


def _load_asnc_sm(state_layer, K):
    t, y = state_layer
    m = ASNCSoftmax(K=K)
    m.thresholds.data = t.clone()
    m.y.data = y.clone()
    m.fitted = True
    return m


def install_from_codecs(model, state, use_act, use_ln, use_sm, use_dcr):
    meta = state["meta"]
    K_act, K_ln, K_sm = meta["K_act"], meta["K_ln"], meta["K_sm"]
    K_tail_act = meta.get("K_tail_act", 0)
    K_tail_ln  = meta.get("K_tail_ln", 0)
    layers = model.gpt_neox.layers
    for i, layer in enumerate(layers):
        if use_act and i in state["pre_gelu"]:
            layer.mlp.act = _load_asnc_act(state["pre_gelu"][i], K_act, K_tail_act).to(DEVICE)
        if use_ln:
            if i in state["ln1_in"]:
                layer.input_layernorm = _load_asnc_ln(
                    layer.input_layernorm, state["ln1_in"][i], K_ln, K_tail_ln).to(DEVICE)
            if i in state["ln2_in"]:
                layer.post_attention_layernorm = _load_asnc_ln(
                    layer.post_attention_layernorm, state["ln2_in"][i], K_ln, K_tail_ln).to(DEVICE)
        if use_sm and i in state["post_softmax"]:
            layer.attention.asnc_softmax = _load_asnc_sm(
                state["post_softmax"][i], K_sm).to(DEVICE)
        layer.attention._dcr_on = use_dcr
    return model


@torch.no_grad()
def compute_ppl(model, input_ids):
    seq_len = input_ids.size(1)
    nlls = []
    prev_end = 0
    total = 0
    for begin in tqdm(range(0, seq_len, STRIDE), desc="ppl", leave=False):
        end = min(begin + MAX_LEN, seq_len)
        trg_len = end - prev_end
        ids = input_ids[:, begin:end].to(DEVICE)
        tgt = ids.clone(); tgt[:, :-trg_len] = -100
        out = model(ids, labels=tgt)
        t = trg_len - 1 if trg_len > 1 else 1
        nlls.append(out.loss.float() * t)
        total += t
        prev_end = end
        if end == seq_len:
            break
    return math.exp(sum(nlls) / total)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--codecs", required=True)
    ap.add_argument("--result", required=True)
    ap.add_argument("--fp16_ppl", type=float, required=True)
    ap.add_argument("--use_act", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--use_ln",  action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--use_sm",  action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--use_dcr", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--use_bse", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--force_eager", action="store_true",
                    help="force attn_implementation=eager so laser_eager path is actually used")
    args = ap.parse_args()

    print(f"Loading {args.model}", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    # Match the old exp_full_laser_pythia path: load with default attn_impl
    # (sdpa on modern transformers). This means laser_eager is NOT actually
    # called during PPL — DCR / Softmax ASNC flags are inert. To keep them
    # active, pass --force_eager.
    kwargs = dict(torch_dtype=torch.float16)
    if args.force_eager:
        kwargs["attn_implementation"] = "eager"
    model = AutoModelForCausalLM.from_pretrained(args.model, **kwargs).to(DEVICE).eval()

    print("Loading wikitext-2 test tokens", flush=True)
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(ds["text"])
    enc = tok(text, return_tensors="pt").input_ids

    print(f"Loading codecs from {args.codecs}", flush=True)
    state = torch.load(args.codecs, weights_only=False, map_location="cpu")

    patch_gpt_neox_eager()

    if args.use_bse:
        print("[BSE] quantizing weights", flush=True)
        bse_quantize_linears(model)

    print(f"[LASER] installing codecs (act={args.use_act} ln={args.use_ln} sm={args.use_sm} dcr={args.use_dcr})", flush=True)
    install_from_codecs(
        model, state,
        use_act=args.use_act, use_ln=args.use_ln,
        use_sm=args.use_sm, use_dcr=args.use_dcr,
    )

    print("[Full LASER] running PPL", flush=True)
    t0 = time.time()
    ppl = compute_ppl(model, enc)
    dt = time.time() - t0
    delta = ppl - args.fp16_ppl
    print(f"  Full LASER PPL = {ppl:.4f}  Δ = {delta:+.4f}  ({dt:.0f}s)", flush=True)

    res = dict(
        model=args.model, fp16_ppl=args.fp16_ppl,
        full_laser_ppl=float(ppl), delta_ppl=float(delta),
        laser_sec=float(dt),
        K_activation=state["meta"]["K_act"],
        K_softmax=state["meta"]["K_sm"],
        K_layernorm=state["meta"]["K_ln"],
        use_act=args.use_act, use_ln=args.use_ln,
        use_sm=args.use_sm, use_dcr=args.use_dcr, use_bse=args.use_bse,
    )
    with open(args.result, "w") as f:
        json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
