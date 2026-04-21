"""LLaMA PPL eval that loads pre-fitted codec state. Matches fit_codecs_llama
exactly (laser_eager with fp32 intermediate + DCR + ASNC Softmax)."""
import argparse, json, math, time, torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm.auto import tqdm
from asnc_modules import (
    make_asnc_silu, ASNCLayerNorm, ASNCSoftmax,
    bse_quantize_linears, int16_per_token_quant,
)

MAX_LEN = 2048
STRIDE = 512


def laser_eager_attention_forward(module, query, key, value, attention_mask,
                                  scaling, dropout=0.0, **kwargs):
    orig_dtype = query.dtype
    # GQA repeat
    n_rep = query.shape[1] // key.shape[1]
    if n_rep > 1:
        key = key.repeat_interleave(n_rep, dim=1)
        value = value.repeat_interleave(n_rep, dim=1)

    dcr_on = getattr(module, "_dcr_on", False)
    softmax_on = hasattr(module, "asnc_softmax") and module.asnc_softmax.fitted

    if dcr_on:
        q = int16_per_token_quant(query).float()
        k = int16_per_token_quant(key).float()
        v = int16_per_token_quant(value).float()
    else:
        q, k, v = query.float(), key.float(), value.float()

    attn = torch.matmul(q, k.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal = attention_mask[:, :, :, : k.shape[-2]].float()
        attn = attn + causal

    if softmax_on:
        attn = module.asnc_softmax(attn, dim=-1)
    else:
        attn = F.softmax(attn, dim=-1)

    attn = F.dropout(attn, p=dropout, training=module.training)
    out = torch.matmul(attn, v)
    out = out.transpose(1, 2).contiguous().to(orig_dtype)
    return out, attn.to(orig_dtype)


def patch_llama_eager():
    from transformers.models.llama import modeling_llama as m
    m.eager_attention_forward = laser_eager_attention_forward
    try:
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
        ALL_ATTENTION_FUNCTIONS["eager"] = laser_eager_attention_forward
    except Exception:
        pass


def _load_asnc_act(state_layer, K, K_tail=0, device=None):
    t, y = state_layer
    m = make_asnc_silu(K=K, K_tail=K_tail)
    m.thresholds = t.clone()
    m.y = y.clone()
    m.fitted = True
    if device is not None:
        m.to(device)
    return m


def _load_asnc_ln(base_ln, state_layer, K, K_tail=0):
    t, y = state_layer
    m = ASNCLayerNorm(base_ln, K=K, K_tail=K_tail)
    m.thresholds = t.clone()
    m.y = y.clone()
    m.fitted = True
    m.hidden = t.shape[0]
    # move to same device as base_ln
    try:
        dev = next(base_ln.parameters()).device
    except StopIteration:
        dev = next(base_ln.buffers()).device
    m.to(dev)
    return m


def _load_asnc_sm(state_layer, K, device=None):
    t, y = state_layer
    m = ASNCSoftmax(K=K)
    m.thresholds.data = t.clone()
    m.y.data = y.clone()
    m.fitted = True
    if device is not None:
        m.to(device)
    return m


def install_from_codecs(model, state, use_act, use_ln, use_sm, use_dcr):
    meta = state["meta"]
    K_act = meta["K_act"]; K_ln = meta["K_ln"]; K_sm = meta["K_sm"]
    K_tail_act = meta.get("K_tail_act", 0); K_tail_ln = meta.get("K_tail_ln", 0)
    layers = model.model.layers
    for i, layer in enumerate(layers):
        ly_dev = next(layer.parameters()).device
        if use_act and i in state["pre_silu"]:
            layer.mlp.act_fn = _load_asnc_act(
                state["pre_silu"][i], K_act, K_tail_act, device=ly_dev)
        if use_ln:
            if i in state["ln_in"]:
                layer.input_layernorm = _load_asnc_ln(
                    layer.input_layernorm, state["ln_in"][i], K_ln, K_tail_ln)
            if i in state["ln2_in"]:
                layer.post_attention_layernorm = _load_asnc_ln(
                    layer.post_attention_layernorm, state["ln2_in"][i], K_ln, K_tail_ln)
        if use_sm and i in state["post_softmax"]:
            layer.self_attn.asnc_softmax = _load_asnc_sm(
                state["post_softmax"][i], K_sm, device=ly_dev)
        layer.self_attn._dcr_on = use_dcr
    return model


@torch.no_grad()
def compute_ppl(model, input_ids, first_device):
    seq_len = input_ids.size(1)
    nlls, total = [], 0
    prev_end = 0
    for begin in tqdm(range(0, seq_len, STRIDE), desc="ppl", leave=False):
        end = min(begin + MAX_LEN, seq_len)
        trg_len = end - prev_end
        ids = input_ids[:, begin:end].to(first_device)
        tgt = ids.clone(); tgt[:, :-trg_len] = -100
        out = model(ids, labels=tgt)
        t = trg_len - 1 if trg_len > 1 else 1
        nlls.append(out.loss.float() * t); total += t
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
    ap.add_argument("--device_map", default="auto")
    ap.add_argument("--max_memory_per_gpu", default=None)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    kwargs = dict(torch_dtype=torch.float16, attn_implementation="eager",
                  device_map=args.device_map)
    if args.max_memory_per_gpu:
        n = torch.cuda.device_count()
        kwargs["max_memory"] = {i: args.max_memory_per_gpu for i in range(n)}
    model = AutoModelForCausalLM.from_pretrained(args.model, **kwargs).eval()
    first_device = next(model.parameters()).device

    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(ds["text"])
    enc = tok(text, return_tensors="pt").input_ids

    state = torch.load(args.codecs, weights_only=False, map_location="cpu")
    patch_llama_eager()

    if args.use_bse:
        print("[BSE] quantising weights", flush=True)
        bse_quantize_linears(model)

    print(f"[LASER] installing codecs (act={args.use_act} ln={args.use_ln} sm={args.use_sm} dcr={args.use_dcr})", flush=True)
    install_from_codecs(model, state, args.use_act, args.use_ln, args.use_sm, args.use_dcr)

    print("[Full LASER] running PPL", flush=True)
    t0 = time.time()
    ppl = compute_ppl(model, enc, first_device)
    dt = time.time() - t0
    delta = ppl - args.fp16_ppl
    print(f"  Full LASER PPL = {ppl:.4f}  Δ = {delta:+.4f}  ({dt:.0f}s)", flush=True)

    res = dict(
        model=args.model, fp16_ppl=args.fp16_ppl,
        full_laser_ppl=float(ppl), delta_ppl=float(delta), laser_sec=float(dt),
        K_activation=state["meta"]["K_act"], K_softmax=state["meta"]["K_sm"],
        K_layernorm=state["meta"]["K_ln"],
        K_tail_act=state["meta"].get("K_tail_act", 0),
        K_tail_ln=state["meta"].get("K_tail_ln", 0),
        use_act=args.use_act, use_ln=args.use_ln,
        use_sm=args.use_sm, use_dcr=args.use_dcr, use_bse=args.use_bse,
    )
    with open(args.result, "w") as f:
        json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
