"""Layer-by-layer ASNC fit for LLaMA (SiLU, RMSNorm, GQA), matched to
eval_ppl_llama. Same architecture as Pythia's fit_codecs.py.

Supports multi-GPU via device_map="auto" (for 70B)."""
import argparse, time, sys, torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm.auto import tqdm
from asnc_modules import (
    make_asnc_silu, ASNCLayerNorm, ASNCSoftmax,
    bse_quantize_linears, int16_per_token_quant,
)


def calib_eager_attention_forward(module, query, key, value, attention_mask,
                                   scaling, dropout=0.0, **kwargs):
    orig_dtype = query.dtype
    # GQA: repeat k/v if key has fewer heads than query
    n_rep = query.shape[1] // key.shape[1]
    if n_rep > 1:
        key = key.repeat_interleave(n_rep, dim=1)
        value = value.repeat_interleave(n_rep, dim=1)

    dcr_on = getattr(module, "_dcr_on_calib", True)
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
    attn = F.softmax(attn, dim=-1)
    attn = F.dropout(attn, p=dropout, training=module.training)
    out = torch.matmul(attn, v)
    out = out.transpose(1, 2).contiguous().to(orig_dtype)
    return out, attn.to(orig_dtype)


def patch_llama_calib_eager():
    from transformers.models.llama import modeling_llama as m
    m.eager_attention_forward = calib_eager_attention_forward
    try:
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
        ALL_ATTENTION_FUNCTIONS["eager"] = calib_eager_attention_forward
    except Exception:
        pass


def _flat_rows(t):
    return t.reshape(-1, t.shape[-1]).detach().float().cpu()


@torch.no_grad()
def fit_all_layers(model, tokens, first_device, K_act, K_ln, K_sm, n_calib_rows,
                    K_tail_act=0, K_tail_ln=0):
    layers = model.model.layers
    L = len(layers)
    state = {
        "pre_silu": {}, "ln_in": {}, "ln2_in": {}, "post_softmax": {},
    }

    total_t0 = time.time()
    for i in tqdm(range(L), desc="layers", dynamic_ncols=True, file=sys.stdout):
        t0 = time.time()
        ps_buf, ln1_buf, ln2_buf, sm_buf = [], [], [], []
        ps_need = [n_calib_rows]; ln1_need = [n_calib_rows]; ln2_need = [n_calib_rows]
        sm_need = [200_000]

        def h_ps(m, a, out):
            if ps_need[0] > 0:
                r = _flat_rows(out)[:ps_need[0]]
                ps_buf.append(r); ps_need[0] -= r.shape[0]

        def h_ln1(m, args, kwargs):
            if ln1_need[0] > 0:
                x = args[0] if args else kwargs.get("hidden_states")
                r = _flat_rows(x)[:ln1_need[0]]
                ln1_buf.append(r); ln1_need[0] -= r.shape[0]

        def h_ln2(m, args, kwargs):
            if ln2_need[0] > 0:
                x = args[0] if args else kwargs.get("hidden_states")
                r = _flat_rows(x)[:ln2_need[0]]
                ln2_buf.append(r); ln2_need[0] -= r.shape[0]

        in_attn = [False]
        def pre_attn(m, a, kw): in_attn[0] = True
        def post_attn(m, a, kw, out): in_attn[0] = False
        orig_softmax = F.softmax
        def sm_spy(*args, **kwargs):
            out = orig_softmax(*args, **kwargs)
            if in_attn[0] and sm_need[0] > 0:
                flat = out.detach().float().flatten()
                if flat.numel() > sm_need[0]:
                    flat = flat[torch.randperm(flat.numel(), device=flat.device)[:sm_need[0]]]
                sm_buf.append(flat.cpu()); sm_need[0] -= flat.numel()
            return out
        F.softmax = sm_spy

        ly = layers[i]
        hs = [
            ly.mlp.gate_proj.register_forward_hook(h_ps),
            ly.input_layernorm.register_forward_pre_hook(h_ln1, with_kwargs=True),
            ly.post_attention_layernorm.register_forward_pre_hook(h_ln2, with_kwargs=True),
            ly.self_attn.register_forward_pre_hook(pre_attn, with_kwargs=True),
            ly.self_attn.register_forward_hook(post_attn, with_kwargs=True),
        ]

        B, T = tokens.shape
        for b in tqdm(range(B), desc=f"L{i:02d} fwd", leave=False,
                       dynamic_ncols=True, file=sys.stdout):
            _ = model(tokens[b:b+1].to(first_device), use_cache=False)

        F.softmax = orig_softmax
        for h in hs: h.remove()

        ps = torch.cat(ps_buf) if ps_buf else torch.empty(0, 0)
        ln1 = torch.cat(ln1_buf) if ln1_buf else torch.empty(0, 0)
        ln2 = torch.cat(ln2_buf) if ln2_buf else torch.empty(0, 0)
        sm = torch.cat(sm_buf) if sm_buf else None
        print(f"  layer {i:3d}: ps={tuple(ps.shape)} ln1={tuple(ln1.shape)} "
              f"sm={None if sm is None else sm.shape[0]}", flush=True)

        device = torch.device("cuda")
        def _fit_act(samples):
            m = make_asnc_silu(K=K_act, K_tail=K_tail_act).to(device)
            m.fit(samples)
            return m.thresholds.detach().cpu(), m.y.detach().cpu()

        def _fit_ln(samples):
            base = torch.nn.LayerNorm(samples.shape[-1]).to(device)
            m = ASNCLayerNorm(base, K=K_ln, K_tail=K_tail_ln).to(device)
            m.fit(samples)
            return m.thresholds.detach().cpu(), m.y.detach().cpu()

        def _fit_sm(samples):
            m = ASNCSoftmax(K=K_sm).to(device)
            m.fit(samples)
            return m.thresholds.detach().cpu(), m.y.detach().cpu()

        state["pre_silu"][i] = _fit_act(ps)
        state["ln_in"][i]    = _fit_ln(ln1)
        state["ln2_in"][i]   = _fit_ln(ln2)
        if sm is not None and sm.numel() > 0:
            state["post_softmax"][i] = _fit_sm(sm)
        del ps, ln1, ln2, sm, ps_buf, ln1_buf, ln2_buf, sm_buf
        torch.cuda.empty_cache()
        print(f"  layer {i+1:3d}/{L} fit in {time.time()-t0:.1f}s (cum {time.time()-total_t0:.0f}s)", flush=True)

    state["meta"] = dict(
        K_act=K_act, K_ln=K_ln, K_sm=K_sm,
        K_tail_act=K_tail_act, K_tail_ln=K_tail_ln,
        num_layers=L, model="LLaMA",
    )
    return state


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--K_act", type=int, default=4096)
    ap.add_argument("--K_ln", type=int, default=4096)
    ap.add_argument("--K_sm", type=int, default=512)
    ap.add_argument("--K_tail_act", type=int, default=64)
    ap.add_argument("--K_tail_ln", type=int, default=64)
    ap.add_argument("--n_rows", type=int, default=50000)
    ap.add_argument("--seq_len", type=int, default=1024)
    ap.add_argument("--n_batches", type=int, default=48)
    ap.add_argument("--device_map", default="auto")
    ap.add_argument("--max_memory_per_gpu", default=None,
                    help="e.g. '45GiB' for multi-GPU split")
    args = ap.parse_args()

    print(f"Loading {args.model}", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    patch_llama_calib_eager()
    kwargs = dict(torch_dtype=torch.float16, attn_implementation="eager",
                  device_map=args.device_map)
    if args.max_memory_per_gpu:
        n = torch.cuda.device_count()
        kwargs["max_memory"] = {i: args.max_memory_per_gpu for i in range(n)}
    model = AutoModelForCausalLM.from_pretrained(args.model, **kwargs).eval()
    first_device = next(model.parameters()).device

    print("[BSE] per-channel INT16 weight quantisation", flush=True)
    bse_quantize_linears(model)
    for layer in model.model.layers:
        layer.self_attn._dcr_on_calib = True

    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    text = "\n\n".join(ds["text"])
    enc = tok(text, return_tensors="pt", truncation=False).input_ids[0]
    total = args.n_batches * args.seq_len
    tokens = enc[:total].contiguous().view(args.n_batches, args.seq_len)
    print(f"  tokens.shape={tuple(tokens.shape)}, L={model.config.num_hidden_layers}", flush=True)

    state = fit_all_layers(
        model, tokens, first_device,
        K_act=args.K_act, K_ln=args.K_ln, K_sm=args.K_sm,
        n_calib_rows=args.n_rows,
        K_tail_act=args.K_tail_act, K_tail_ln=args.K_tail_ln,
    )
    torch.save(state, args.out)
    print(f"saved codecs to {args.out}", flush=True)


if __name__ == "__main__":
    main()
