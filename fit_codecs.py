"""Layer-by-layer component alignment.

Flow (one layer at a time, peak CPU RAM = one layer of samples):
  for each layer i in model:
    1. register hooks on layer i (pre-GeLU, LN1-in, LN2-in, post-softmax)
    2. forward on CALIB tokens
    3. fit ASNC codecs on captured samples
    4. save thresholds+y to disk (tiny: K*hidden*8 bytes each)
    5. discard samples & remove hooks

No layer accumulates across others — 6.9b/12b fit in CPU RAM trivially.

Run once per model:  python fit_codecs.py --model EleutherAI/pythia-6.9b --out codecs_pythia-6.9b.pt
Pipeline then loads codecs from disk and only measures PPL.
"""
import argparse, os, time, torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from asnc_modules import (
    make_asnc_gelu, ASNCLayerNorm, ASNCSoftmax,
    bse_quantize_linears, int16_per_token_quant,
)


def calib_eager_attention_forward(module, query, key, value, attention_mask,
                                   scaling, dropout=0.0, head_mask=None, **kwargs):
    """Calibration-time eager attention: matches the EVAL pipeline's
    `laser_eager_attention_forward` exactly — fp32 intermediates + DCR
    (per-token INT16 Q/K/V). This ensures the post-softmax / LN-input /
    pre-GeLU activations seen during calibration are from the SAME
    distribution that the quantised inference path sees. No distribution
    drift between fit and eval."""
    orig_dtype = query.dtype
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
    attn = F.softmax(attn, dim=-1)  # F.softmax is what our spy wraps
    if head_mask is not None:
        attn = attn * head_mask.float()
    attn = F.dropout(attn, p=dropout, training=module.training)
    out = torch.matmul(attn, v)
    out = out.transpose(1, 2).contiguous().to(orig_dtype)
    return out, attn.to(orig_dtype)


def patch_gpt_neox_calib_eager():
    from transformers.models.gpt_neox import modeling_gpt_neox as m
    m.eager_attention_forward = calib_eager_attention_forward
    try:
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
        ALL_ATTENTION_FUNCTIONS["eager"] = calib_eager_attention_forward
    except Exception:
        pass


def _flat_rows(t):
    return t.reshape(-1, t.shape[-1]).detach().float().cpu()


@torch.no_grad()
def fit_all_layers(model, tokens, K_act, K_ln, K_sm, n_calib_rows):
    """Fit codecs layer-by-layer and return a dict of state tensors."""
    layers = model.gpt_neox.layers
    L = len(layers)
    H = model.config.hidden_size
    device = next(model.parameters()).device

    # Storage for codec state (CPU)
    state = {
        "pre_gelu": {},      # layer -> (thresholds [4H, K-1], y [4H, K])
        "ln1_in": {},
        "ln2_in": {},
        "post_softmax": {},  # layer -> (thresholds [K-1], y [K])
    }

    total_t0 = time.time()
    for i in range(L):
        t0 = time.time()
        # ------ capture ONE layer only ------
        pg_buf, ln1_buf, ln2_buf, sm_buf = [], [], [], []
        pg_need = [n_calib_rows]
        ln1_need = [n_calib_rows]
        ln2_need = [n_calib_rows]
        sm_need = [200_000]

        def h_pg(m, a, out):
            if pg_need[0] > 0:
                r = _flat_rows(out)[:pg_need[0]]
                pg_buf.append(r); pg_need[0] -= r.shape[0]

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
            ly.mlp.dense_h_to_4h.register_forward_hook(h_pg),
            ly.input_layernorm.register_forward_pre_hook(h_ln1, with_kwargs=True),
            ly.post_attention_layernorm.register_forward_pre_hook(h_ln2, with_kwargs=True),
            ly.attention.register_forward_pre_hook(pre_attn, with_kwargs=True),
            ly.attention.register_forward_hook(post_attn, with_kwargs=True),
        ]

        # forward through ALL calib batches (no early exit) so every layer
        # gets the full budget even if a hook under-captures.
        B, T = tokens.shape
        for b in range(B):
            _ = model(tokens[b:b+1].to(device), use_cache=False)

        F.softmax = orig_softmax
        for h in hs: h.remove()

        pg = torch.cat(pg_buf) if pg_buf else torch.empty(0, 0)
        ln1 = torch.cat(ln1_buf) if ln1_buf else torch.empty(0, 0)
        ln2 = torch.cat(ln2_buf) if ln2_buf else torch.empty(0, 0)
        sm = torch.cat(sm_buf) if sm_buf else None
        print(f"  layer {i:3d}: pg_chunks={len(pg_buf)} ln1_chunks={len(ln1_buf)} "
              f"pg={tuple(pg.shape)} ln1={tuple(ln1.shape)} "
              f"sm={None if sm is None else sm.shape[0]}",
              flush=True)

        # ------ fit codecs for layer i ------
        def _fit_act(samples):
            m = make_asnc_gelu(K=K_act).to(device)
            m.fit(samples)
            return m.thresholds.detach().cpu(), m.y.detach().cpu()

        def _fit_ln(samples):
            # dummy base_ln; we only want the codec params
            base = torch.nn.LayerNorm(samples.shape[-1]).to(device)
            m = ASNCLayerNorm(base, K=K_ln).to(device)
            m.fit(samples)
            return m.thresholds.detach().cpu(), m.y.detach().cpu()

        def _fit_sm(samples):
            m = ASNCSoftmax(K=K_sm).to(device)
            m.fit(samples)
            return m.thresholds.detach().cpu(), m.y.detach().cpu()

        state["pre_gelu"][i] = _fit_act(pg)
        state["ln1_in"][i] = _fit_ln(ln1)
        state["ln2_in"][i] = _fit_ln(ln2)
        if sm is not None and sm.numel() > 0:
            state["post_softmax"][i] = _fit_sm(sm)
        del pg, ln1, ln2, sm, pg_buf, ln1_buf, ln2_buf, sm_buf
        torch.cuda.empty_cache()

        print(f"  layer {i+1:3d}/{L}  fit in {time.time()-t0:.1f}s  (cumulative {time.time()-total_t0:.0f}s)", flush=True)

    state["meta"] = dict(
        K_act=K_act, K_ln=K_ln, K_sm=K_sm,
        hidden=H, num_layers=L, model=type(model).__name__,
    )
    return state


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--K_act", type=int, default=1024)
    ap.add_argument("--K_ln", type=int, default=1024)
    ap.add_argument("--K_sm", type=int, default=256)
    ap.add_argument("--n_rows", type=int, default=8192, help="calib rows per layer")
    ap.add_argument("--seq_len", type=int, default=1024)
    ap.add_argument("--n_batches", type=int, default=16)
    args = ap.parse_args()

    print(f"Loading {args.model}", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    patch_gpt_neox_calib_eager()    # must happen before model instantiation
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16,
        attn_implementation="eager",  # force eager so F.softmax spy works
    ).to("cuda").eval()

    print("Loading wikitext-2 train tokens", flush=True)
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    text = "\n\n".join(ds["text"])
    enc = tok(text, return_tensors="pt", truncation=False).input_ids[0]
    print(f"  raw enc.numel()={enc.numel()}", flush=True)
    total = args.n_batches * args.seq_len
    if enc.numel() < total:
        raise RuntimeError(f"wikitext-2 tokenised to only {enc.numel()} tokens < required {total}")
    tokens = enc[:total].contiguous().view(args.n_batches, args.seq_len)
    print(f"  tokens.shape={tuple(tokens.shape)}, L={model.config.num_hidden_layers}", flush=True)

    state = fit_all_layers(
        model, tokens,
        K_act=args.K_act, K_ln=args.K_ln, K_sm=args.K_sm,
        n_calib_rows=args.n_rows,
    )

    torch.save(state, args.out)
    print(f"saved codecs to {args.out}", flush=True)


if __name__ == "__main__":
    main()
