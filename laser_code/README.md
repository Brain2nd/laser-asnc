# LASER: A High-Fidelity Spike Representation SNN Framework With Surrogate-Free Training

Reproducibility artifacts for the anonymous NeurIPS 2026 submission.
This repository contains the source code and experiment scripts used to
produce the tables and figures in the paper. All random seeds are fixed,
all calibration data is drawn from the public WikiText-2 train split, and
all model checkpoints are downloaded from public HuggingFace mirrors.

---

## 1 · Directory Layout

```
laser_code/
├── src/                                   Core implementation
│   ├── asnc_modules.py                    BSE / ASNC / DCR simulation modules
│   └── demo_bse_int16_equivalence.py      BSE↔INT16 bit-level equivalence demo
├── experiments/                           End-to-end experiment scripts
│   ├── full_laser_pythia.py               Full-LASER PPL on Pythia (BSE+ASNC+DCR)
│   ├── pythia_ppl.py                      BSE-only PPL across all Pythia sizes
│   ├── llama_ppl.py                       BSE-only PPL on LLaMA-2 7B
│   ├── llama70b_ppl.py                    BSE-only PPL on LLaMA-2 70B
│   ├── llama_ablation.py                  Layer-wise + component ablations
│   ├── baselines.py                       Rate / TTFS / ReLU / Uniform baselines
│   ├── asnc_distortion.py                 D_K vs K scaling for SiLU / Softmax
│   ├── delta_min.py                       δ_min + exceed-rate per module
│   ├── linear_fidelity.py                 FFN single-layer reconstruction MSE
│   ├── encoding_fidelity.py               BSE/Rate/TTFS reconstruction MSE
│   ├── gaussian_ks.py                     Activation-distribution KS test
│   ├── calibration_stability.py           Cross-set (λ, μ) stability
│   ├── capture_activations.py             Hook-based calibration sampler
│   ├── capture_softmax.py                 Pre-softmax score sampler (Q·Kᵀ)
│   ├── spectral_*.py                      Spectral-norm product measurements
│   ├── lipschitz_*.py                     Empirical-Lipschitz measurements
│   ├── accuracy.py                        lm-eval MMLU/HellaSwag/ARC/TruthfulQA
│   └── test_asnc_codecs.py                Unit tests for ASNC codec accuracy
├── scripts/
│   ├── run_full_laser_all_pythia.sh       Sequential batch: all 8 Pythia sizes
│   ├── run_component_isolation.sh         Per-component ΔPPL ablation
│   └── download_70b.py                    Robust 70B snapshot downloader
└── results/
    ├── results_full_laser_pythia-*.json   Per-model Full-LASER ΔPPL
    ├── results_asnc.json                  D_K vs K curves
    ├── results_llama.json                 LLaMA-2 7B BSE-only
    ├── results_llama70b.json              LLaMA-2 70B BSE-only
    ├── results_llama_ablation.json        16 ablation points on LLaMA-2 7B
    ├── results_baselines.json             Rate/TTFS/Uniform baselines
    ├── spectral_*.json                    Spectral-norm measurements
    ├── lipschitz_*.json                   Empirical-Lipschitz measurements
    └── ALL_RESULTS.json                   Machine-readable aggregate of all above
```

## 2 · Environment

The experiments were run inside a CUDA 13 container with the following
package versions:

```
python          3.12
torch           2.9.1 + cu130
transformers    4.56.1
datasets        4.3.0
huggingface_hub 0.26.x
lm-eval         0.4.11
numpy / scipy   standard
```

A conda environment can be recreated with:

```bash
conda create -n laser python=3.12 -y
conda activate laser
pip install torch --index-url https://download.pytorch.org/whl/cu130
pip install transformers==4.56.1 datasets==4.3.0 tqdm scipy lm-eval
```

Hardware: a single NVIDIA-class device with ≥16 GB VRAM is sufficient for
all Pythia-70M – 12B and LLaMA-2 7B experiments. LLaMA-2 70B requires
~140 GB of aggregate VRAM (four 48 GB cards with HuggingFace
`device_map="auto"`).

## 3 · BSE ≡ INT16 Equivalence

`src/demo_bse_int16_equivalence.py` is a 160-line self-contained script
that proves the core identity used throughout our experiments:

```
IF-neuron(BSE-encode(x))  ≡  INT16-dequantize(INT16-quantize(x))
```

on 10,000 randomly drawn FP values and on a 512→128 linear layer.
`torch.equal(·)` returns `True`, i.e. every reconstructed bit is
identical. Consequently, simulating BSE with per-channel INT16
fake-quantisation introduces **zero** additional error; it is purely a
throughput optimisation (one matrix multiply instead of N time-stepped
spike accumulations).

Run:

```bash
python src/demo_bse_int16_equivalence.py
```

## 4 · Reproducing the main tables

### 4.1 End-to-end PPL (Full-LASER) on the Pythia suite

```bash
bash scripts/run_full_laser_all_pythia.sh
```

This sequentially evaluates `70m, 160m, 410m, 1b, 1.4b, 2.8b, 6.9b, 12b`
with `K_SiLU=32`, `K_Softmax=64`, `K_LN=24`, per-channel codec fitting,
and DCR on the activation-activation products. Each run writes
`results/results_full_laser_pythia-<size>.json`.

### 4.2 Layer-wise / component ablations on LLaMA-2 7B

```bash
python experiments/llama_ablation.py
```

Produces `results/results_llama_ablation.json` with 16 rows (Full,
FFN-only, Attn-only, Embed-only, LN-only, five progressive SNN-isation
steps, five attention components).

### 4.3 Non-linear diagnostic tables

Activation capture must be run once; the other scripts are offline:

```bash
python experiments/capture_activations.py      # stores ./activations/
python experiments/capture_softmax.py          # pre-softmax scores
python experiments/gaussian_ks.py              # KS test → results_ks.json
python experiments/asnc_distortion.py          # D_K curves
python experiments/delta_min.py                # δ_min + exceed rate
python experiments/linear_fidelity.py          # FFN single-layer MSE
python experiments/encoding_fidelity.py        # random-value BSE / Rate / TTFS MSE
```

### 4.4 Spectral-norm products (Appendix B)

```bash
python experiments/spectral_bare.py            # bare Π‖W‖
python experiments/spectral_diag.py            # γ-absorbed per-layer diagnostic
python experiments/spectral_llama.py           # LLaMA-2 7B variants
python experiments/spectral_3perblock.py       # M=3L pure-W composite
python experiments/lipschitz.py                # M=L empirical Lipschitz
python experiments/lipschitz_llama.py          # LLaMA-2 7B empirical Lipschitz
python experiments/lipschitz_3perblock.py      # M=3L empirical Lipschitz
```

### 4.5 LLaMA-2 70B (large-memory)

```bash
python scripts/download_70b.py                 # resumable snapshot downloader
python experiments/llama70b_ppl.py             # requires device_map="auto"
```

## 5 · Key numerical results (from `results/ALL_RESULTS.json`)

| Model | FP16 PPL | BSE-only ΔPPL | Full-LASER ΔPPL |
|-------|---------:|--------------:|----------------:|
| Pythia-70M  | 40.89 | +0.230 | −0.017 |
| Pythia-160M | 23.50 | +0.031 | +0.029 |
| Pythia-410M | 13.98 | +0.191 | +0.007 |
| Pythia-1B   | 11.61 | +0.110 | +0.0005 |
| Pythia-1.4B | 10.43 | +0.959 | −0.008 |
| Pythia-2.8B |  9.00 | +0.026 | +0.002 |
| Pythia-6.9B |  8.23 | +0.134 |  (running — see results/) |
| Pythia-12B  |  7.60 | +0.014 |  (running — see results/) |
| LLaMA-2 7B  |  4.86 | +0.000 |  (pending) |
| LLaMA-2 70B |  2.96 | +6e-6  |  (pending) |

Full-LASER absorbs the BSE-only outlier at Pythia-1.4B (+0.96 → −0.008),
confirming ASNC's universal-approximation role in the overall pipeline.

## 6 · Pipeline correspondence

| Component in paper | Implementation in this repo |
|--------------------|-----------------------------|
| BSE (Bit Spike Encoder) | `asnc_modules.bse_quantize_linears`, mathematically identical to per-channel INT16 fake-quant (proved in §3 above) |
| ASNC (Adaptive Spiking Neural Codec) | `asnc_modules.ASNCActivation` (SiLU/GeLU), `ASNCSoftmax`, `ASNCLayerNorm` — all per-channel Bennett-optimal K-level codecs fitted from calibration activations |
| DCR (Decode-Compute-Reencode) | `asnc_modules.int16_per_token_quant` applied to Q, K, V inside the patched `eager_attention_forward` |
| Calibration | `experiments/capture_activations.py` + `experiments/capture_softmax.py` |

## 7 · Licensing / Data sources

* Model weights: `NousResearch/Llama-2-*-hf`, `EleutherAI/pythia-*` (all public).
* Dataset: `wikitext-2-raw-v1`, `c4` (subset), `openwebtext-10k`.
* Code: released under Apache 2.0 (see `LICENSE`).

No personally identifying information is embedded in the code, data, or
result files.
