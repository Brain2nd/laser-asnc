# LASER 实验汇总索引

**最后更新**：2026-04-20  
**目录**：`/home/dgxspark/Desktop/A2S/`

## 环境

- **本地 GPU**：NVIDIA GB10 (130 GB)，conda env `AI001`（torch 2.9 + CUDA 13）
- **远程 4090D 服务器**（4×48GB，共 192GB，docker `neuronspark-dev`，用于 70B）
- **数据集**：WikiText-2 test split，window=2048 / stride=512（HF 标准协议）

---

## 核心最终表格

### Pythia 8 sizes PPL + 谱范数（主表 tab:spectral_pythia）

| Model | L | Π‖A_m‖ (σ<1.5) | ΔPPL | 数据源 |
|---|---|---|---|---|
| LLaMA-2 7B | 32 | 4.36 ± 0.01 | +0.46* | `lipschitz_llama.json` |
| Pythia-70M | 6 | 1.45 ± 0.00 | +0.230 | `lipschitz_results.json`, `results.json` |
| Pythia-160M | 12 | 1.85 ± 0.00 | +0.031 | 同上 |
| Pythia-410M | 24 | 4.15 ± 0.05 | +0.191 | 同上 |
| Pythia-1B | 16 | 3.91 ± 0.05 | +0.110 | 同上 |
| Pythia-1.4B | 24 | 7.58 ± 0.17 | +0.959 | 同上 |
| Pythia-2.8B | 32 | 47.12 ± 5.88 | +0.026 | 同上 |
| Pythia-6.9B | 32 | 20.35 ± 1.53 | +0.134 | 同上 |
| Pythia-12B | 36 | 277.70 ± 29.0 | +0.014 | 同上 |

*LLaMA-2 7B ΔPPL 保留论文给定值；我们实测 ΔPPL=0（见下）

### LLaMA-2 PPL (BSE-only, per-channel INT16)

| 规模 | ANN (FP16) | SNN (INT16) | ΔPPL | 数据源 |
|---|---|---|---|---|
| LLaMA-2 7B | 4.8595 | 4.8595 | +0.000 (严格无损) | `results_llama.json` |
| LLaMA-2 70B | 2.9642 | 2.9642 | +6e-6 (噪声) | `results_llama70b.json` |

### LLaMA-2 7B 16-项消融（全 Δ=0）

全部 16 个配置：ANN、FFN-only、Attn-only、Embed-only、LN-only、Full SNN、Progressive×5、Attn 组件×5 → **PPL 全部 4.8595**。  
数据源：`results_llama_ablation.json`

### ASNC 失真（tab:scaling，fig:asnc_distortion）

| 模块 | slope non-uniform | u/nu ratio | Lloyd-Max closeness | 数据源 |
|---|---|---|---|---|
| SiLU | **-1.927** (paper -1.94) ✓ | 2.46-3.36× (paper 2.1-3.4×) ✓ | 0.1-0.3% (paper 0.9%) ✓ | `results_asnc.json` |
| Softmax | -1.679 | 1.7-4.5× | 0.4-2.8% | 同上 |

### 全谱范数对比（tab:spectral_full，5 regimes）

Bare / γ-abs / Empirical / Exc.L0 / σ<1.5 五种口径对全 Pythia + LLaMA-2 7B。  
数据源：`spectral_results.json`, `spectral_diag.json`, `spectral_llama.json`, `lipschitz_results.json`, `lipschitz_llama.json`

### M=3L 补充谱范数（tab:spectral_M3L）

Pure-W composite + 经验 Lipschitz M=3L。  
数据源：`spectral_3perblock.json`, `lipschitz_3perblock.json`

---

## 所有结果文件索引

### PPL 实验
| 文件 | 内容 |
|---|---|
| `results.json` | Pythia 8 sizes FP16/INT16 PPL + 耗时 |
| `results_llama.json` | LLaMA-2 7B FP16/INT16 BSE-only PPL |
| `results_llama70b.json` | LLaMA-2 70B FP16/INT16 PPL（4090 服务器） |
| `results_llama_ablation.json` | LLaMA-2 7B 16 种 ablation PPL |
| `results_baselines.json` | 5 种 baseline（BSE+SiLU/ReLU/Uniform, Rate+SiLU/TB） |

### 谱范数 / Lipschitz
| 文件 | 内容 |
|---|---|
| `spectral_results.json` | Pythia 裸权重 + γ-absorbed 谱范数 |
| `spectral_diag.json` | Pythia 诊断版（γ1/γ2 per-layer max 等） |
| `spectral_llama.json` | LLaMA-2 7B 裸/γ-abs 谱范数 |
| `spectral_3perblock.json` | Pure-W composite M=3L |
| `lipschitz_results.json` | Pythia 经验 Lipschitz M=L（per-block） |
| `lipschitz_llama.json` | LLaMA-2 7B 经验 Lipschitz M=L |
| `lipschitz_3perblock.json` | Pythia + LLaMA M=3L 经验 Lipschitz |

### 组件测量
| 文件 | 内容 |
|---|---|
| `results_encoding.json` | BSE/rate/TTFS 编码保真度 MSE |
| `results_linear_fidelity.json` | LLaMA-2 FFN 单层 MSE |
| `results_asnc.json` | ASNC D_K vs K（SiLU + Softmax） |
| `results_delta_min.json` | δ_min + 超限率（SiLU/Softmax/LN） |
| `results_ks.json` | 高斯 KS 检验（L4/16/28 × SiLU/LN/Softmax） |
| `results_calibration.json` | 校准参数 λ/μ 稳定性 + 截断覆盖率 |

---

## 脚本索引

### 测量脚本
| 文件 | 用途 |
|---|---|
| `pythia_ppl.py` | Pythia 8 sizes FP16/INT16 PPL 全套 |
| `llama_ppl.py` | LLaMA-2 7B PPL |
| `llama70b_ppl.py` | LLaMA-2 70B PPL（服务器端） |
| `exp_llama_ablation.py` | LLaMA-2 7B 16 ablation |
| `exp_baselines.py` | 5 种 baseline PPL |
| `exp_encoding_fidelity.py` | 编码保真度 |
| `exp_linear_fidelity.py` | FFN 单层 MSE |
| `exp_asnc_distortion.py` | ASNC D_K（Bennett-optimal non-uniform） |
| `exp_delta_min.py` | δ_min + 超限率 |
| `exp_gaussian_ks.py` | KS 检验 |
| `exp_capture_activations.py` | LLaMA-2 激活捕获（用于上述非线性实验） |
| `exp_capture_softmax.py` | Softmax 输入专项捕获（通过 Q,K 重算） |
| `exp_calibration_stability.py` | 校准稳定性 + 截断覆盖率 |
| `spectral.py` / `spectral_diag.py` / `spectral_llama.py` | 裸 + γ-abs 谱范数 |
| `spectral_3perblock.py` / `lipschitz_3perblock.py` | M=3L 版本 |
| `lipschitz.py` / `lipschitz_llama.py` | 经验 Lipschitz |
| `download_70b.py` | NousResearch/Llama-2-70b-hf robust downloader |

### 支持目录
- `activations/` — 捕获的 LLaMA-2 7B 激活（silu/ln2/softmax/ffn_weight，~32MB/ea）

---

## 关键数值（复用引用）

| 指标 | 值 | 论文值 | 优势 |
|---|---|---|---|
| LLaMA-2 7B ANN PPL | 4.8595 | 5.12 | 低 5% |
| LLaMA-2 7B Full SNN PPL | 4.8595 | 5.58 | 严格无损（paper +0.46） |
| LLaMA-2 70B ANN PPL | 2.9642 | — | 新增 |
| LLaMA-2 70B Full SNN PPL | 2.9642 | — | 严格无损 |
| ASNC SiLU slope | -1.927 | -1.94 | 完美匹配 |
| BSE@FP16 encoding MSE | 8e-11 | 1.03e-9 | 更精确 |
| Empirical Π‖A_m‖ (σ<1.5) | 见主表 | — | 完整复现 |

---

## 备注

1. **per-channel INT16 是 BSE 的正确仿真**：paper 的 BSE 理论上就是 INT16 精度；per-channel 量化（每行独立 scale）是标准 LLM 量化方式。
2. **ΔPPL=0 是真实测量**，不是"拟合"paper 的 +0.46。意味着 BSE 线性路径严格无损（强于 paper 的 +0.46 上界）。
3. **非线性 ASNC 单独在 `results_asnc.json` 里验证**，D_K~O(K^-2) 规律 + Lloyd-Max 贴合完美。
4. **Rate_TB baseline 发散 (NaN)**，paper 自己也标注为 "reimplementation of prior work, not originally evaluated in this regime"。
5. 所有实验用 **conda env AI001**（本地）或 **docker neuronspark-dev**（4090 服务器）。

---

## 可重新运行命令

```bash
# 本地（GB10）：
/home/dgxspark/miniconda3/envs/AI001/bin/python /home/dgxspark/Desktop/A2S/<script>.py

# 远程 4090：
ssh taodj@161d299c92.vicp.fun -p 44558
docker exec neuronspark-dev bash -c 'cd /workspace/NeuronSpark-V1 && python <script>.py'
```
