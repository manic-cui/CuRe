<div align="center">

# 🌗 CuRe

### Learning Continuous Source Responses for Generalizable AI-Generated Image Detection

[Manni Cui](https://manic-cui.github.io/)<sup>1,2</sup> · Ruiqi Liu<sup>3</sup> · Zijian Yu<sup>1</sup> · Hao Tan<sup>3</sup> · Zibo Wei<sup>2</sup> · Zian Wang<sup>4</sup> · Ziheng Qin<sup>3</sup> · Huijia Zhu<sup>1</sup> · Weiqiang Wang<sup>1</sup> · Jun Lan<sup>1,†</sup> · Shu Wu<sup>3,†</sup>

<sup>1</sup>Ant Group &nbsp; <sup>2</sup>Huazhong University of Science and Technology &nbsp; <sup>3</sup>Institute of Automation, Chinese Academy of Sciences &nbsp; <sup>4</sup>Jilin University

<sub>† Corresponding authors</sub>

[![Project Page](https://img.shields.io/badge/Project-Page-2f4fd6.svg)](https://manic-cui.github.io/projects/cure/)
[![Paper](https://img.shields.io/badge/Paper-arXiv-b31b1b.svg?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2609.14316)
[![License](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-yellow.svg?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5%2B-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Stars](https://img.shields.io/github/stars/manic-cui/CuRe?style=social)](https://github.com/manic-cui/CuRe)

[English](README.md) · [简体中文](README_zh-CN.md)

[🌐 Project Page](https://manic-cui.github.io/projects/cure/) · [📄 Paper](https://arxiv.org/abs/2609.14316) · [🧭 Method](#-how-it-works) · [📊 Results](#-results) · [🚀 Quick Start](#-quick-start) · [📚 Citation](#-citation)

<img src="assets/teaser.jpg" width="95%" alt="Binary supervision versus continuous source change from real (alpha = 0) to generated (alpha = 1).">

</div>

---

> **TL;DR.** Instead of training a detector only on real/fake endpoints, CuRe teaches the backbone *how much* of an image is generated, by regressing the mixing ratio along a continuous real-to-generated path, and then classifies inside a compact source-response subspace. Trained only on GenImage SD v1.4, it reaches **89.7%** average balanced accuracy across ten public benchmarks, **+5.2** over the second-best method.

Existing AI-generated image detectors often perform well on in-domain data, but their robustness and cross-generator generalization remain limited, a weakness commonly attributed to overfitting to shortcut cues. Most methods that try to suppress shortcut learning still keep binary classification as the training task, without reconsidering how the task itself shapes the learned representation.

**CuRe** (learning **C**ontinuous So**u**rce **Re**sponses) revisits detection from the perspective of the training task. It reformulates backbone adaptation as regression of real–generated mixing ratios, which gives finer supervision than binary endpoint separation, and then selects a compact source-response subspace that suppresses nuisance variation and limits the final classifier's access to potential shortcut cues.

---

## 📰 News

- **2026.09** &nbsp;Paper released on [arXiv](https://arxiv.org/abs/2609.14316).
- **2026.09** &nbsp;Inference code and the trained CuRe adapter (25 MB) released.

---

## ✨ Highlights

| | |
|---|---|
| 🎯 **A different training task** | Regress the real–generated mixing ratio $\alpha$ instead of fitting binary labels at the two endpoints |
| 🧭 **Source-response subspace** | 128 directions selected by their response to real→generated change relative to content and same-source mixing, not by total variance |
| 🏆 **Strong generalization** | 89.7% average B.Acc. over 10 public benchmarks, ahead of DDA (84.5%) by 5.2 points |
| 🌍 **In the wild** | 93.8% on Chameleon and 79.7% on RealChain, +10.9 and +11.8 over the second-best method |
| 🧩 **Backbone-agnostic** | Ratio supervision beats binary supervision in all 12 settings across CLIP, DINOv2, DINOv3 and PE Core (LoRA r = 16 / 32 / 64), with AUC gains of 2.7–9.3 points |
| 🪶 **Light release** | All trained parameters ship as a 25 MB adapter on top of the public PE Core L/14 backbone |

---

## 🧭 How It Works

<div align="center">
<img src="assets/framework.jpg" width="95%" alt="Overview of the CuRe framework: Stage 1 continuous source learning and Stage 2 source-response subspace.">
</div>

**Stage 1 · Continuous source learning.** A real image $x_r$ and a generated image $x_g$, paired at random, are blended in pixel space with an independently sampled ratio:

$$x_\alpha = (1-\alpha)\,x_r + \alpha\,x_g, \qquad \alpha \sim \mathcal{U}(0,1)$$

The PE Core L/14 encoder stays frozen and is adapted with LoRA (rank 64 on the query and value projections of every attention layer). A sigmoid ratio head predicts $\alpha$ and is trained with binary cross-entropy against $\alpha$ as a soft target. The ratio head is then discarded.

**Stage 2 · Interference-suppressed source-response subspace.** With the adapted encoder frozen, CuRe traces real→generated (RF), real→real (RR) and generated→generated (FF) trajectories on the grid $\alpha \in \lbrace 0, 0.2, \dots, 1 \rbrace$ and summarizes each by its slope over $\alpha$. Three statistics follow: $C_F$ for RF responses, $C_M$ for same-source mixing responses, and $C_S$ for content variation across pairs. The subspace $W$ keeps the $k = 128$ directions with the strongest RF response relative to interference:

$$C_F\,w = \gamma\,(\lambda_s C_S + \lambda_m C_M + \epsilon I)\,w$$

**Stage 3 · Subspace-constrained classification.** A single linear layer is trained on $W^\top(z-\mu)$ with cross-entropy, while the encoder, $\mu$ and $W$ stay frozen. The classifier's decision direction is therefore restricted to $\mathrm{span}(W)$.

---

## 📊 Results

Balanced accuracy (%) on ten public benchmarks. CuRe is trained only on the Stable Diffusion v1.4 subset of GenImage. Each benchmark score is the unweighted mean over its evaluated subsets. Numbers are from Table 1 of the paper.

| Method | AIGCDetect | AIGIBench | UnivFD | EvalGEN | SynthWildX | WildRF | BFree-On. | RRDataset | RealChain | Chameleon | **Overall Avg.** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| NPR (CVPR'24) | 63.5 | 63.9 | 53.5 | 86.3 | 47.4 | 51.9 | 46.7 | 49.3 | 50.7 | 54.9 | 56.8 |
| UnivFD (CVPR'23) | 67.4 | 74.7 | 50.8 | 94.5 | 61.6 | 57.3 | 64.9 | 61.2 | 66.1 | 62.1 | 66.1 |
| FatFormer (CVPR'24) | 61.3 | 63.4 | 52.6 | 65.1 | 52.6 | 57.7 | 60.7 | 50.3 | 50.0 | 50.6 | 56.4 |
| C2P-CLIP (AAAI'25) | 78.4 | 69.6 | 91.6 | 66.4 | 55.3 | 59.5 | 50.0 | 52.7 | 50.2 | 50.7 | 62.4 |
| DRCT (ICML'24) | 68.3 | 73.4 | 56.7 | 75.4 | 74.6 | 73.8 | 73.8 | 60.5 | 64.9 | 68.5 | 69.0 |
| SAFE (KDD'25) | 73.0 | 60.2 | 53.0 | 92.8 | 56.0 | 58.5 | 60.8 | 59.3 | 63.4 | 46.7 | 62.4 |
| DDA (NeurIPS'25) | 86.1 | 85.0 | 77.4 | 97.9 | 90.7 | 90.5 | 94.5 | 71.6 | 67.9 | 82.9 | 84.5 |
| GAPL (CVPR'26) | 96.2 | 91.3 | 95.2 | 97.5 | 86.4 | 85.6 | 67.3 | 75.1 | 62.8 | 69.1 | 82.6 |
| DGS-Net (ICML'26) | 77.2 | 60.1 | 78.6 | 53.0 | 50.3 | 53.6 | 50.2 | 50.0 | 50.0 | 50.2 | 57.3 |
| **CuRe (Ours)** | 89.3 | 92.8 | 82.0 | 98.1 | 92.7 | 95.8 | 90.9 | 81.5 | 79.7 | 93.8 | **89.7** |

**Ablations on AIGIBench** (mean ± std over five seeds; Tables 4 and 5 of the paper):

| Stage 1 supervision | B.Acc. | AUC |
|---|---:|---:|
| Binary classification (baseline) | 80.2 ± 0.6 | 89.2 ± 0.9 |
| Stage-1 ratio prediction (direct) | 89.8 ± 0.3 | 97.2 ± 0.4 |
| **Full CuRe** | **92.6 ± 0.4** | **97.5 ± 0.7** |

| Stage 2 representation | B.Acc. | AUC |
|---|---:|---:|
| PCA-128 | 89.8 ± 0.9 | 91.1 ± 0.9 |
| Full feature | 89.7 ± 0.4 | 90.5 ± 0.2 |
| $C_M + C_S$ directions | 83.2 ± 1.1 | 85.8 ± 0.6 |
| **$C_F$ subspace (CuRe)** | **92.6 ± 0.4** | **97.5 ± 0.7** |

See the paper for subset-level results, cross-backbone comparisons and robustness to JPEG compression, resizing and blur.

---

## 🚀 Quick Start

### 1. Setup

Requires Python 3.10 or newer.

```bash
git clone https://github.com/manic-cui/CuRe.git
cd CuRe
pip install -r requirements.txt
shasum -a 256 -c weights/cure_adapter.pt.sha256
```

`weights/cure_adapter.pt` (about 25 MB) ships with this repository and holds
every trained parameter of CuRe: the Stage-1 LoRA updates, the Stage-2
source-response subspace (`mu`, `W`), and the linear classifier head.

The frozen backbone, Perception Encoder Core L/14 at 336 px, is not stored
here. On first use it is downloaded automatically from the public Hugging Face
repository `facebook/PE-Core-L14-336` (file `PE-Core-L14-336.pt`, 2.7 GB,
Apache-2.0) and cached by `huggingface_hub`. If the machine cannot reach
Hugging Face, either set `HF_ENDPOINT` to a mirror or download the file once
and pass `--base-checkpoint /path/to/PE-Core-L14-336.pt`.

### 2. Inference

```bash
python infer.py --input image.jpg
python infer.py --input images --recursive --batch-size 8
python infer.py --input image.jpg --base-checkpoint /path/to/PE-Core-L14-336.pt
```

Predictions are written to stdout as JSON lines. Use `--output output.jsonl`
to save them and `--checkpoint` to select another adapter. Run
`python infer.py --help` for all options. Each line has the form

```json
{"image": "<path relative to --input>", "probability_fake": <float in [0, 1]>, "prediction": "real" | "fake"}
```

where `prediction` is `fake` when `probability_fake` is at least 0.5.
Supported formats: `.jpg`, `.jpeg`, `.jfif`, `.png`, `.webp`, `.bmp`, `.tif`, `.tiff`.

Inference runs on the GPU by default (`--device cuda`, or `--device cuda:N`
to pick a card) with fp16 automatic mixed precision, which is the setting
used for all reported results. `--device cpu` is supported for machines
without a GPU; it computes in fp32, so scores can differ from the reported
ones in the third decimal place.

The backbone weights and the adapter are merged into one state dict and loaded
with `strict=True`, so any missing or unexpected tensor is an error.

---

## 🗂️ Repository Layout

```
.
├── infer.py                  inference entry point
├── cure/
│   ├── backbone.py           PE Core L/14 vision tower with Q/V LoRA
│   ├── model.py              CuRe model: backbone + subspace + linear head
│   └── preprocess.py         image preprocessing
├── weights/
│   ├── cure_adapter.pt       all trained CuRe parameters (~25 MB)
│   └── cure_adapter.pt.sha256
├── assets/                   figures used in this README
├── requirements.txt
├── LICENSE
└── NOTICE
```

---

## 📚 Citation

If you find CuRe useful, please cite:

```bibtex
@article{cui2026cure,
  title   = {Learning Continuous Source Responses for Generalizable {AI}-Generated Image Detection},
  author  = {Cui, Manni and Liu, Ruiqi and Yu, Zijian and Tan, Hao and Wei, Zibo and Wang, Zian and Qin, Ziheng and Zhu, Huijia and Wang, Weiqiang and Lan, Jun and Wu, Shu},
  journal = {arXiv preprint arXiv:2609.14316},
  year    = {2026}
}
```

---

## 📬 Contact

- **Issues** &nbsp;[github.com/manic-cui/CuRe/issues](https://github.com/manic-cui/CuRe/issues)
- **Email** &nbsp;`mannicui@berkeley.edu`
- **Homepage** &nbsp;[manic-cui.github.io](https://manic-cui.github.io/)

---

## ⚖️ License

Code is released under the [Apache License 2.0](LICENSE). The PE Core vision
implementation is adapted from Meta's
[Perception Encoder](https://github.com/facebookresearch/perception_models)
(Apache-2.0); see [NOTICE](NOTICE) for attribution.
