# The Cognitive Workspace Transformer (CWT v5.6)

**Structured State Management for Compute-Efficient Language Modeling**

*Author: [GitHub](https://github.com/Steel-skull) · [Discord](https://discord.com/users/steelcortex) · Independent Research, 2025*

---

## 📄 Read the Full Paper

** [Live Paper (GitHub Pages)](https://steel-skull.github.io/CWT-V5.6/)** — interactive Plotly visualizations included
** [Live Model (Huggingface)](https://huggingface.co/Steelskull/CWT-V5.6)**
---

## Overview

The Cognitive Workspace Transformer (CWT) replaces the standard transformer's undifferentiated residual stream with a **structured hub-and-spoke workspace** featuring:

- **Content-addressed decay gates** — selective forgetting based on who wrote what
- **Dual-system processing** (S1/S2) — inspired by dual-process theory
- **PonderNet-style adaptive compute** — variable depth per token at inference time
- **Hub-derived epistemic signals** — honest uncertainty without auxiliary classifiers

CWT is trained at **57.8M parameters** and evaluated against two controlled baselines on identical data (FineWeb-Edu, 5.2B tokens).

## Key Results

| Model | Total Params | Core Compute (Attn+FFN) | Layers | Val PPL |
|---|:---:|:---:|:---:|:---:|
| **CWT v5.6 (pondered)** | **57.8M** | **22.9M** | **8** | **29.54** |
| Parameter-matched baseline | 57.9M | ~32M | 8 | 30.67 |
| Compute-matched baseline | 67.5M | 41.7M | 13 | 29.04 |

- **Beats the parameter-matched baseline by 3.7%** despite fewer attention+FFN parameters
- **Comes within 1.7% of a 13-layer baseline** that has **51% more core compute capacity** (41.7M vs 22.9M)
- Provides a **smooth inference-time compute/quality tradeoff** (PPL 34.82 at 1.0× to PPL 28.49 at 2.25× compute)
- **18.7% pondering benefit** at convergence — stable from step 4,500 onward

## Architecture

```
┌─────────────────────── Workspace State (d_s = 896) ──────────────────────┐
│                                                                           │
│  ┌──────────┐  ┌──────────────┐  ┌──────────────────┐  ┌──────────────┐  │
│  │  Spokes   │  │  Billboards  │  │   Hub Shared     │  │    Tags      │  │
│  │  48d × 8  │  │  16d × 8     │  │   256d           │  │  16d × 8     │  │
│  │  Private   │  │  Permanent   │  │   Decay-gated    │  │  Identity    │  │
│  │  per-layer │  │  broadcast   │  │   shared memory  │  │  markers     │  │
│  └──────────┘  └──────────────┘  └──────────────────┘  └──────────────┘  │
│                                                                           │
└───────────────────────────────────────────────────────────────────────────┘
        │                                      │
  ┌─────┴─────┐                         ┌──────┴──────┐
  │  System 1  │ ──── 6 layers ────→    │  System 2   │ ──── 2 layers × N steps
  │  (S1)      │  Fast processing       │  (S2)       │  Adaptive pondering
  │  bias: 3.0 │                        │  bias: 5.0  │  PonderNet halting
  └───────────┘                         └─────────────┘
```

### Workspace Regions
- **Spokes** (48d × 8 layers) — private scratch memory per layer
- **Billboards** (16d × 8 layers) — permanent broadcast channels, never decayed
- **Hub Shared** (256d) — central communication channel with decay gates (+8,114% degradation when zeroed)
- **Tags** (16d × 8 layers) — identity markers for content-addressed forgetting

### Key Innovations
- **CWT-MLA**: Hub-derived Multi-head Latent Attention with 8× KV cache compression
- **Hub Self-Distillation**: 250× cheaper deep supervision versus vocabulary probes (v5.4 → v5.6)
- **FlooredMultiply**: Straight-through gradient estimation with configurable gradient floors
- **Epistemic Self-Monitoring**: Calibrated uncertainty from hub delta dynamics — no auxiliary classifiers

## Ablation Highlights (30+ interventions)

| Tier | Ablation | Degradation |
|---|---|---|
| 🔴 Existential | Zero Hub Shared | +8,114% |
| 🔴 Existential | Zero Tags | +4,392% |
| 🟠 Critical | Hub Writes ×2 | +547% |
| 🟡 Important | Dead Gates (all → 1.0) | +54% |
| 🟡 Important | Kill Pondering | +22% |
| 🟢 Minimal | Epistemic Gate | +0.04% |

Hub write sensitivity increased from +76% (step 8K) to +547% (step 20K), revealing precise calibration that develops over training.

## Visualizations

The paper includes **18 interactive Plotly visualizations** embedded as iframes:

**In-Distribution** ("The process of photosynthesis involves…"):
- 3D UMAP hub trajectory · Topology animation · Workspace regions · Hub deltas
- Hub similarity · Decay gates · Gate selectivity · Ponder oscillation
- Write magnitudes · Layer ranking

**Out-of-Distribution** ("hey bud no cap fo real fo real"):
- 3D UMAP hub trajectory · Workspace regions · Hub deltas
- Decay gates · Ponder oscillation · Write magnitudes

**Comparative**: Overlaid in-distribution vs OOD hub trajectories

## Training Details

| Parameter | Value |
|---|---|
| Dataset | FineWeb-Edu (sample-10BT), ~5.2B tokens |
| Hardware | 4× NVIDIA RTX 3090, DDP |
| Optimizer | AdamW (β₁=0.9, β₂=0.95) |
| LR Schedule | Cosine decay, 3×10⁻⁴ → 10⁻⁵ |
| Batch Size | 96 sequences (3 × 8 accum × 4 GPUs) |
| Phase 1 | Steps 0–2K, S1 only, ~29K tok/s |
| Phase 2 | Steps 2K–20K, full pondering, ~16K tok/s |
| Tokenizer | GPT-2 (50,257 vocab) |
| Sequence Length | 4,096 |

## Scaling Outlook

- At 130M params → overhead drops to ~5–7%
- At 300M+ → overhead becomes noise
- CWT's efficiency advantage is predicted to increase with scale

## Citation

If you reference this work:

```
GVP (2025). The Cognitive Workspace Transformer: Structured State Management
for Compute-Efficient Language Modeling. Independent Research.
https://steel-skull.github.io/CWT-V5.6/
```

## License

MIT - This paper and its visualizations are published for research purposes. Please cite appropriately if referencing this work.
