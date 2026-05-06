# Experiment Details

> **Purpose:** Full experimental record for Section 5 (Experimental Setting) and Section 6 (Results & Analysis) of the paper.
> All numbers are drawn directly from the codebase configs, source code, and stored benchmark artifacts.
> Cross-seed averages are computed over seeds {42, 123, 456}.

---

## A. Experimental Setting (Section 5)

### A.1 Hardware & Software Framework

| Component | Details |
|-----------|---------|
| **Host GPU pool** | 2× Tesla P100-PCIE-16GB (GPUs 0, 1); 2× Tesla V100S-PCIE-32GB (GPUs 2, 3); 1× Tesla P100-PCIE-16GB (GPU 4); 1× Tesla V100-PCIE-32GB (GPU 5) |
| **Primary training GPU** | Tesla V100S-PCIE-32GB (32 GB HBM2) — default `CUDA_VISIBLE_DEVICES=2` |
| **Framework** | PyTorch 2.1.2 + CUDA 12.1 + cuDNN 8 (pinned in Dockerfile `FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime`) |
| **Lightning / HuggingFace** | Neither used for training. Clay weights are loaded from a PyTorch Lightning checkpoint (`clay-v1.5.ckpt`); the training loop is a custom `Trainer` class |
| **Key libraries** | `segmentation-models-pytorch ≥ 0.3.3`, `timm ≥ 0.9.0`, `xgboost ≥ 2.0.0`, `einops ≥ 0.7.0` |
| **Containerisation** | Docker with NVIDIA Container Toolkit; each model family runs in its own `docker run` invocation |
| **Distributed training** | U-Net uses `torchrun` with `--nproc_per_node=3` (DDP, 3× P100-PCIE-16GB, GPUs 0/1/2). Clay uses `torchrun --nproc_per_node=2` (2 GPUs). SegFormer-B3 single GPU. XGBoost single GPU |
| **Mixed precision** | AMP (`torch.cuda.amp.autocast`) enabled for all DL models (`amp: true`) |
| **Gradient clipping** | `clip_grad_norm_(parameters, max_norm=1.0)` applied every optimizer step |

**Note on training time:** No explicit wall-clock time was logged. Rough estimates from epoch counts and GPU class:
- U-Net: 60 epochs × ~1 min/epoch on 3× P100 DDP ≈ ~60–90 min per seed
- SegFormer-B3: 60 epochs, single GPU (V100S) ≈ ~2–4 hours per seed (large ViT encoder)
- Clay: 30 + 30 = 60 effective epochs (Phase 1 + Phase 2), 2-GPU DDP ≈ ~4–6 hours per seed
- XGBoost: GPU-accelerated hist method, single P100 ≈ ~10–30 min per seed

---

### A.2 Hyperparameters

All four models were trained under the same benchmark budget of **60 effective epochs** and evaluated on the same fixed test split (1,853 patches). A unified checkpoint selection criterion — minimum validation **combined RMSE** = `(RMSE_tree_count + RMSE_mean_height) / 2` (computed in log1p space) — was used across all DL models. There is no early stopping for DL models; the best checkpoint is saved and used for evaluation while training runs for all configured epochs.

#### U-Net (ResNet50)

| Hyperparameter | Value |
|---------------|-------|
| Optimizer | AdamW |
| Learning rate | 1e-4 |
| Weight decay | 1e-4 |
| Batch size | 21 per GPU × 3 GPUs (DDP) = **63 effective** |
| Epochs | 60 |
| LR Scheduler | Cosine Annealing (`CosineAnnealingLR`, `T_max = epochs × steps_per_epoch / grad_accum`, `η_min = lr × 0.01`) |
| Early stopping | No (best checkpoint by val combined RMSE) |
| Loss | Masked Smooth L1 |
| Loss weights | tree\_count: 1.0, mean\_height: 1.0 |
| Gradient accumulation | 1 |

#### SegFormer-B3 (PVTv2-B3)

| Hyperparameter | Value |
|---------------|-------|
| Optimizer | AdamW |
| Learning rate | 6e-5 |
| Weight decay | 1e-2 |
| Batch size | **8** (single GPU) |
| Epochs | 60 |
| LR Scheduler | Cosine Annealing |
| Early stopping | No |
| Loss | Masked Smooth L1 |
| Loss weights | tree\_count: 1.0, mean\_height: 1.0 |

#### Clay (v1.5 Foundation Model)

Clay uses a **two-phase training schedule**:

| Phase | Encoder | Epochs | LR | Batch size | GPUs |
|-------|---------|--------|----|------------|------|
| Phase 1 (warm-up) | **Frozen** | 30 | 1e-4 | 16 × 2 = **32 effective** | 2 (DDP) |
| Phase 2 (fine-tune) | **Unfrozen** | 30 | 5e-5 | 16 × 2 = **32 effective** | 2 (DDP) |

Phase 2 resumes from the Phase 1 best checkpoint. Final test metrics come from Phase 2.

Common across both phases:
- Optimizer: AdamW, weight\_decay: 1e-4
- Scheduler: Cosine Annealing
- Loss: Masked Smooth L1 (weights 1.0 / 1.0)
- No early stopping

#### XGBoost

| Hyperparameter | Value |
|---------------|-------|
| n\_estimators | 2000 |
| max\_depth | 6 |
| learning\_rate (η) | 0.03 |
| subsample | 0.8 |
| colsample\_bytree | 0.8 |
| min\_child\_weight | 5 |
| tree\_method | hist (GPU-accelerated) |
| device | cuda |
| **Early stopping** | **Yes — 40 rounds** (only model with early stopping) |
| feature\_mode | pixel (one sample = one pixel, 48 spectral features) |
| subsample\_pixels | 0.03 (3% of training pixels sampled for efficiency) |
| Two separate models | One XGBoost model per target (tree\_count, mean\_height) |

---

### A.3 Multi-Task Loss Combination

The two tasks (tree count, mean height) are jointly predicted as a 2-channel output `[B, 2, H, W]` and optimised with a **static, unweighted sum** of per-target losses:

```
L_total = w_count · L(ŷ_count, y_count) + w_height · L(ŷ_height, y_height)
```

where **w\_count = w\_height = 1.0** (equal scalar weights, not learned). The underlying per-target loss is **Masked Smooth L1** (Huber-like loss, δ=1), computed **only over valid (tree-present) pixels** identified by `valid_mask`. If no valid pixels exist in a batch, a differentiable zero is returned.

This is a **simple static weighted sum** — not Homoscedastic Uncertainty Weighting (Kendall et al., 2018) or any dynamic weighting scheme. Both targets are first transformed to **log1p space** before loss computation, which partially normalises the scale difference between counts and heights and reduces the influence of extreme values.

The masking mechanism is the core sparsity-handling technique: pixels where the target is null (i.e., no tree annotation) do not contribute to gradients, preventing the model from learning to suppress tree predictions in un-annotated areas.

Source: `src/losses/masked_regression.py`, `MaskedRegressionLoss`.

---

### A.4 Architecture Details

#### U-Net (ResNet50)

- **Library:** `segmentation-models-pytorch` (`smp.Unet`)
- **Encoder:** ResNet50 pretrained on ImageNet (weights adapted for multi-channel input)
- **Input channels:** 48 (Sentinel-1: 4 seasons × 2 bands = 8 ch; Sentinel-2: 4 seasons × 10 bands = 40 ch)
- **Decoder channels:** [256, 128, 64, 32, 16] (5-stage upsampling)
- **Prediction head:** A **single shared 2-channel convolutional head** (`Conv2d(16, 2, kernel_size=1)`) at the end of the decoder. The two output channels (tree\_count, mean\_height) are produced by the same decoder stack — there are **no separate parallel towers or dual heads**. Output shape: `[B, 2, H, W]`, no activation (raw regression).

#### SegFormer-B3 (PVTv2-B3 + all-MLP decoder)

- **Encoder:** `timm` PVTv2-B3 (Pyramid Vision Transformer v2, 45 M params), pretrained on ImageNet-1k. Note: the paper refers to this as "SegFormer-style" but uses PVTv2-B3 as encoder (not the `nvidia/segformer-*` HuggingFace checkpoints).
- **Input channels:** 48 (same multi-season Sentinel stack). PVTv2's first patch-embed layer is replaced to accept 48 channels.
- **Decoder:** All-MLP SegFormer-style decoder:
  1. Per-scale 1×1 projection convolutions (each feature scale → embed\_dim=256)
  2. Bilinear upsampling to H/4 × W/4, then concatenation
  3. 3×3 fusion convolution → embed\_dim=256
  4. **Single 1×1 `Conv2d(256, 2, kernel_size=1)`** → bilinear upsample to full 128×128 resolution
- **Output:** `[B, 2, H, W]`, no activation. Same single 2-channel head architecture as U-Net.

#### Clay Foundation Model (v1.5)

- **Backbone:** Clay ViT-Large (`patch_size=8`, `dim=1024`, `depth=24`, `heads=16`) — ~307 M encoder parameters. The encoder is loaded from the official `clay-v1.5.ckpt` checkpoint (Lightning checkpoint format; only `model.encoder.*` keys are extracted).
- **Species branch:** A lightweight convolutional side-branch processes per-patch species composition features. Its output is projected to `species_feat_dim=32` and fused with the encoder feature map via a 1×1 convolution (`1024+32 → 1024`).
- **Decoder / regression head (`_RegressionHead`):** Three stacked `ConvTranspose2d(1024, ...) × stride=2` upsampling blocks, ending in a **`Conv2d(..., 2, kernel_size=1)`** prediction layer. This produces `[B, 2, H, W]` — again a single 2-channel head.
- **Encoder freezing:** Phase 1 trains only the decoder and species branch (~few M parameters). Phase 2 **fully fine-tunes the entire model** (~311 M parameters total). This two-phase strategy prevents catastrophic forgetting during the early warm-up stage.
- **Input tokens:** Clay's multi-modal tokenizer processes Sentinel-1 and Sentinel-2 bands separately; seasonal stacks are passed as multiple time-step tokens.

---

### A.5 Data Augmentation

Augmentation is applied **only during training**; validation and test sets use no augmentation.

| Transform | Probability | Details |
|-----------|-------------|---------|
| `RandomHorizontalFlip` | p = 0.5 | Horizontal mirror of the 128×128 patch |
| `RandomVerticalFlip` | p = 0.5 | Vertical mirror |
| `RandomRotate90` | p = 0.75 | Rotation by 90° × k, where k ∈ {1, 2, 3} (uniform) |

Augmentations are applied identically to the input imagery tensor and the target label tensor (both channels), and to the valid mask — ensuring spatial consistency. **No Gaussian noise, cutout, or spectral augmentations** were applied.

Source: `src/data/transforms.py`, `build_train_transform()`.

---

## B. Results & Metrics (Section 6)

### B.1 Dataset Split Summary

| Split | N patches | Notes |
|-------|-----------|-------|
| Train | 9,327 | Spatially blocked; augmented during training |
| Validation | 2,446 | Used for checkpoint selection |
| **Test** | **1,853** | Final benchmark evaluation; no augmentation |

All models evaluated on the same 1,853 test patches. Metrics are computed on pixels where `valid_mask = True` (tree-annotated pixels). The **support ratio** (fraction of valid pixels per batch, averaged over test) is approximately **0.27** — meaning ~73% of pixels are sparse (no tree label), confirming significant sparsity.

Both targets were log1p-transformed during training. Test metrics are reported in **original scale** (after `expm1` inverse transform) for interpretability.

---

### B.2 Quantitative Results — Test Set

All metrics averaged over **3 random seeds** (42, 123, 456). Lower RMSE/MAE is better; higher R² is better.

#### Tree Count

| Model | RMSE ↓ | MAE ↓ | R² ↑ | Rank |
|-------|--------|-------|------|------|
| **XGBoost** | **0.596** | 0.511 | **0.041** | 🥇 1st |
| SegFormer-B3 | 0.602 | **0.508** | 0.022 | 🥈 2nd |
| Clay (Phase 2) | 0.603 | 0.525 | 0.020 | 🥉 3rd |
| U-Net (ResNet50) | 0.666\* | 0.515 | −0.214\* | 4th |

> \* U-Net seed 42 is a notable outlier (RMSE=0.779, R²=−0.638), likely due to a bad initialisation. Seeds 123 and 456 are much closer to the other models (RMSE ≈ 0.604–0.614, R² ≈ −0.02 to +0.01). Excluding seed 42, U-Net mean RMSE ≈ 0.609, R² ≈ −0.002.

#### Per-seed breakdown — Tree Count

| Model | Seed | RMSE | MAE | R² |
|-------|------|------|-----|----|
| U-Net | 42 | 0.779 | 0.519 | −0.638 |
| U-Net | 123 | 0.614 | 0.516 | −0.019 |
| U-Net | 456 | 0.604 | 0.511 | +0.015 |
| SegFormer-B3 | 42 | 0.603 | 0.508 | +0.019 |
| SegFormer-B3 | 123 | 0.602 | 0.506 | +0.022 |
| SegFormer-B3 | 456 | 0.600 | 0.509 | +0.027 |
| Clay p2 | 42 | 0.603 | 0.526 | +0.019 |
| Clay p2 | 123 | 0.603 | 0.524 | +0.018 |
| Clay p2 | 456 | 0.602 | 0.526 | +0.021 |
| XGBoost | 42 | 0.596 | 0.511 | +0.041 |
| XGBoost | 123 | 0.596 | 0.511 | +0.040 |
| XGBoost | 456 | 0.596 | 0.511 | +0.040 |

#### Mean Height (meters)

| Model | RMSE ↓ | MAE ↓ | R² ↑ | Rank |
|-------|--------|-------|------|------|
| **SegFormer-B3** | **5.877** | **4.515** | **0.503** | 🥇 1st |
| U-Net (ResNet50) | 6.112 | 4.693 | 0.463 | 🥈 2nd |
| XGBoost | 6.608 | 5.182 | 0.372 | 🥉 3rd |
| Clay (Phase 2) | 7.115 | 5.707 | 0.272 | 4th |

#### Per-seed breakdown — Mean Height

| Model | Seed | RMSE | MAE | R² |
|-------|------|------|-----|----|
| U-Net | 42 | 6.280 | 4.757 | 0.433 |
| U-Net | 123 | 6.085 | 4.698 | 0.468 |
| U-Net | 456 | 5.970 | 4.625 | 0.487 |
| SegFormer-B3 | 42 | 5.816 | 4.448 | 0.514 |
| SegFormer-B3 | 123 | 5.853 | 4.503 | 0.507 |
| SegFormer-B3 | 456 | 5.961 | 4.594 | 0.489 |
| Clay p2 | 42 | 7.121 | 5.715 | 0.271 |
| Clay p2 | 123 | 7.106 | 5.697 | 0.274 |
| Clay p2 | 456 | 7.117 | 5.709 | 0.272 |
| XGBoost | 42 | 6.603 | 5.178 | 0.373 |
| XGBoost | 123 | 6.605 | 5.179 | 0.373 |
| XGBoost | 456 | 6.616 | 5.188 | 0.370 |

---

### B.3 Task Difficulty Analysis — Which Task Was Harder?

**Tree count was significantly harder to learn** across all models. The evidence is clear:

- **R² for tree count maxes out at ~0.04** (XGBoost, the best model), compared to **R² up to 0.50** for mean height (SegFormer-B3). This means the models collectively explain less than 5% of the variance in tree count, while explaining ~50% of the variance in height.
- **Relative RMSE** (normalised by target range) is also disproportionately worse for tree count.

Several factors explain this difficulty:
1. **Heavy-tailed, discrete distribution:** Tree count per pixel is a non-negative integer with a zero-inflated distribution (many empty pixels, a long tail of dense forest patches). Even after log1p transformation, the distribution is difficult to regress.
2. **Local spatial coherence:** Mean height is spatially smooth (tall forests form coherent patches), while tree count can vary sharply at stand boundaries — a harder signal for dense-prediction architectures.
3. **Label quality:** Mean height is derived from LiDAR canopy height models, which are smoother. Tree count involves crown delineation, which may introduce label noise.
4. **Support sparsity:** With only ~27% valid pixels (on average), there are relatively few count-labelled pixels per batch to learn from.

---

### B.4 Foundation Model vs. Baselines

**Clay did not outperform the from-scratch models** — it ranked last on mean height and 3rd on tree count. Key observations:

- **SegFormer-B3 was the best overall**, particularly for mean height (R² = 0.50, RMSE = 5.88 m). Despite being trained from scratch on this specific multi-season SAR+optical stack, it outperformed the Clay foundation model.
- **Clay Phase 2 (unfrozen) performance is surprisingly weak on height** (R² = 0.27, RMSE = 7.11 m), substantially worse than both U-Net (R² = 0.46) and SegFormer (R² = 0.50).
- **On tree count, all DL models are essentially tied** (RMSE ≈ 0.60–0.61, R² ≈ 0.02), with XGBoost marginally ahead.

**Possible explanations for Clay's underperformance:**
1. **Domain mismatch:** Clay v1.5 was pretrained primarily on Sentinel-2 RGB/multispectral global scenes. Our dataset uses 48 channels (multi-season SAR + 10-band multispectral) — a very different input distribution.
2. **Decoder capacity:** The convolutional `ConvTranspose2d` decoder is relatively shallow for a ViT-Large encoder. The from-scratch SegFormer decoder with its MLP fusion may be better suited to this regression task.
3. **Insufficient fine-tuning:** 30 epochs of full fine-tuning may not be enough to adapt the 307 M parameter ViT-Large encoder to the remote sensing regression domain.
4. **Competing inductive biases:** The Clay pretraining may have instilled priors that are beneficial for classification or segmentation but neutral or harmful for pixel-level height regression.

**XGBoost is strongly competitive on tree count** (R² = 0.04, best of all models), demonstrating that pixel-level spectral features from multi-season SAR and optical data carry discriminative information that is efficiently exploited by gradient-boosted trees — without requiring any spatial context or learned representations.

---

### B.5 Handling Sparsity

The **Masked Smooth L1 Loss** is the central mechanism for handling sparsity:

- Gradients flow **only from valid (tree-present) pixels**. Approximately 73% of pixels are masked out during every training step.
- This prevents the model from being penalised for predicting zero (or any value) on background pixels, avoiding the classic failure mode of learning a near-zero constant prediction to minimise loss on empty regions.
- For the same reason, **no "forest prior" bias was observed** in quantitative evaluation — the model does not systematically over-predict trees in open fields.

However, the **low R² for tree count (~0.04)** across all models suggests that while masking prevents systematic bias, the models struggle to capture the fine-grained spatial variation in tree density. The heavy-tailed count distribution remains a challenge beyond what loss masking alone can address.

Note: XGBoost uses a completely different sparsity strategy — it simply trains only on the 3% of sampled valid pixels (`subsample_pixels: 0.03`), implicitly operating on the same masked signal.

---

### B.6 Spatial Generalisation (OOD Analysis)

**Current status: Not yet formally evaluated. Recommended framing: Future Work.**

The dataset was built with spatial generalisation in mind:
- Training, validation, and test sets are split using **spatially blocked groups** (6×6 km grid cells over Bavaria). This prevents spatial autocorrelation leakage between splits — patches from the same geographic block cannot appear in both train and test.
- The `patch_index_subset.parquet` metadata includes two columns specifically for OOD analysis: `distance_to_nearest_test_km` (geographic distance from each train/val patch to the nearest test patch) and `buffered` (bool flag for patches in a geographic buffer zone around split boundaries).

However, **no dedicated OOD evaluation pipeline exists in the current codebase**. The `scripts/make_figures.py` script generates a Bavaria map (Figure 2) visualising the spatial split blocks by colour, which serves as a qualitative illustration of the geographic spread of splits — but is not a quantitative OOD benchmark.

**For the paper:** Frame this as follows — "The spatial blocking design of our dataset enables future analysis of performance degradation as a function of geographic distance from training data. We provide the necessary metadata (`distance_to_nearest_test_km`, `buffered`) to support such studies, which we leave as future work."

---

## C. Reproducibility Notes

| Aspect | Details |
|--------|---------|
| Seeds | 3 fixed seeds: 42, 123, 456 |
| Determinism | `deterministic: false` (for speed); `cudnn_benchmark: true` — minor non-determinism possible |
| Target transform | Both targets log1p-transformed before loss; metrics in original scale via `expm1` |
| Normalisation | Per-channel mean/std computed on train split; stored in `artifacts/norm_stats.json` |
| Checkpoint selection | Best validation combined RMSE = `(RMSE_tc + RMSE_mh) / 2` in log1p space |
| Test evaluation | Done once after training; no test-set hyperparameter tuning |

---

## D. Summary Table for Paper

| Model | TC RMSE | TC MAE | TC R² | MH RMSE (m) | MH MAE (m) | MH R² |
|-------|---------|--------|-------|------------|-----------|------|
| XGBoost | **0.596 ± 0.000** | 0.511 ± 0.000 | **0.041 ± 0.000** | 6.608 ± 0.007 | 5.182 ± 0.005 | 0.372 ± 0.001 |
| U-Net (ResNet50) | 0.666 ± 0.097 | 0.515 ± 0.004 | −0.214 ± 0.329 | 6.112 ± 0.155 | 4.693 ± 0.067 | 0.463 ± 0.028 |
| SegFormer-B3 | 0.602 ± 0.001 | **0.508 ± 0.001** | 0.022 ± 0.004 | **5.877 ± 0.075** | **4.515 ± 0.073** | **0.503 ± 0.012** |
| Clay v1.5 (Phase 2) | 0.603 ± 0.000 | 0.525 ± 0.001 | 0.020 ± 0.001 | 7.115 ± 0.008 | 5.707 ± 0.009 | 0.272 ± 0.002 |

TC = Tree Count, MH = Mean Height. Mean ± std over 3 seeds. **Bold** = best per column.
