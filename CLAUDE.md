# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Senior AI Researcher home assignment for ELTA. Build a multimodal Human Activity Classification system using the **UTD-MHAD** dataset (RGB video + inertial sensors). The assignment has 9 parts covering EDA, unimodal models, fusion, missing-modality robustness, class imbalance, confidence estimation, OOD detection, retrieval vs fine-tuning (written), and production monitoring (written + diagram).

## Running Scripts

```bash
# Train models with variants (LOSO cross-validation)
python3 train.py --modality sensor
python3 train.py --modality video
python3 train.py --modality fusion
python3 train.py --modality fusion --imbalance                    # Undersample majority
python3 train.py --modality fusion --imbalance --augment_minority # + warp augmentation
python3 train.py --modality fusion --imbalance --weighted_loss    # + weighted CE

# Evaluate checkpoints (all load from checkpoints/final_eval/)
python3 test.py --modality sensor
python3 test.py --modality video
python3 test.py --modality fusion
python3 test.py --modality fusion --imbalance
python3 test.py --modality fusion --imbalance --augment_minority
python3 test.py --modality fusion --imbalance --weighted_loss

# Part 7: OOD Detection (Mahalanobis distance with two-tier alarm)
python3 test_ood.py --modality sensor --ood-actions 24 9 12
python3 test_ood.py --modality video --ood-actions 24 9 12
python3 test_ood.py --modality fusion --ood-actions 24 9 12

# Legacy part scripts (standalone, no wandb)
python3 part1_eda.py
python3 part2_video.py          # MediaPipe → Kalman → Transformer
python3 part2_classical.py
python3 part3_fusion.py         # Bayesian fusion
python3 part3_fusion_feature.py # Feature-MLP fusion
python3 part5_imbalance.py
python3 part5_fusion.py
python3 part6_confidence.py     # Temperature scaling + conformal prediction
```

**Device:** All scripts use MPS (Apple Silicon) > CUDA > CPU automatically.

**Notebooks:** All test*.py scripts work in Jupyter with `subprocess.run()` — confusion matrices, ROC curves, and summary tables display inline automatically.

## Dataset Structure

Files live directly in the project root:
- `Inertial/` — 861 `.mat` files: `a{1-27}_s{1-8}_t{1-4}_inertial.mat` (6-axis IMU: 3-axis accel + 3-axis gyro, ~25 Hz)
- `RGB-part1/` through `RGB-part4/` — `.avi` color videos: `a{1-27}_s{1-8}_t{1-4}_color.avi`
- `Depth/`, `Skeleton/` — not used
- `Sample_Code/Action_List.txt` — authoritative class names (parsed by `utils.load_action_names`)

**Naming convention**: `a` = action (1–27), `s` = subject (1–8), `t` = trial (1–4). A paired sample is one `(a_s_t_inertial.mat, a_s_t_color.avi)` tuple.

**27 action classes** — Actions 1–21: wrist-worn sensor; Actions 22–27: thigh-worn sensor. Only a subset of 8 actions is used: `[1, 2, 4, 13, 19, 22, 23, 27]` (defined in `configs/sensor.yaml` and `configs/video.yaml`).

## Code Architecture

### Key Constraints
- Use only RGB video + inertial (accel + gyro) modalities.
- All experiments use Leave-One-Subject-Out (LOSO) cross-validation over 8 subjects.
- Seed is fixed everywhere (`seed: 42` in configs).
- Outputs (plots, confusion matrices) go to `outputs/`. Checkpoints go to `checkpoints/`.

### Shared Modules

**`utils.py`** — Three shared utilities used by all train/test/part scripts:
- `load_config(path)` → `SimpleNamespace` (YAML → nested namespace)
- `load_action_names(sample_dir)` → `dict[int, str]`
- `save_confusion_matrix(y_true, y_pred, class_names, title, path, cmap="Blues")`

**`models/model.py`** — Single `TransformerClassifier` used for both modalities:
- Input `(B, T, in_dim)` → linear projection → positional encoding → Transformer encoder → mean+max pooling → MLP head → `(B, n_classes)`
- `in_dim=12` for sensor (6 IMU + 6 velocity), `in_dim=len(keep_feats)*2` for video (pose + velocity)
- **OOD Detection:** `forward(..., return_embedding=True)` returns `(logits, embedding)` where embedding is `(B, d_model)` after pooling, before classification head
- Embeddings used for Mahalanobis distance computation in `test_ood.py`

**`models/fusion.py`** — `FusionTransformerClassifier` for late fusion:
- Concatenates sensor + video embeddings: `[e_s_mean, e_s_max, e_v_mean, e_v_max]` → `(B, 4*d_model)`
- Passes fusion embedding through MLP head → `(B, n_classes)`
- **OOD Detection:** `forward(..., return_embedding=True)` returns `(logits, fusion_embedding)` where embedding is `(B, 4*d_model)` (4× larger than unimodal)
- Handles null embeddings for missing modalities during training (`drop_v`, `drop_s` masks)

**`data/sensor_dataset.py`**:
- `load_sensor_samples(inertial_dir, subset, label_map)` → `list[(subject, label, raw_iner)]` where `raw_iner` is `(T, 6)`
- `SensorDataset`: pads/truncates to `max_len=256`, adds velocity → `(max_len, 12)`, z-score normalizes
- `extract_imu_features(iner)` → 36-dim vector: per-channel × [mean, std, RMS, energy, dom_freq, spectral_entropy]

**`data/video_dataset.py`**:
- `load_video_samples(kalman_cache, subset, label_map, keep_feats)` → reads `.npy` files from `outputs/pose_cache_kalman22/`
- `PoseDataset`: adds velocity → `(T, pos_dim*2)`, z-score normalizes. No fixed-length padding (sequences have variable length)

**`data/augmentation.py`** — `time_warp` and `amplitude_warp` for IMU sequences (nonlinear time stretch and smooth amplitude scaling).

### Unified Train/Test Scripts

**`train.py`** — Unified training for all modality + variant combinations:
- `--modality {sensor, video, fusion}` selects backbone
- `--imbalance` undersample majority class (8:1 → balanced ratio)
- `--augment_minority` add time/amplitude warp to minority class samples
- `--weighted_loss` use weighted cross-entropy (inverse frequency weights)
- All variants save to `checkpoints/final_eval/{modality}{suffix}`
- Uses `SensorLightningModule` or `VideoLightningModule` (identical structure): Adam + CosineAnnealingLR + cross-entropy loss

**`test.py`** — Unified evaluation for all modality + variant combinations:
- Same `--modality`, `--imbalance`, `--augment_minority`, `--weighted_loss` flags
- Loads from same `checkpoints/final_eval/{modality}{suffix}` location
- Outputs:
  - Confusion matrix (displayed inline in Jupyter)
  - Per-class accuracy + F1 scores
  - Aggregate metrics (mean ± std across 8 folds)
  - Optional conformal prediction sets (`--conformal` flag)
- **Jupyter Compatible:** All plots display inline when run via `subprocess.run(['python3', 'test.py', ...])`

### Video Pipeline (part2_video.py)
MediaPipe Pose (33 landmarks × 2D) → Kalman filter smoothing → normalize to hip center + shoulder width → cache as `.npy` in `outputs/pose_cache_kalman22/`. The cache is the input to all subsequent video model training. `drop_feats` in `configs/video.yaml` lists 12 low-importance features identified by permutation importance.

### Fusion Strategies (part3)
- **Bayesian fusion** (`part3_fusion.py`): sensor determines wrist/thigh placement prior → gates video softmax probabilities
- **Feature-MLP fusion** (`part3_fusion_feature.py`): frozen encoders from Part 2 checkpoints feed into `FusionMLP`

### Part 6 — Confidence Estimation
`part6_confidence.py` uses a **3-way cyclic LOSO split**: fold `i` has test=s_i, cal=s_{(i+1) mod 8}, train=remaining 6. Retrains both models per fold, then applies:
1. **Temperature scaling**: `T* = argmin NLL(logits/T, y_cal)` via `scipy.optimize.minimize_scalar`
2. **Split conformal prediction**: non-conformity score `s = 1 − softmax(z/T*)[y_true]`, quantile gives coverage guarantee `P(y ∈ C(x)) ≥ 1−α`

### Part 7 — OOD Detection
`test_ood.py` detects out-of-distribution samples using **Mahalanobis distance** on learned embeddings:

**Setup (3-fold split per fold):**
- Train subjects: 6 subjects (learn class distribution centroids + shrunk covariances)
- Cal subject: 1 subject (calibrate Mahalanobis threshold to 95th percentile = 5% FAR)
- Test subject: 1 subject (evaluate ID accuracy)
- OOD samples: actions 24, 9, 12 (truly unseen during training)

**Algorithm:**
1. Extract embeddings using `model.forward(..., return_embedding=True)` (mean+max pooling before classification head)
2. Compute Ledoit-Wolf shrunk covariance per class on training data
3. Threshold: 95th percentile of Mahalanobis distances on calibration set
4. Alarm: sample is OOD if `min_k mahal_distance(x, class_k) > threshold`

**Outputs:**
- **Metrics per fold:** ID accuracy, FAR (false alarm rate), OOD TPR (true positive rate), AUROC
- **Visualizations:** ROC curve (all 8 folds combined), Mahalanobis distance histograms per fold, misclassification analysis
- **Caching:** `outputs/ood_stats/ood_stats_{modality}_fold_s{test_subject}.pt` stores centroids, covariances, threshold

**Key findings:**
- Fusion model achieves ~98% AUROC with 83% OOD TPR and 5% FAR (excellent calibration)
- Action 9 shows subject-dependent failures (subjects 3-5 harder to detect)
- Actions 12, 24 have near-perfect detection rates

### Configs (`configs/`)
All configs (`sensor.yaml`, `video.yaml`, `fusion.yaml`) share schema: `seed`, `subset`, modality-specific paths, `model` (d_model, n_heads, n_layers, dropout), `training` (n_epochs=60, lr, weight_decay, batch_size, grad_clip), `wandb`. 

**Modality-specific:**
- `sensor.yaml`: `inertial_dir`, baseline sensor config
- `video.yaml`: `kalman_cache` (pose cache path), `drop_feats` (low-importance landmarks to exclude)
- `fusion.yaml`: both sensor and video paths, combined model config

**ID subset:** All configs define `subset: [1, 2, 4, 13, 19, 22, 23, 27]` (8 in-distribution actions). OOD actions 24, 9, 12 handled separately by `test_ood.py`.

## Assignment Parts

| Part | Script(s) | Status |
|------|-----------|--------|
| 1 | `part1_eda.py` | Done |
| 2 | `train.py`, `test.py`, `part2_*.py` | Done |
| 3 | `part3_fusion.py`, `part3_fusion_feature.py` | Done |
| 4A/4B | (part of fusion scripts) | Done |
| 5 | `part5_imbalance.py`, `part5_fusion.py` | Done |
| 6 | `part6_confidence.py` | Done |
| 7 | `test_ood.py` (Mahalanobis distance + two-tier alarm) | Done |
| 8 | Retrieval vs fine-tuning (written) | Not started |
| 9 | Production monitoring (written + diagram) | Not started |

## Checkpoint Structure

**Unified Directory:** All checkpoints (training and evaluation) use `checkpoints/final_eval/` with modality + experiment suffixes:

```
checkpoints/final_eval/
├── sensor/                      # 8 folds: baseline sensor model
├── video/                       # 8 folds: baseline video model
├── fusion/                      # 8 folds: baseline fusion (best: 96.9% accuracy)
├── fusion_imbalance/            # 8 folds: undersample majority class
├── fusion_imbalance_aug/        # 8 folds: + time/amplitude warp augmentation
├── fusion_imbalance_weighted/   # 8 folds: + weighted cross-entropy loss
└── sensor_imbalance_aug/        # 5+ folds: sensor variant
```

**Design:** 
- Training saves to `final_eval/{modality}{suffix}` (lines 381-389 in `train.py`)
- Evaluation loads from same location (lines 85-104 in `test.py`)
- Prevents overwrites: reviewer can train 1 epoch without affecting evaluation weights
- All variants isolated in separate subdirectories for easy comparison

**Name Mapping:** 
- No suffix = baseline (e.g., `final_eval/sensor/` is standard unimodal)
- `_imbalance` = undersample majority
- `_imbalance_aug` = + augmentation
- `_imbalance_weighted` = + weighted CE loss

## Key Constraints

- Use only RGB video + inertial (accel + gyro) modalities — not depth or skeleton.
- Reproducibility required: fix random seeds everywhere.
- Submission is a single ZIP with all code, docs, and `HOW_TO_RUN.md`.
- **Jupyter Compatible:** All test*.py scripts display plots inline when run in notebooks via `subprocess.run()`.
