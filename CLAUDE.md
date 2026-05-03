# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Senior AI Researcher home assignment for ELTA. Build a multimodal Human Activity Classification system using the **UTD-MHAD** dataset (RGB video + inertial sensors). The assignment has 9 parts covering EDA, unimodal models, fusion, missing-modality robustness, class imbalance, confidence estimation, OOD detection, retrieval vs fine-tuning (written), and production monitoring (written + diagram).

## Running Scripts

```bash
# Train unimodal models (LOSO cross-validation, logs to wandb)
python3 train_sensor.py --config configs/sensor.yaml
python3 train_video.py  --config configs/video.yaml

# Evaluate saved checkpoints
python3 test_sensor.py --config configs/sensor.yaml
python3 test_video.py  --config configs/video.yaml

# Part scripts (standalone, no wandb)
python3 part1_eda.py
python3 part2_video.py          # MediaPipe → Kalman → Transformer
python3 part2_classical.py
python3 part3_fusion.py         # Bayesian fusion
python3 part3_fusion_feature.py # Feature-MLP fusion
python3 part5_imbalance.py
python3 part5_fusion.py
python3 part6_confidence.py     # Temperature scaling + conformal prediction
```

All scripts use MPS (Apple Silicon) > CUDA > CPU automatically.

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
- Input `(B, T, in_dim)` → linear projection → positional encoding → Transformer encoder → mean pooling → MLP head → `(B, n_classes)`
- `in_dim=12` for sensor (6 IMU + 6 velocity), `in_dim=len(keep_feats)*2` for video (pose + velocity)

**`models/fusion.py`** — `FusionMLP` for late fusion with learned null embeddings:
- Concatenates video embedding `e_v` and sensor embedding `e_s` (both 64-dim)
- `null_v` / `null_s` learned parameters substitute a missing modality at inference
- `drop_v` / `drop_s` boolean masks enable modality dropout during training

**`data/sensor_dataset.py`**:
- `load_sensor_samples(inertial_dir, subset, label_map)` → `list[(subject, label, raw_iner)]` where `raw_iner` is `(T, 6)`
- `SensorDataset`: pads/truncates to `max_len=256`, adds velocity → `(max_len, 12)`, z-score normalizes
- `extract_imu_features(iner)` → 36-dim vector: per-channel × [mean, std, RMS, energy, dom_freq, spectral_entropy]

**`data/video_dataset.py`**:
- `load_video_samples(kalman_cache, subset, label_map, keep_feats)` → reads `.npy` files from `outputs/pose_cache_kalman22/`
- `PoseDataset`: adds velocity → `(T, pos_dim*2)`, z-score normalizes. No fixed-length padding (sequences have variable length)

**`data/augmentation.py`** — `time_warp` and `amplitude_warp` for IMU sequences (nonlinear time stretch and smooth amplitude scaling).

### Lightning Modules
`SensorLightningModule` (in `train_sensor.py`) and `VideoLightningModule` (in `train_video.py`) wrap `TransformerClassifier`, share identical structure: Adam optimizer + CosineAnnealingLR, cross-entropy loss, train/acc logging.

### Video Pipeline (part2_video.py)
MediaPipe Pose (33 landmarks × 2D) → Kalman filter smoothing → normalize to hip center + shoulder width → cache as `.npy` in `outputs/pose_cache_kalman22/`. The cache is the input to all subsequent video model training. `drop_feats` in `configs/video.yaml` lists 12 low-importance features identified by permutation importance.

### Fusion Strategies (part3)
- **Bayesian fusion** (`part3_fusion.py`): sensor determines wrist/thigh placement prior → gates video softmax probabilities
- **Feature-MLP fusion** (`part3_fusion_feature.py`): frozen encoders from Part 2 checkpoints feed into `FusionMLP`

### Part 6 — Confidence Estimation
`part6_confidence.py` uses a **3-way cyclic LOSO split**: fold `i` has test=s_i, cal=s_{(i+1) mod 8}, train=remaining 6. Retrains both models per fold, then applies:
1. **Temperature scaling**: `T* = argmin NLL(logits/T, y_cal)` via `scipy.optimize.minimize_scalar`
2. **Split conformal prediction**: non-conformity score `s = 1 − softmax(z/T*)[y_true]`, quantile gives coverage guarantee `P(y ∈ C(x)) ≥ 1−α`

### Configs (`configs/`)
Both `sensor.yaml` and `video.yaml` share the same schema: `seed`, `subset`, modality-specific data path, `model` (d_model, n_heads, n_layers, dropout), `training` (n_epochs=60, lr, weight_decay, batch_size, grad_clip), `wandb`. Video config additionally has `kalman_cache` and `drop_feats`.

## Assignment Parts

| Part | Script(s) | Status |
|------|-----------|--------|
| 1 | `part1_eda.py` | Done |
| 2 | `train_sensor.py`, `train_video.py`, `part2_*.py` | Done |
| 3 | `part3_fusion.py`, `part3_fusion_feature.py` | Done |
| 4A/4B | (part of fusion scripts) | Done |
| 5 | `part5_imbalance.py`, `part5_fusion.py` | Done |
| 6 | `part6_confidence.py` | Done |
| 7 | OOD detection | Not started |
| 8 | Retrieval vs fine-tuning (written) | Not started |
| 9 | Production monitoring (written + diagram) | Not started |

## Key Constraints

- Use only RGB video + inertial (accel + gyro) modalities — not depth or skeleton.
- Reproducibility required: fix random seeds everywhere.
- Submission is a single ZIP with all code, docs, and `HOW_TO_RUN.md`.
