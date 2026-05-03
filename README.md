# ELTA HAR: Multimodal Human Activity Recognition

Senior AI Researcher home assignment: Transformer-based activity classification from RGB video and inertial sensors using the UTD-MHAD dataset.

## Quick Start

```bash
# Train models (LOSO cross-validation)
python3 train.py --modality sensor --no-validation
python3 train.py --modality video --no-validation

# Evaluate models
python3 test.py --modality sensor
python3 test.py --modality video
```

## Results (Corrected, Valid Metrics)

| Modality | Accuracy | F1-Score | Notes |
|----------|----------|----------|-------|
| **Sensor Transformer** | 92.1% ± 7.8% | 90.9% ± 9.5% | IMU-only (accel + gyro) |
| **Video Transformer** | 89.4% ± 11.5% | 88.3% ± 12.4% | Pose-only (MediaPipe landmarks) |

## Critical Bugs Fixed (Session 3)

### Bug 1: Wrong Wrist Landmarks
- **Issue**: Feature engineering used mouth corners [9,10] instead of wrists [15,16]
- **Impact**: "Wrist deltas" were measuring facial movement, not hand motion
- **Fix**: Corrected landmark indices in `data/video_dataset.py` line 32
- **Result**: Swipe right accuracy improved from 0.66 → 0.81

### Bug 2: Data Leakage in Global Statistics
- **Issue**: Computing normalization stats on ALL samples (including test subject)
- **Impact**: Violated LOSO cross-validation protocol, invalid metrics
- **Fix**: Moved `compute_global_stats()` inside fold loops → compute on training data only
- **Files**: `train.py` and `test.py`
- **Result**: Restored strict data isolation, valid metrics

## Architecture Improvements

### Phase 1: Critical Fixes
- ✅ **Padding Contamination**: Normalize before padding (already correct)
- ✅ **Attention Masking**: `src_key_padding_mask` in TransformerEncoder (already implemented)
- ✅ **Drop Features**: Removed unused drop_feats parameter

### Phase 2: Fairness & Parity
- ✅ **Temporal Windows**: Both modalities use 256 frames (~5 sec temporal context)
- ✅ **Sensor Velocity**: Added velocity features (6 → 12-dim, matching video)
- ✅ **Zero Augmentation**: Clean data without synthetic noise
- ✅ **Global Normalization**: Preserves amplitude differences (amplitude matters for action intensity)

## Dataset

**UTD-MHAD Dataset**
- 27 action classes across 8 subjects
- 8-class subset used: [1, 2, 4, 13, 19, 22, 23, 27]
- LOSO cross-validation (8 folds)

**Modalities**
- **Inertial**: 6-axis IMU (3-axis accel + 3-axis gyro) at ~50Hz → 256-frame sequences
- **Video**: RGB color video at ~30fps → 256-frame sequences with MediaPipe Pose (33 landmarks)

## Model Architecture

**TransformerClassifier**
```
Input (B, T, in_dim)
  ↓
Linear Projection → d_model=64
  ↓
Positional Encoding
  ↓
Transformer Encoder (2 layers, 4 heads)
  ↓
Mean Pooling + Max Pooling
  ↓
MLP Head (64 → 64 → n_classes)
  ↓
Output (B, n_classes)
```

**Input Dimensions**
- Sensor: 12-dim (6 raw IMU + 6 velocity)
- Video: 98-dim (48 pose coordinates + 48 velocity + 2 raw wrist deltas)

## Code Structure

```
├── train.py                    # Training script (LOSO CV)
├── test.py                     # Evaluation & confusion matrices
├── configs/
│   ├── sensor.yaml            # Sensor model config
│   └── video.yaml             # Video model config
├── data/
│   ├── sensor_dataset.py      # IMU data loading & preprocessing
│   ├── video_dataset.py       # Pose data loading & preprocessing
│   └── augmentation.py        # Data augmentation (disabled)
├── models/
│   ├── model.py               # TransformerClassifier
│   └── fusion.py              # Late fusion for multimodal
├── checkpoints/               # Trained .pt weights (8 folds each)
├── outputs/                   # Confusion matrices & analysis
└── part*.py                   # Assignment parts (EDA, fusion, confidence, OOD)
```

## Configurations

### Sensor Config (`configs/sensor.yaml`)
```yaml
feature_type: raw+velocity     # 12-dim features
normalization_type: global     # Preserve amplitude
in_dim: 12
max_len: 256                   # Pad/truncate length
n_epochs: 60
batch_size: 16
```

### Video Config (`configs/video.yaml`)
```yaml
landmark_set: hands_legs_hips  # 24 landmarks → 98-dim
normalization_type: global     # Preserve pose amplitude
in_dim: 98                      # 48 pos + 48 vel + 2 wrist deltas
max_len: 256                    # Harmonized with sensor
n_epochs: 60
batch_size: 16
```

## Key Features

### Wrist Delta Features (Video)
- **Purpose**: Preserve directional information for swipe detection
- **Method**: Raw, unnormalized frame-to-frame x-differences for wrists [15, 16]
- **Impact**: Swipe left/right distinction critical signal
- **Implementation**: `extract_wrist_deltas()` in `data/video_dataset.py`

### Global Normalization
- **Preserves**: Amplitude differences (soft knock vs hard punch)
- **Computed**: Per-fold on training data only (no test leakage)
- **Applied**: Consistently to both train and test data within each fold

### Attention Masking
- **Purpose**: Prevent model from attending to padding
- **Method**: `src_key_padding_mask` in `nn.TransformerEncoder`
- **Effect**: Real frames get attention, padding frames are masked out

## Previous Invalid Results

The previous confusion matrix (83.8% ± 15.5% for video) was **invalid** because:
1. Wrong landmarks measured facial motion, not hand dynamics
2. Global stats included test subject data (protocol violation)

These results have been discarded and replaced with corrected metrics above.

## Assignment Parts

- **Part 1**: EDA - `part1_eda.py`
- **Part 2**: Unimodal models - `train.py`, `test.py`, sensor/video configs
- **Part 3**: Multimodal fusion - `part3_fusion.py`, `part3_fusion_feature.py`
- **Part 5**: Class imbalance - `part5_imbalance.py`, `part5_fusion.py`
- **Part 6**: Confidence estimation - `part6_confidence.py`
- **Part 7**: OOD detection - `part7_ood.py`
- **Parts 8-9**: Written components - `part8_9_written.txt`

## Training Time

- Sensor: ~1 hour (8 folds × 60 epochs)
- Video: ~1.5 hours (8 folds × 60 epochs)
- **Total**: ~2.5 hours on Apple Silicon (MPS)

## Requirements

```
torch
pytorch-lightning
wandb
scikit-learn
numpy
pandas
matplotlib
mediapipe
pyyaml
```

## Notes

- Uses MPS (Apple Silicon) > CUDA > CPU automatically
- Seeds fixed at 42 for reproducibility
- LOSO cross-validation ensures per-subject generalization
- All metrics reported as mean ± std over 8 folds
