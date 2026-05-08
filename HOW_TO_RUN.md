# How to Run: ELTA Human Activity Recognition Assignment

## Overview

This assignment is structured as **Jupyter notebooks and Python scripts** - `ELTA_HAR_Assignment.ipynb` walks through all 9 parts of the HAR (Human Activity Recognition) system. The complete assignment is also available as a **PDF version** for review at `ELTA_HAR_Assignment.pdf`.

---

## File Structure

```
ELTA Home Assignment/
├── HOW_TO_RUN.md                          ← You are here
├── ELTA_HAR_Assignment.pdf                ← PDF version (all parts + results)
├── CLAUDE.md                              ← Architecture guide
│
├── 📓 Jupyter Notebooks (Main execution)
│   ├── ELTA_HAR_Assignment.ipynb          ← Complete walkthrough (all 9 parts)
│   └── (Generated outputs: .ipynb_checkpoints/)
│
├── 🐍 Python Scripts (Modular execution)
│   ├── train.py                           ← Train models (sensor/video/fusion)
│   ├── test.py                            ← Evaluate models
│   ├── test_ood.py                        ← Part 7: OOD detection
│   ├── part1_eda.py                       ← Part 1: EDA
│   ├── part2_video.py                     ← Part 2: Video preprocessing
│   ├── part2_classical.py                 ← Part 2: Classical features
│   └── part4_a.py                         ← Part 4a: Missing modality
│
├── 📋 Configuration Files
│   └── configs/
│       ├── sensor.yaml                    ← Sensor model config
│       ├── video.yaml                     ← Video model config
│       └── fusion.yaml                    ← Fusion model config
│
├── 🧠 Model Code
│   └── models/
│       ├── model.py                       ← TransformerClassifier (sensor + video)
│       └── fusion.py                      ← FusionTransformerClassifier
│
├── 📊 Data Loading
│   └── data/
│       ├── sensor_dataset.py              ← IMU data loader
│       ├── video_dataset.py               ← Video pose data loader
│       ├── fusion_dataset.py              ← Fusion data loader
│       ├── augmentation.py                ← Data augmentation
│       └── __init__.py
│
├── 🛠️ Utilities
│   └── utils.py                           ← Config loading, metrics, visualization
│
├── 📁 Data Directories
│   ├── Inertial/                          ← Raw IMU .mat files (27 actions × 8 subjects)
│   ├── RGB-part1/ to RGB-part4/           ← Raw video .avi files
│   ├── Sample_Code/                       ← Action names reference
│   └── outputs/pose_cache_kalman22/       ← Processed pose landmarks (.npy)
│
├── 💾 Checkpoints
│   └── checkpoints/
│       └── final_eval/
│           ├── sensor/                    ← 8 sensor model folds
│           ├── video/                     ← 8 video model folds
│           ├── fusion/                    ← 8 fusion model folds
│           ├── fusion_imbalance/          ← Imbalanced data variants
│           ├── fusion_imbalance_aug/
│           ├── fusion_imbalance_weighted/
│           └── conformal/                 ← Conformal prediction folds
│
├── 📈 Outputs & Results
│   └── outputs/
│       ├── *.png                          ← Confusion matrices, ROC curves, etc.
│       ├── ood_stats/                     ← Part 7 cached statistics
│       └── pose_cache_kalman22/           ← Part 2 pose landmarks
│
└── 📄 Documentation
    ├── README.md                          ← Assignment overview

```

---

## Quick Start: Run Everything

### Option 1: Jupyter Notebook (Recommended for this assignment)

**The complete assignment is in `ELTA_HAR_Assignment.ipynb`** — a comprehensive notebook that:
- Walks through all 9 parts step-by-step
- Includes code cells (executable)
- Displays results inline (plots, confusion matrices, tables)
- Matches the PDF version exactly

**To run:**

```bash
# 1. Install dependencies
pip install jupyter jupyterlab
pip install torch torchvision torchaudio
pip install numpy pandas matplotlib scipy scikit-learn
pip install pyyaml pytorch-lightning wandb

# 2. Start Jupyter
jupyter notebook

# 3. Open ELTA_HAR_Assignment.ipynb
# 4. Click "Kernel" → "Run All Cells"
# 5. Wait for all parts to execute (30-60 minutes depending on GPU)
```

**In Jupyter:**
- All plots display inline ✓
- All confusion matrices show up ✓
- All metrics printed to cell output ✓
- Easy to modify and re-run individual parts

---

### Option 2: Individual Python Scripts (For Experimentation)

Run specific parts independently:

```bash
# Part 1: Exploratory Data Analysis
python3 part1_eda.py

# Part 2: Train models (LOSO cross-validation)
python3 train.py --modality sensor      # Sensor-only model
python3 train.py --modality video       # Video-only model
python3 train.py --modality fusion      # Fusion model

# Part 2: Evaluate models
python3 test.py --modality sensor
python3 test.py --modality video
python3 test.py --modality fusion

# Part 3: Fusion analysis
python3 part3_fusion.py
python3 part3_fusion_feature.py

# Part 4a: Missing modality
python3 part4_a.py

# Part 5: Class imbalance
python3 part5_imbalance.py
python3 part5_fusion.py

# Part 6: Conformal prediction
python3 part6_confidence.py

# Part 7: OOD detection
python3 test_ood.py --modality fusion --ood-actions 24 9 12
```

---

## Part-by-Part Breakdown

### Part 1: Exploratory Data Analysis
**Location**: `part1_eda.py` or Jupyter notebook cells
**What it does**:
- Loads UTD-MHAD dataset (27 actions, 8 subjects, 4 trials each)
- Plots action distributions, sensor traces, video frames
- Computes basic statistics

**Output**: `outputs/part1_*.png` (distributions, sample plots)

---

### Part 2: Unimodal Models (Sensor & Video)
**Scripts**: `train.py --modality sensor/video`, `test.py --modality sensor/video`
**What it does**:
- Trains TransformerClassifier on sensor data (12-dim IMU + velocity)
- Trains TransformerClassifier on video data (48-dim pose landmarks)
- LOSO cross-validation (8 folds, one subject left out per fold)
- Reports accuracy, F1, per-action breakdown

**Output**: 
- Checkpoints: `checkpoints/final_eval/sensor/fold_s*.pt`, `checkpoints/final_eval/video/fold_s*.pt`
- Metrics: Confusion matrices saved to `outputs/`

**Expected Results**: 
- Sensor: ~85% accuracy
- Video: ~80% accuracy

---

### Part 3: Fusion (Multimodal)
**Scripts**: `train.py --modality fusion`, `test.py --modality fusion`
**Related**: `part3_fusion.py`, `part3_fusion_feature.py` (analysis)
**What it does**:
- Trains FusionTransformerClassifier (sensor + video combined)
- Late fusion: concatenate [sensor_mean, sensor_max, video_mean, video_max] → 256-dim
- Passes through fusion MLP head
- LOSO evaluation

**Output**: `checkpoints/final_eval/fusion/fold_s*.pt`
**Expected Results**: ~96.9% accuracy (best performance)

---

### Part 4a: Missing Modality Analysis
**Script**: `part4_a.py`
**What it does**:
- Shows what happens when one modality is missing at inference
- Evaluates with sensor only (zero-pad video)
- Evaluates with video only (zero-pad sensor)
- Compares to full fusion baseline

**Output**: `outputs/part4a_missing_modality.png` (side-by-side confusion matrices)

---

### Part 5: Class Imbalance & Augmentation
**Scripts**: `part5_imbalance.py`, `part5_fusion.py`
**Training variants**:
```bash
python3 train.py --modality fusion --imbalance              # Undersample
python3 train.py --modality fusion --imbalance --augment_minority  # + augmentation
python3 train.py --modality fusion --imbalance --weighted_loss    # + weighted CE
```

**What it does**:
- Handles imbalanced action classes (actions 19, 22 are minority)
- Three mitigation strategies:
  1. Undersampling majority class
  2. Data augmentation (time warp, amplitude scaling)
  3. Weighted cross-entropy loss

**Output**: Multiple checkpoint variants in `checkpoints/final_eval/`

---

### Part 6: Confidence Estimation & Conformal Prediction
**Script**: `part6_confidence.py`
**What it does**:
- 3-way split: train (6 subjects) + calibrate (1 subject) + test (1 subject)
- Temperature scaling: calibrate softmax confidence
- Conformal prediction: compute prediction sets with ~95% coverage guarantee
- Non-conformity score: how far true label is from predicted

**Output**: Conformal prediction sets and coverage metrics

---

### Part 7: OOD Detection (Mahalanobis Distance)
**Script**: `test_ood.py`
**What it does**:
- Detects out-of-distribution samples (actions 24, 9, 12 never seen in training)
- Uses Mahalanobis distance on learned embeddings (256-dim latent space)
- Three phases: Train (compute centroids), Calibrate (set threshold), Test (evaluate)
- Metrics: False Alarm Rate (FAR), OOD True Positive Rate (TPR), AUROC

**Commands**:
```bash
python3 test_ood.py --modality fusion --ood-actions 24 9 12
```

**Output**: 
- `outputs/ood_roc_fusion.png` — ROC curve with operating points
- `outputs/ood_confusion_fusion.png` — Binary confusion matrix (ID vs OOD)
- Metrics table with per-action breakdown

**Results**: AUROC 0.977, FAR 0.063, OOD TPR 0.832 (but Subject 2 fails at 0.24 TPR)

---

### Part 8: Retrieval vs Fine-Tuning (Written)
**Location**: `PART8_RETRIEVAL_VS_FINETUNING.md`
**What it covers**:
- When to use retrieval (simple, repetitive data)
- When to use fine-tuning (complex, variable data)
- Advantages and risks of each approach
- Application to HAR (why fine-tuning is necessary)

**No code** — written explanation with examples

---

### Part 9: Production Monitoring & Drift Detection (Written + Design)
**Location**: `PART9_PRODUCTION_MONITORING.md`
**What it covers**:
- Metrics to monitor (accuracy/F1 on labeled data, confidence on all data)
- Drift detection methods:
  - Mahalanobis distance (OOD rate)
  - Conformal prediction set sizes
  - Embedding space drift (Wasserstein distance)
- Retraining triggers (accuracy drop, OOD spike, uncertainty growing)
- System architecture (sensors → logging → drift detection → retraining → deployment)

**Includes**: Block diagram of production system

---

## Environment Setup

### Requirements
- **Python**: 3.8+
- **GPU**: Recommended (NVIDIA CUDA) but not required (CPU works, slower)
- **RAM**: 8GB+ (for embeddings and caching)

### Installation

```bash
# 1. Clone or download the project
cd /Users/alonfines/Desktop/ELTA\ Home\ Assignment

# 2. Create virtual environment (optional but recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# Or install manually:
pip install torch torchvision torchaudio
pip install numpy pandas matplotlib scipy scikit-learn
pip install pyyaml pytorch-lightning wandb
pip install jupyterlab ipykernel
```

### Device Auto-Detection
All scripts automatically detect GPU:
```python
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
```
- **MPS** (Apple Silicon): Fast on M1/M2 Mac
- **CUDA** (NVIDIA GPU): Fast on GPUs
- **CPU**: Works but slow (~2-3x slower)

---

## Running in Jupyter vs Command Line

### Jupyter Notebook (Recommended)
**Pros**:
- ✓ Inline visualizations (plots display in notebook)
- ✓ Easy to modify and re-run cells
- ✓ Clear output formatting
- ✓ All results visible in one place

**Cons**:
- Takes longer (30-60 min for full run)
- Requires Jupyter server

```bash
jupyter notebook ELTA_HAR_Assignment.ipynb
```

---

### Command Line Scripts (For CI/Testing)
**Pros**:
- ✓ Fast to run individual parts
- ✓ Easy to script/automate
- ✓ Good for experimentation

**Cons**:
- Plots saved to disk (not displayed inline)
- Must run each script separately

```bash
python3 train.py --modality fusion
python3 test.py --modality fusion
python3 test_ood.py --modality fusion --ood-actions 24 9 12
```

---

## Key Commands Reference

### Training
```bash
# Train all modalities (takes 1-2 hours with GPU)
python3 train.py --modality sensor
python3 train.py --modality video
python3 train.py --modality fusion

# With variants
python3 train.py --modality fusion --imbalance
python3 train.py --modality fusion --imbalance --augment_minority
python3 train.py --modality fusion --imbalance --weighted_loss
```

### Evaluation
```bash
# Evaluate all modalities
python3 test.py --modality sensor
python3 test.py --modality video
python3 test.py --modality fusion

# With conformal prediction
python3 test.py --modality fusion --conformal

# OOD detection
python3 test_ood.py --modality fusion --ood-actions 24 9 12
python3 test_ood.py --modality fusion --ood-actions 24 9 12 --force-recache
```

### Individual Parts
```bash
python3 part1_eda.py           # EDA
python3 part2_video.py         # Video preprocessing
python3 part3_fusion.py        # Fusion analysis
python3 part4_a.py             # Missing modality
python3 part5_imbalance.py     # Imbalance analysis
python3 part6_confidence.py    # Conformal prediction
python3 test_ood.py            # OOD detection
```

---

## PDF Version

**`ELTA_HAR_Assignment.pdf`** contains:
- Complete notebook rendered as PDF
- All code cells with outputs
- All plots and visualizations
- All results and metrics tables

**Use for**:
- Reading without running code
- Reviewing results
- Offline reference
- Printing

---

## Output Files

After running, you'll have:

```
outputs/
├── confusion_matrices/
│   ├── sensor_confusion.png
│   ├── video_confusion.png
│   └── fusion_confusion.png
├── part1_eda.png               # Part 1 distributions
├── part4a_missing_modality.png # Part 4a analysis
├── part5_*.png                 # Part 5 results
├── ood_roc_fusion.png          # Part 7 ROC curve
├── ood_confusion_fusion.png    # Part 7 confusion matrix
├── ood_stats/                  # Part 7 cached statistics
└── pose_cache_kalman22/        # Part 2 pose landmarks (1.5GB)

checkpoints/
└── final_eval/
    ├── sensor/fold_s*.pt       # 8 folds
    ├── video/fold_s*.pt
    ├── fusion/fold_s*.pt
    ├── fusion_imbalance/
    ├── fusion_imbalance_aug/
    ├── fusion_imbalance_weighted/
    └── conformal/fold_s*.pt
```

---

## Troubleshooting

### "CUDA out of memory"
```bash
# Use CPU instead
# Edit script: change device to torch.device("cpu")
# Or set batch_size=8 in config
```

### "File not found: Inertial/..."
- Ensure raw dataset files are in project root
- Run from project directory: `cd /Users/alonfines/Desktop/ELTA\ Home\ Assignment`

### "Checkpoint not found"
- Checkpoints must be pre-trained
- Run `python3 train.py --modality fusion` first
- Or download from assignment submission

### Notebook kernels dying
- Reduce batch_size in configs (8 → 4)
- Run fewer folds (only s1-s4)
- Use command-line scripts instead

---

## Summary

**To run the complete assignment:**

1. **Jupyter (All-in-one)**:
   ```bash
   jupyter notebook ELTA_HAR_Assignment.ipynb
   # Run all cells
   ```

2. **Command-line (Modular)**:
   ```bash
   python3 train.py --modality fusion
   python3 test.py --modality fusion
   python3 test_ood.py --modality fusion --ood-actions 24 9 12
   ```

3. **PDF (Review)**:
   - Open `ELTA_HAR_Assignment.pdf` to see complete results

All code is in the **Jupyter notebook and Python scripts**. The PDF is a rendered version for review.

---

## Documentation

For detailed explanations, see:
- `CLAUDE.md` — Architecture guide
- `PART7_OOD_DETECTION.md` — OOD detection methodology
- `PART8_RETRIEVAL_VS_FINETUNING.md` — Retrieval vs fine-tuning
- `PART9_PRODUCTION_MONITORING.md` — Production system design
- `README.md` — Assignment overview
