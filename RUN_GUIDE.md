# Quick Start Guide

## Running the Scripts

### **1. Evaluate Sensor Model (Inference Only)**
```bash
python3 test.py --modality sensor
```
- Loads pre-trained checkpoints
- Runs LOSO evaluation on all 8 subjects
- Plots confusion matrix + failure case
- No training required
- Output: `outputs/test_sensor_confusion.png` + `outputs/test_sensor_failure.png`

### **2. Train Sensor Model**
```bash
python3 train.py --modality sensor --config configs/sensor.yaml
```
- Trains Transformer model with LOSO cross-validation
- Logs to Weights & Biases (wandb)
- Saves checkpoints as `.pt` files
- Generates training confusion matrix
- Takes ~30-60 minutes for full 8 folds

### **3. Train Video Model**
```bash
python3 part2_video.py  # Generate pose cache first
python3 train.py --modality video --config configs/video.yaml
```
- First: Extract poses from videos using MediaPipe + Kalman smoothing
- Then: Train Transformer model on video data
- Requires RGB video files in `RGB-part1/`, `RGB-part2/`, etc.

### **4. Evaluate Video Model**
```bash
python3 test.py --modality video
```
- Requires video pose cache to exist
- Runs LOSO evaluation on video model

## Verify Your Setup

```bash
python3 verify_setup.py
```
This will:
- Check all dependencies
- Verify train.py initializes correctly
- Run test.py sensor evaluation to completion
- Test error handling for missing video cache
- Provide a summary report

## Troubleshooting

### "No video samples found" error
- You're trying to use video modality but the pose cache doesn't exist
- Either: Run `python3 part2_video.py` first, or use sensor modality instead

### Jupyter notebook issues
- If you get mixed error messages in Jupyter, restart the kernel
- Or run from terminal instead: `python3 test.py --modality sensor`

### Out of memory error
- Reduce batch size in `configs/sensor.yaml` or `configs/video.yaml`
- Change `training.batch_size: 32` to `16` or `8`

## Output Files

After running, you'll have:

```
outputs/
├── test_sensor_confusion.png      # Confusion matrix (sensor model)
├── test_sensor_failure.png        # Failure case visualization
├── sensor_loso_confusion.png      # Training confusion matrix
├── video_loso_confusion.png       # Video model confusion matrix (if trained)
└── pose_cache_kalman22/           # Cached video poses (if generated)
    ├── a1_s1_t1_poses.npy
    ├── a1_s1_t2_poses.npy
    └── ...
```

## Model Checkpoints

Training saves checkpoints in:
```
checkpoints/
├── sensor/
│   ├── fold_s1.pt
│   ├── fold_s2.pt
│   └── ... (one per subject)
└── video/
    ├── fold_s1.pt
    ├── fold_s2.pt
    └── ...
```

Each `.pt` file is ~550KB (lightweight model weights).

## Performance Expectations

**Sensor Model** (Transformer)
- Accuracy: ~91.3% ± 7.5%
- Macro F1: 0.903 ± 0.088
- Training time: ~40-60 minutes (8 folds, 60 epochs each)

**Video Model** (Transformer)
- TBD (depends on video preprocessing setup)

**Baseline** (Random Forest on hand-crafted features)
- Accuracy: ~94.1% ± 4.9%
- Macro F1: 0.941 ± 0.053
