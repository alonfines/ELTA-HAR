#!/usr/bin/env python3
"""Part 4a: Missing Modality Analysis — Frozen Fusion Weights"""

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from data.sensor_dataset import load_sensor_samples, compute_global_stats
from data.video_dataset import load_video_samples, compute_global_stats_video, get_landmark_indices
from data.fusion_dataset import extract_matched_keys
from models.fusion import FusionTransformerClassifier
from utils import load_config, load_action_names

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent
cfg = load_config(ROOT / "configs/fusion.yaml")
SUBSET = cfg.subset
label_map = {a: i for i, a in enumerate(SUBSET)}
action_names = load_action_names(ROOT / "Sample_Code")
CLASS_NAMES = [action_names[a] for a in SUBSET]

device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

INERTIAL_DIR = ROOT / cfg.inertial_dir
KALMAN_CACHE = ROOT / cfg.kalman_cache
CHECKPOINT_DIR = ROOT / "checkpoints/final_eval/fusion"
OUT_DIR = ROOT / "outputs"
OUT_DIR.mkdir(exist_ok=True)

sensor_samples = load_sensor_samples(INERTIAL_DIR, SUBSET, label_map)
video_samples = load_video_samples(KALMAN_CACHE, SUBSET, label_map)
matched_samples = extract_matched_keys(sensor_samples, video_samples)
subjects = sorted(set(s for s, _, _ in matched_samples))

def forward_with_missing_modality(model, x_sensor, x_video, missing_modality, device):
    model.eval()
    with torch.no_grad():
        if missing_modality == 'sensor':
            x_video = x_video.float().to(device)
            z_video = model.video_backbone.get_encoder_features(x_video)
            z_video_mean = z_video.mean(dim=1)
            z_video_max = z_video.max(dim=1)[0]
            z_sensor_mean = torch.zeros(z_video_mean.shape[0], 64, device=device, dtype=torch.float32)
            z_sensor_max = torch.zeros(z_video_max.shape[0], 64, device=device, dtype=torch.float32)
        else:
            x_sensor = x_sensor.float().to(device)
            z_sensor = model.sensor_backbone.get_encoder_features(x_sensor)
            z_sensor_mean = z_sensor.mean(dim=1)
            z_sensor_max = z_sensor.max(dim=1)[0]
            z_video_mean = torch.zeros(z_sensor_mean.shape[0], 64, device=device, dtype=torch.float32)
            z_video_max = torch.zeros(z_sensor_max.shape[0], 64, device=device, dtype=torch.float32)
        fusion_embedding = torch.cat([z_sensor_mean, z_sensor_max, z_video_mean, z_video_max], dim=1)
        logits = model.fusion_head(fusion_embedding)

    return logits

# ── Prepare data ───────────────────────────────────────────────────────────────
# Compute global stats for normalization
sensor_max_len = getattr(cfg, "max_len_sensor", 256)
video_max_len = getattr(cfg, "max_len_video", 128)
global_stats_sensor = compute_global_stats(sensor_samples)

landmark_set = getattr(cfg, "landmark_set", "hands_legs_hips")
landmark_indices = get_landmark_indices(landmark_set)
global_stats_video = compute_global_stats_video(video_samples, landmark_indices)

y_true_sensor_missing = []
y_pred_sensor_missing = []
y_true_video_missing = []
y_pred_video_missing = []

for test_subject in subjects:
    ckpt_path = CHECKPOINT_DIR / f"fold_s{test_subject}.pt"
    model = FusionTransformerClassifier(
        n_classes=len(SUBSET),
        in_dim_sensor=cfg.in_dim_sensor,
        in_dim_video=cfg.in_dim_video,
        d_model=cfg.model.d_model,
        n_heads=cfg.model.n_heads,
        n_layers=cfg.model.n_layers,
        dropout=cfg.model.dropout,
        d_fusion=cfg.model.d_fusion,
    )
    state = torch.load(str(ckpt_path), map_location=device, weights_only=True)
    model.load_state_dict(state, strict=False)
    model = model.to(device)
    model.eval()

    test_sensor_samples = [(label, raw) for s, label, raw in sensor_samples if s == test_subject]
    test_video_samples = [(label, raw) for s, label, raw in video_samples if s == test_subject]
    test_matched = [(label_s, raw_s, raw_v)
                    for (label_s, raw_s), (label_v, raw_v) in zip(test_sensor_samples, test_video_samples)
                    if label_s == label_v]

    from data.video_dataset import PoseDataset
    preds_sensor_missing, labels_sensor_missing = [], []
    for label, raw_sensor, raw_video in test_matched:
        pose_dataset = PoseDataset(
            [(label, raw_video)],
            max_len=video_max_len,
            landmark_set=landmark_set,
            normalization_type="global",
            global_stats=(global_stats_video[0], global_stats_video[1])
        )
        x_video, _ = pose_dataset[0]
        x_video = x_video.float().unsqueeze(0)
        logits = forward_with_missing_modality(model, None, x_video, 'sensor', device)
        pred = logits.argmax(dim=1).cpu().numpy()[0]
        preds_sensor_missing.append(pred)
        labels_sensor_missing.append(label)

    y_true_sensor_missing.extend(labels_sensor_missing)
    y_pred_sensor_missing.extend(preds_sensor_missing)

    from data.sensor_dataset import SensorDataset
    preds_video_missing, labels_video_missing = [], []
    for label, raw_sensor, raw_video in test_matched:
        sensor_dataset = SensorDataset(
            [(label, raw_sensor)],
            max_len=sensor_max_len,
            feature_type="raw+velocity",
            normalization_type="global",
            global_stats=(global_stats_sensor[0], global_stats_sensor[1])
        )
        x_sensor, _ = sensor_dataset[0]
        x_sensor = x_sensor.float().unsqueeze(0)
        logits = forward_with_missing_modality(model, x_sensor, None, 'video', device)
        pred = logits.argmax(dim=1).cpu().numpy()[0]
        preds_video_missing.append(pred)
        labels_video_missing.append(label)

    y_true_video_missing.extend(labels_video_missing)
    y_pred_video_missing.extend(preds_video_missing)

acc_sensor_missing = accuracy_score(y_true_sensor_missing, y_pred_sensor_missing)
f1_sensor_missing = f1_score(y_true_sensor_missing, y_pred_sensor_missing, average="macro", zero_division=0)
acc_video_missing = accuracy_score(y_true_video_missing, y_pred_video_missing)
f1_video_missing = f1_score(y_true_video_missing, y_pred_video_missing, average="macro", zero_division=0)

cm_sensor_missing = confusion_matrix(y_true_sensor_missing, y_pred_sensor_missing, labels=range(len(SUBSET)))
cm_video_missing = confusion_matrix(y_true_video_missing, y_pred_video_missing, labels=range(len(SUBSET)))

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

cm_norm_sm = cm_sensor_missing.astype(float) / cm_sensor_missing.sum(axis=1, keepdims=True)
im1 = axes[0].imshow(cm_norm_sm, cmap="Blues", vmin=0, vmax=1)
axes[0].set_xticks(range(len(SUBSET)))
axes[0].set_xticklabels(CLASS_NAMES, rotation=45, ha="right", fontsize=9)
axes[0].set_yticks(range(len(SUBSET)))
axes[0].set_yticklabels(CLASS_NAMES, fontsize=9)
axes[0].set_xlabel("Predicted", fontsize=11)
axes[0].set_ylabel("True", fontsize=11)
axes[0].set_title(f"Sensor Missing (Video Only)\nAcc={acc_sensor_missing:.3f} | F1={f1_sensor_missing:.3f}", fontweight="bold", fontsize=12)
for i in range(len(SUBSET)):
    for j in range(len(SUBSET)):
        axes[0].text(j, i, f"{cm_norm_sm[i,j]:.2f}", ha="center", va="center",
                    fontsize=7, color="white" if cm_norm_sm[i,j] > 0.5 else "black")
plt.colorbar(im1, ax=axes[0], fraction=0.046)

cm_norm_vm = cm_video_missing.astype(float) / cm_video_missing.sum(axis=1, keepdims=True)
im2 = axes[1].imshow(cm_norm_vm, cmap="Blues", vmin=0, vmax=1)
axes[1].set_xticks(range(len(SUBSET)))
axes[1].set_xticklabels(CLASS_NAMES, rotation=45, ha="right", fontsize=9)
axes[1].set_yticks(range(len(SUBSET)))
axes[1].set_yticklabels(CLASS_NAMES, fontsize=9)
axes[1].set_xlabel("Predicted", fontsize=11)
axes[1].set_ylabel("True", fontsize=11)
axes[1].set_title(f"Video Missing (Sensor Only)\nAcc={acc_video_missing:.3f} | F1={f1_video_missing:.3f}", fontweight="bold", fontsize=12)
for i in range(len(SUBSET)):
    for j in range(len(SUBSET)):
        axes[1].text(j, i, f"{cm_norm_vm[i,j]:.2f}", ha="center", va="center",
                    fontsize=7, color="white" if cm_norm_vm[i,j] > 0.5 else "black")
plt.colorbar(im2, ax=axes[1], fraction=0.046)

plt.tight_layout()
path = OUT_DIR / "part4a_missing_modality.png"
plt.savefig(path, dpi=150, bbox_inches="tight")
plt.show()
