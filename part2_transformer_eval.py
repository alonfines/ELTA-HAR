"""
Part 2 — Transformer Sensor Model Evaluation (Load Weights Only)

Loads one pre-trained Transformer checkpoint per LOSO fold and evaluates on the
held-out subject. Does NOT train — just loads weights and infers.

Prerequisites: checkpoints/sensor/fold_s{subject}.pt for each subject in the
dataset (produced by train_sensor.py).

Run: python3 part2_transformer_eval.py
"""

import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from data.sensor_dataset import SensorDataset, load_sensor_samples
from models.model import TransformerClassifier
from utils import load_action_names

warnings.filterwarnings("ignore")

# ── Load config ────────────────────────────────────────────────────────────────
with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

SUBSET = cfg["subset"]

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
INERTIAL_DIR = ROOT / cfg["paths"]["inertial_dir"]
SAMPLE_DIR = ROOT / cfg["paths"]["sample_dir"]
OUT_DIR = ROOT / cfg["paths"]["output_dir"]
CHECKPOINT_DIR = ROOT / cfg["paths"]["checkpoints_dir"] / "sensor"
OUT_DIR.mkdir(exist_ok=True)

# Per-fold checkpoint naming. Adjust to match what train_sensor.py writes.
def fold_ckpt_path(subject: int) -> Path:
    return CHECKPOINT_DIR / f"fold_s{subject}.pt"

# ── Setup device ───────────────────────────────────────────────────────────────
device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Device: {device}")

# ── Load action names ──────────────────────────────────────────────────────────
action_names = load_action_names(SAMPLE_DIR)
CLASS_NAMES = [action_names[a] for a in SUBSET]
label_map = {a: i for i, a in enumerate(SUBSET)}

# ── Load dataset ───────────────────────────────────────────────────────────────
samples = load_sensor_samples(INERTIAL_DIR, SUBSET, label_map)
subjects = sorted(set(s for s, _, _ in samples))

# ── Verify all per-fold checkpoints exist before starting ──────────────────────
missing = [s for s in subjects if not fold_ckpt_path(s).exists()]
if missing:
    print(f"\n❌ Missing per-fold checkpoints for subject(s): {missing}")
    print(f"   Expected files in: {CHECKPOINT_DIR}")
    print(f"   Pattern: fold_s{{subject}}.pt   (one per held-out subject)")
    print("   Run training first:  python3 train_sensor.py --config configs/sensor.yaml")
    sys.exit(1)

# ── Build a model shell once; reload weights per fold ──────────────────────────
def build_model() -> TransformerClassifier:
    return TransformerClassifier(
        in_dim=12,                      # 6 IMU channels + 6 velocity
        n_classes=len(SUBSET),
        d_model=cfg["sensor"]["model"]["d_model"],
        n_heads=cfg["sensor"]["model"]["n_heads"],
        n_layers=cfg["sensor"]["model"]["n_layers"],
        dropout=cfg["sensor"]["model"]["dropout"],
    )

model = build_model().to(device)

# ── LOSO inference (one checkpoint per fold) ───────────────────────────────────
print("\nRunning LOSO inference (loading per-fold checkpoints)...")

y_true_all: list[int] = []
y_pred_all: list[int] = []
fold_accuracies: list[float] = []
fold_f1_scores: list[float] = []

with torch.no_grad():
    for test_subject in subjects:
        ckpt_file = fold_ckpt_path(test_subject)
        state = torch.load(ckpt_file, map_location=device, weights_only=True)
        model.load_state_dict(state)
        model.eval()

        test_samples = [(label, raw) for s, label, raw in samples if s == test_subject]
        test_data = SensorDataset(test_samples, max_len=cfg["sensor"]["max_len"])
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=False)

        y_fold_true: list[int] = []
        y_fold_pred: list[int] = []

        for x, y in test_loader:
            x = x.to(device)
            logits = model(x)
            preds = logits.argmax(dim=1).cpu().numpy()
            y_np = y.numpy()
            y_fold_true.extend(y_np)
            y_fold_pred.extend(preds)

        # Per-fold metrics
        fold_acc = accuracy_score(y_fold_true, y_fold_pred)
        fold_f1 = f1_score(y_fold_true, y_fold_pred, average="macro")
        fold_accuracies.append(fold_acc)
        fold_f1_scores.append(fold_f1)

        # Pool for confusion matrix
        y_true_all.extend(y_fold_true)
        y_pred_all.extend(y_fold_pred)


y_true_all = np.array(y_true_all)
y_pred_all = np.array(y_pred_all)

# ── Aggregate (mean ± std over folds — consistent across acc and F1) ───────────
acc_mean, acc_std = float(np.mean(fold_accuracies)), float(np.std(fold_accuracies))
f1_mean, f1_std = float(np.mean(fold_f1_scores)), float(np.std(fold_f1_scores))

print(
    f"\n✓ Transformer (Sensor) — "
    f"Accuracy: {acc_mean:.3f} ± {acc_std:.3f} | "
    f"Macro F1: {f1_mean:.3f} ± {f1_std:.3f}"
)

# ── Confusion matrix (pooled predictions across folds) ─────────────────────────
fig, ax = plt.subplots(figsize=(10, 8))
cm = confusion_matrix(y_true_all, y_pred_all, labels=list(range(len(SUBSET))))
cm_norm = cm.astype(float) / np.clip(cm.sum(axis=1, keepdims=True), 1, None)

im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
ax.set_xticks(range(len(SUBSET)))
ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right", fontsize=9)
ax.set_yticks(range(len(SUBSET)))
ax.set_yticklabels(CLASS_NAMES, fontsize=9)
ax.set_xlabel("Predicted", fontsize=11)
ax.set_ylabel("True", fontsize=11)
ax.set_title(
    f"Transformer (Sensor) — "
    f"Accuracy={acc_mean:.3f}±{acc_std:.3f} | "
    f"F1={f1_mean:.3f}±{f1_std:.3f}",
    fontweight="bold",
    fontsize=12,
)

for i in range(len(SUBSET)):
    for j in range(len(SUBSET)):
        ax.text(
            j, i, f"{cm_norm[i, j]:.2f}",
            ha="center", va="center", fontsize=8,
            color="white" if cm_norm[i, j] > 0.5 else "black",
        )

plt.colorbar(im, ax=ax, fraction=0.046)
plt.tight_layout()
path = OUT_DIR / "part2_transformer_confusion.png"
plt.savefig(path, dpi=150)
plt.show()
print(f"\n✓ Saved {path}")

print("\n✓ Done.")