"""
Part 3 — Bayesian Sensor-Video Fusion

The IMU signal reliably identifies sensor placement (wrist vs thigh),
which perfectly partitions the 8-class action space:

    wrist → {swipe_left, swipe_right, clap, boxing, knock}   (actions 1,2,4,13,19)
    thigh → {jog, walk, squat}                               (actions 22,23,27)

Fusion rule (Bayes):
    P(class | video, sensor) ∝ P(class | video) × P(placement | sensor)
                              = video softmax   × structured placement prior

Two variants are evaluated:
    Soft  — multiply video probs by placement probability (smooth, principled)
    Hard  — zero out impossible classes based on argmax placement (crisp gate)

Run: python part3_fusion.py
"""

import re
import random
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import torch
import torch.nn.functional as F
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score)
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from data.video_dataset import PoseDataset, load_video_samples
from data.sensor_dataset import SensorDataset, load_sensor_samples
from train import SensorLightningModule, VideoLightningModule
from utils import load_action_names, save_confusion_matrix
from data.sensor_dataset import extract_imu_features

warnings.filterwarnings("ignore")

# ── Constants ──────────────────────────────────────────────────────────────────
SEED   = 42
SUBSET = [1, 2, 4, 13, 19, 22, 23, 27]

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

# Sensor placement: wrist (0) or thigh (1) per action
PLACEMENT_MAP = {1: 0, 2: 0, 4: 0, 13: 0, 19: 0, 22: 1, 23: 1, 27: 1}

# Label indices (0..7) for each placement
WRIST_IDX = [i for i, a in enumerate(SUBSET) if PLACEMENT_MAP[a] == 0]  # [0,1,2,3,4]
THIGH_IDX = [i for i, a in enumerate(SUBSET) if PLACEMENT_MAP[a] == 1]  # [5,6,7]

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT         = Path(__file__).parent
SAMPLE_DIR   = ROOT / "Sample_Code"
INERTIAL_DIR = ROOT / "Inertial"
KALMAN_CACHE = ROOT / "outputs" / "pose_cache_kalman22"
VIDEO_CKPTS  = ROOT / "checkpoints" / "video"
SENSOR_CKPTS = ROOT / "checkpoints" / "sensor"
OUT_DIR      = ROOT / "outputs"
OUT_DIR.mkdir(exist_ok=True)

DROP_FEATS = [0, 24, 25, 26, 27, 29, 30, 31, 32, 34, 35, 62]
KEEP_FEATS = [i for i in range(66) if i not in DROP_FEATS]

DEVICE = torch.device(
    "mps"  if torch.backends.mps.is_available()  else
    "cuda" if torch.cuda.is_available()           else "cpu"
)

# ── Fusion functions ───────────────────────────────────────────────────────────
def soft_bayesian_fusion(
    video_probs: np.ndarray,        # (N, 8)
    placement_probs: np.ndarray,    # (N, 2)  col 0 = p_wrist, col 1 = p_thigh
) -> np.ndarray:
    """Multiply video posterior by placement prior, then renormalise."""
    prior = np.zeros_like(video_probs)
    prior[:, WRIST_IDX] = placement_probs[:, [0]]
    prior[:, THIGH_IDX] = placement_probs[:, [1]]
    fused = video_probs * prior
    fused /= fused.sum(axis=1, keepdims=True) + 1e-8
    return fused


def hard_gating(
    video_probs: np.ndarray,        # (N, 8)
    placement_probs: np.ndarray,    # (N, 2)
) -> np.ndarray:
    """Zero out impossible classes based on the argmax placement decision."""
    placement = placement_probs.argmax(axis=1)  # 0=wrist, 1=thigh
    mask = np.zeros_like(video_probs)
    for i, p in enumerate(placement):
        if p == 0:
            mask[i, WRIST_IDX] = 1.0
        else:
            mask[i, THIGH_IDX] = 1.0
    gated = video_probs * mask
    gated /= gated.sum(axis=1, keepdims=True) + 1e-8
    return gated


# ── Inference helpers ──────────────────────────────────────────────────────────
@torch.no_grad()
def get_video_probs(
    ckpt_path: Path,
    test_samples: list,
    batch_size: int = 16,
) -> tuple:
    """Load video checkpoint, return (y_true, softmax_probs) for test samples."""
    module = VideoLightningModule.load_from_checkpoint(str(ckpt_path))
    module.to(DEVICE).eval()
    loader = DataLoader(PoseDataset(test_samples, augment=False),
                        batch_size=batch_size, shuffle=False, num_workers=0)
    y_true, probs = [], []
    for x, y in loader:
        logits = module(x.to(DEVICE))
        probs.append(F.softmax(logits, dim=1).cpu().numpy())
        y_true.extend(y.numpy())
    return np.array(y_true), np.concatenate(probs, axis=0)


@torch.no_grad()
def get_sensor_probs(
    ckpt_path: Path,
    test_samples: list,
    max_len: int = 256,
    batch_size: int = 16,
) -> tuple:
    """Load sensor checkpoint, return (y_true, softmax_probs) for test samples."""
    module = SensorLightningModule.load_from_checkpoint(str(ckpt_path))
    module.to(DEVICE).eval()
    loader = DataLoader(SensorDataset(test_samples, max_len=max_len, augment=False),
                        batch_size=batch_size, shuffle=False, num_workers=0)
    y_true, probs = [], []
    for x, y in loader:
        logits = module(x.to(DEVICE))
        probs.append(F.softmax(logits, dim=1).cpu().numpy())
        y_true.extend(y.numpy())
    return np.array(y_true), np.concatenate(probs, axis=0)


# ── Main ───────────────────────────────────────────────────────────────────────
def main() -> None:
    label_map    = {a: i for i, a in enumerate(SUBSET)}
    action_names = load_action_names(SAMPLE_DIR)
    class_names  = [action_names[a] for a in SUBSET]
    fname_re     = re.compile(r"a(\d+)_s(\d+)_t(\d+)_")
    subset_set   = set(SUBSET)

    # ── Load IMU features ──────────────────────────────────────────────────────
    print("Loading IMU features...")
    imu_samples = []   # (subject, label_idx, placement, feature_vec, raw_iner)
    for path in sorted(INERTIAL_DIR.glob("*_inertial.mat")):
        m = fname_re.match(path.name)
        if not m:
            continue
        a, s, _ = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if a not in subset_set:
            continue
        raw   = scipy.io.loadmat(str(path))["d_iner"]
        feats = extract_imu_features(raw)
        imu_samples.append((s, label_map[a], PLACEMENT_MAP[a], feats, raw))

    # ── Load video sequences ───────────────────────────────────────────────────
    print("Loading video sequences...")
    video_samples = load_video_samples(KALMAN_CACHE, SUBSET, label_map, KEEP_FEATS)

    subjects = sorted(set(s for s, _, _, _, _ in imu_samples))
    print(f"  {len(imu_samples)} IMU samples | {len(video_samples)} video sequences | {len(subjects)} subjects\n")

    # ── LOSO evaluation ────────────────────────────────────────────────────────
    results = {
        "video":        {"y_true": [], "y_pred": []},
        "sensor_rf":    {"y_true": [], "y_pred": []},   # classical RF — best sensor model
        "sensor_deep":  {"y_true": [], "y_pred": []},   # SensorTransformer for reference
        "soft_fusion":  {"y_true": [], "y_pred": []},
        "hard_gating":  {"y_true": [], "y_pred": []},
    }
    placement_results = {"y_true": [], "y_pred": []}

    for test_subject in subjects:
        print(f"── Fold s{test_subject} ──────────────────────────────────────────")

        # Split IMU
        train_imu = [(y, pl, f) for s, y, pl, f, _ in imu_samples if s != test_subject]
        test_imu  = [(y, pl, f, r) for s, y, pl, f, r in imu_samples if s == test_subject]

        X_tr = np.array([f for _, _, f in train_imu])
        y_placement_tr = np.array([pl for _, pl, _ in train_imu])
        X_te = np.array([f for _, _, f, _ in test_imu])
        y_placement_te = np.array([pl for _, pl, _, _ in test_imu])

        sc = StandardScaler().fit(X_tr)
        clf = LogisticRegression(C=10, max_iter=1000, random_state=SEED)
        clf.fit(sc.transform(X_tr), y_placement_tr)
        placement_probs = clf.predict_proba(sc.transform(X_te))  # (N, 2)
        placement_preds = clf.predict(sc.transform(X_te))
        pl_acc = accuracy_score(y_placement_te, placement_preds)
        placement_results["y_true"].extend(y_placement_te)
        placement_results["y_pred"].extend(placement_preds)
        print(f"  Placement classifier accuracy: {pl_acc:.3f}")

        # ── RF sensor (classical best) ─────────────────────────────────────────
        X_tr_8  = np.array([f for s, _, _, f, _ in imu_samples if s != test_subject])
        y_tr_8  = np.array([y for s, y, _, _, _ in imu_samples if s != test_subject])
        X_te_8  = np.array([f for s, _, _, f, _ in imu_samples if s == test_subject])
        y_te_8  = np.array([y for s, y, _, _, _ in imu_samples if s == test_subject])

        sc8 = StandardScaler().fit(X_tr_8)
        rf  = RandomForestClassifier(n_estimators=200, random_state=SEED)
        rf.fit(sc8.transform(X_tr_8), y_tr_8)
        rf_preds = rf.predict(sc8.transform(X_te_8))
        results["sensor_rf"]["y_true"].extend(y_te_8)
        results["sensor_rf"]["y_pred"].extend(rf_preds)

        # ── Video + deep sensor probabilities from checkpoints ─────────────────
        video_ckpt  = VIDEO_CKPTS / f"fold_s{test_subject}" / "best.ckpt"
        sensor_ckpt = SENSOR_CKPTS / f"fold_s{test_subject}" / "best.ckpt"

        test_video_samples  = [(y, seq) for s, y, seq in video_samples if s == test_subject]
        test_sensor_samples = [(y, r)   for s, y, _, _, r in imu_samples if s == test_subject]

        y_true_v, video_probs  = get_video_probs(video_ckpt, test_video_samples)
        y_true_s, sensor_probs = get_sensor_probs(sensor_ckpt, test_sensor_samples)

        # ── Bayesian fusion ────────────────────────────────────────────────────
        soft_probs = soft_bayesian_fusion(video_probs, placement_probs)
        hard_probs = hard_gating(video_probs, placement_probs)

        for name, probs, y_true in [
            ("video",       video_probs,  y_true_v),
            ("sensor_deep", sensor_probs, y_true_s),
            ("soft_fusion", soft_probs,   y_true_v),
            ("hard_gating", hard_probs,   y_true_v),
        ]:
            preds = probs.argmax(axis=1)
            results[name]["y_true"].extend(y_true)
            results[name]["y_pred"].extend(preds)

        v_acc  = accuracy_score(y_true_v, video_probs.argmax(1))
        rf_acc = accuracy_score(y_te_8, rf_preds)
        sf_acc = accuracy_score(y_true_v, soft_probs.argmax(1))
        hg_acc = accuracy_score(y_true_v, hard_probs.argmax(1))
        print(f"  Video: {v_acc:.3f}  |  RF sensor: {rf_acc:.3f}  |  Soft fusion: {sf_acc:.3f}  |  Hard gating: {hg_acc:.3f}")

    # ── Placement classifier summary ───────────────────────────────────────────
    pl_acc_overall = accuracy_score(placement_results["y_true"], placement_results["y_pred"])
    print(f"\nPlacement classifier (wrist vs thigh)  —  Accuracy: {pl_acc_overall:.3f}")

    # ── Results summary ────────────────────────────────────────────────────────
    LABELS = {
        "video":        "Video alone (PoseTransformer)",
        "sensor_rf":    "Sensor alone (RF, classical)",
        "sensor_deep":  "Sensor alone (Transformer)",
        "soft_fusion":  "Soft Bayesian fusion",
        "hard_gating":  "Hard gating",
    }

    print(f"\n{'Method':<25} {'Accuracy':>10} {'Macro F1':>10}")
    print("─" * 47)
    for key, label in LABELS.items():
        acc = accuracy_score(results[key]["y_true"], results[key]["y_pred"])
        f1  = f1_score(results[key]["y_true"], results[key]["y_pred"], average="macro")
        print(f"{label:<25} {acc:>10.3f} {f1:>10.3f}")

    print()
    best_key = max(results, key=lambda k: accuracy_score(results[k]["y_true"], results[k]["y_pred"]))
    best_acc = accuracy_score(results[best_key]["y_true"], results[best_key]["y_pred"])
    best_f1  = f1_score(results[best_key]["y_true"], results[best_key]["y_pred"], average="macro")
    print(f"Best: {LABELS[best_key]}  —  Accuracy: {best_acc:.3f}  |  Macro F1: {best_f1:.3f}")
    print()
    print(classification_report(
        results[best_key]["y_true"], results[best_key]["y_pred"],
        target_names=class_names, digits=3,
    ))

    # ── Confusion matrices (2×3 grid, last cell empty) ────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(27, 16))
    cmaps = {
        "video":        "Greens",
        "sensor_rf":    "Blues",
        "sensor_deep":  "Blues",
        "soft_fusion":  "Oranges",
        "hard_gating":  "Purples",
    }

    axes.flat[-1].axis("off")   # 6th cell unused
    for ax, (key, label) in zip(axes.flat, LABELS.items()):
        yt = results[key]["y_true"]
        yp = results[key]["y_pred"]
        cm      = confusion_matrix(yt, yp)
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        acc = accuracy_score(yt, yp)
        f1  = f1_score(yt, yp, average="macro")
        n   = len(class_names)

        im = ax.imshow(cm_norm, cmap=cmaps[key], vmin=0, vmax=1)
        ax.set_xticks(range(n)); ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(n)); ax.set_yticklabels(class_names, fontsize=8)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        ax.set_title(f"{label}\nAcc={acc:.3f}  F1={f1:.3f}", fontweight="bold", fontsize=11)

        for i in range(n):
            for j in range(n):
                ax.text(j, i, f"{cm_norm[i,j]:.2f}", ha="center", va="center",
                        fontsize=7, color="white" if cm_norm[i, j] > 0.5 else "black")
        plt.colorbar(im, ax=ax, fraction=0.046)

    plt.suptitle(
        f"Part 3 — Bayesian Sensor-Video Fusion\n"
        f"Placement classifier accuracy: {pl_acc_overall:.3f}",
        fontsize=13, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    path = OUT_DIR / "part3_fusion_confusion.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")

    # ── Placement confusion matrix ─────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(5, 4))
    pl_cm      = confusion_matrix(placement_results["y_true"], placement_results["y_pred"])
    pl_cm_norm = pl_cm.astype(float) / pl_cm.sum(axis=1, keepdims=True)
    im = ax.imshow(pl_cm_norm, cmap="YlOrRd", vmin=0, vmax=1)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Wrist", "Thigh"], fontsize=11)
    ax.set_yticks([0, 1]); ax.set_yticklabels(["Wrist", "Thigh"], fontsize=11)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"Placement Classifier\nAccuracy={pl_acc_overall:.3f}", fontweight="bold")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{pl_cm_norm[i,j]:.3f}", ha="center", va="center",
                    fontsize=12, color="white" if pl_cm_norm[i, j] > 0.5 else "black")
    plt.colorbar(im, fraction=0.046)
    plt.tight_layout()
    path = OUT_DIR / "part3_placement_confusion.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
