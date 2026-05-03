"""
Part 5 (Fusion) — Class Imbalance on the Feature-MLP Fusion Model.

Uses frozen encoders from Part 2 checkpoints and trains only the FusionMLP head
under four conditions:
  balanced   — full paired data, uniform loss                  (reference)
  baseline   — imbalanced (20% minority), uniform loss         (shows the problem)
  weighted   — imbalanced, inverse-frequency CE weights        (Fix 1)
  augmented  — imbalanced + minority-class augmentation        (Fix 2)

Augmentation details:
  Video  — stronger Gaussian noise (σ=0.03) + smooth amplitude warp on skeleton keypoints.
           Physically: pose-estimation jitter and body-proportion variation.
           Time warp is NOT applied to video because the pose sequences already
           have variable length that is preserved by the dataset — temporal
           structure is captured via velocity features, not raw timing.
  Sensor — time warp + amplitude warp + stronger noise (σ=0.02) on IMU readings.
           Time warp is ideal for IMU: the same gesture performed faster/slower
           produces a genuinely different raw-acceleration signal that the model
           should learn to treat as equivalent.

Usage:
    python part5_fusion.py
"""

import random
import re
import warnings
from pathlib import Path
from typing import List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch.utils.data import DataLoader, TensorDataset

from data.video_dataset import PoseDataset, add_velocity as add_velocity_v
from data.sensor_dataset import (SensorDataset, add_velocity as add_velocity_s,
                                  pad_or_truncate)
from data.augmentation import time_warp, amplitude_warp
from train import SensorLightningModule, VideoLightningModule
from utils import load_action_names, load_config, save_confusion_matrix
from models.fusion import FusionMLP

warnings.filterwarnings("ignore")
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# ── Config ─────────────────────────────────────────────────────────────────────
ROOT    = Path(__file__).parent
OUT_DIR = ROOT / "outputs"
OUT_DIR.mkdir(exist_ok=True)

VCFG = load_config(str(ROOT / "configs/video.yaml"))
SCFG = load_config(str(ROOT / "configs/sensor.yaml"))

SUBSET        = VCFG.subset
LABEL_MAP     = {a: i for i, a in enumerate(SUBSET)}
KEEP_FEATS    = [i for i in range(66) if i not in VCFG.drop_feats]
MINORITY_ACTS = {1, 2, 4}
MINORITY_IDX  = {LABEL_MAP[a] for a in MINORITY_ACTS}
KEEP_FRAC     = 0.20
N_AUG         = 4

N_EPOCHS  = 60      # FusionMLP training epochs (Part 3 used 300 — lower here for speed)
LR        = 1e-3
WD        = 5e-3
P_DROP    = 0.3     # modality-dropout probability during training

DEVICE = torch.device(
    "mps"  if torch.backends.mps.is_available() else
    "cuda" if torch.cuda.is_available()          else "cpu"
)


def augment_video_strong(seq: np.ndarray) -> np.ndarray:
    """
    Augment skeleton keypoint sequence for minority-class oversampling.
    Stronger Gaussian noise (σ=0.03) + smooth amplitude warp.
    No time warp: video temporal structure is preserved via velocity features,
    not raw frame timing.
    """
    x  = seq.copy()
    x += np.random.normal(0, 0.03, x.shape).astype(np.float32)
    x  = amplitude_warp(x, sigma=0.08)
    return x


def augment_sensor_strong(raw: np.ndarray) -> np.ndarray:
    """
    Augment raw IMU sequence.  Time warp is the key augmentation for IMU:
    the same gesture at different speeds produces different acceleration profiles.
    """
    x  = raw.copy().astype(np.float32)
    x  = time_warp(x)
    x  = amplitude_warp(x)
    x += np.random.normal(0, 0.02, x.shape).astype(np.float32)
    return x


# ── Preprocessing (replicates Dataset logic without augmentation) ──────────────

def preprocess_video(seq: np.ndarray) -> torch.Tensor:
    """(T, D) → float32 tensor (T, 2D) ready for the video encoder."""
    x   = add_velocity_v(seq.copy())
    mu  = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True) + 1e-8
    return torch.from_numpy((x - mu) / std).float()


def preprocess_sensor(raw: np.ndarray) -> torch.Tensor:
    """(T, 6) → float32 tensor (max_len, 12) ready for the sensor encoder."""
    x   = pad_or_truncate(raw, SCFG.max_len).astype(np.float32)
    x   = add_velocity_s(x)
    mu  = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True) + 1e-8
    return torch.from_numpy((x - mu) / std).float()


# ── Embedding extraction ───────────────────────────────────────────────────────

@torch.no_grad()
def embed_video(v_mod, seq: np.ndarray) -> np.ndarray:
    x  = preprocess_video(seq).unsqueeze(0).to(DEVICE)   # (1, T, 108)
    vm = v_mod.model
    return vm.encoder(vm.pos_enc(vm.proj(x))).mean(dim=1).squeeze(0).cpu().numpy()


@torch.no_grad()
def embed_sensor(s_mod, raw: np.ndarray) -> np.ndarray:
    x  = preprocess_sensor(raw).unsqueeze(0).to(DEVICE)  # (1, 256, 12)
    sm = s_mod.model
    return sm.encoder(sm.pos_enc(sm.proj(x))).mean(dim=1).squeeze(0).cpu().numpy()


def extract_train_embeddings(
    v_mod,
    s_mod,
    train_paired: list,
    variant: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract train embeddings.  For 'augmented', minority-class samples get
    N_AUG additional strongly-augmented copies (video + sensor augmented
    independently then embedded through the frozen encoders).
    """
    e_v_list, e_s_list, y_list = [], [], []
    for _, _, _, lbl, video_seq, sensor_raw, _ in train_paired:
        e_v_list.append(embed_video(v_mod, video_seq))
        e_s_list.append(embed_sensor(s_mod, sensor_raw))
        y_list.append(lbl)
        if variant == "augmented" and lbl in MINORITY_IDX:
            for _ in range(N_AUG):
                e_v_list.append(embed_video(v_mod,   augment_video_strong(video_seq)))
                e_s_list.append(embed_sensor(s_mod, augment_sensor_strong(sensor_raw)))
                y_list.append(lbl)
    return np.stack(e_v_list), np.stack(e_s_list), np.array(y_list)


@torch.no_grad()
def extract_test_embeddings(
    v_mod, s_mod, test_paired: list
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    e_v_list, e_s_list, y_list = [], [], []
    for _, _, _, lbl, video_seq, sensor_raw, _ in test_paired:
        e_v_list.append(embed_video(v_mod, video_seq))
        e_s_list.append(embed_sensor(s_mod, sensor_raw))
        y_list.append(lbl)
    return np.stack(e_v_list), np.stack(e_s_list), np.array(y_list)


def train_fusion_mlp(
    e_v: np.ndarray,
    e_s: np.ndarray,
    y: np.ndarray,
    n_classes: int,
    class_weights: torch.Tensor = None,
) -> FusionMLP:
    ev = torch.tensor(e_v, dtype=torch.float32).to(DEVICE)
    es = torch.tensor(e_s, dtype=torch.float32).to(DEVICE)
    yt = torch.tensor(y,   dtype=torch.long).to(DEVICE)

    model  = FusionMLP(n_classes).to(DEVICE)
    opt    = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    loader = DataLoader(TensorDataset(ev, es, yt), batch_size=32, shuffle=True)
    w      = class_weights.to(DEVICE) if class_weights is not None else None

    for _ in range(N_EPOCHS):
        model.train()
        for bv, bs, by in loader:
            B  = bv.size(0)
            dv = (torch.rand(B) < P_DROP).to(DEVICE)
            ds = (torch.rand(B) < P_DROP).to(DEVICE)
            ds[dv & ds] = False
            loss = F.cross_entropy(model(bv, bs, dv, ds), by, weight=w)
            opt.zero_grad(); loss.backward(); opt.step()

    return model.eval()


@torch.no_grad()
def predict_fusion(model: FusionMLP, e_v: np.ndarray, e_s: np.ndarray) -> np.ndarray:
    ev = torch.tensor(e_v, dtype=torch.float32).to(DEVICE)
    es = torch.tensor(e_s, dtype=torch.float32).to(DEVICE)
    return model(ev, es).argmax(1).cpu().numpy()


# ── Data helpers ───────────────────────────────────────────────────────────────

def load_paired_samples() -> list:
    fname_re   = re.compile(r"a(\d+)_s(\d+)_t(\d+)_")
    subset_set = set(SUBSET)

    video_idx = {}
    for p in sorted((ROOT / VCFG.kalman_cache).glob("*.npy")):
        m = fname_re.match(p.stem)
        if not m: continue
        a, s, t = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if a in subset_set:
            video_idx[(a, s, t)] = np.load(str(p))[:, KEEP_FEATS]

    sensor_idx = {}
    for p in sorted((ROOT / SCFG.inertial_dir).glob("*_inertial.mat")):
        m = fname_re.match(p.name)
        if not m: continue
        a, s, t = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if a in subset_set:
            sensor_idx[(a, s, t)] = scipy.io.loadmat(str(p))["d_iner"]

    common = sorted(set(video_idx) & set(sensor_idx))
    return [
        (a, s, t, LABEL_MAP[a], video_idx[(a,s,t)], sensor_idx[(a,s,t)], None)
        for a, s, t in common
    ]


def introduce_imbalance(paired: list, minority_acts: set, keep_frac: float) -> list:
    rng = random.Random(42)
    return [
        row for row in paired
        if SUBSET[row[3]] not in minority_acts or rng.random() < keep_frac
    ]


def compute_class_weights(y_train: np.ndarray, n_classes: int) -> torch.Tensor:
    counts = np.bincount(y_train, minlength=n_classes).astype(float)
    counts = np.where(counts == 0, 1.0, counts)
    w = len(y_train) / (n_classes * counts)
    return torch.tensor(w, dtype=torch.float32)


# ── Plotting ───────────────────────────────────────────────────────────────────

def plot_summary(results: dict, class_names: List[str], path: Path) -> None:
    variants = ["balanced", "baseline", "weighted", "augmented"]
    labels   = ["Balanced\n(reference)", "Baseline\n(no fix)", "Weighted\nCE loss",
                 "Targeted\naugmentation"]
    colors   = ["#2ca02c", "#d9534f", "#5bc0de", "#f0ad4e"]

    macro_f1    = [f1_score(*results[v], average="macro") for v in variants]
    per_cls_f1  = {v: f1_score(*results[v], average=None) for v in variants}
    minority_f1 = {v: per_cls_f1[v][list(MINORITY_IDX)].mean() for v in variants}

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, vals, title, ylabel in [
        (axes[0], macro_f1,                          "LOSO Macro F1",          "Macro F1"),
        (axes[1], [minority_f1[v] for v in variants], "Minority-class F1\n(actions 1,2,4)", "F1"),
    ]:
        bars = ax.bar(labels, vals, color=colors, width=0.5)
        ax.axhline(vals[0], color="#2ca02c", linestyle="--", linewidth=1.2, alpha=0.5)
        ax.set_ylim(0, 1.08)
        ax.set_title(title, fontweight="bold")
        ax.set_ylabel(ylabel)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01,
                    f"{v:.3f}", ha="center", va="bottom", fontweight="bold", fontsize=10)

    n_cls = len(class_names)
    x, w = np.arange(n_cls), 0.18
    for i, (v, col, lbl) in enumerate(zip(variants, colors, labels)):
        axes[2].bar(x + (i - 1.5) * w, per_cls_f1[v], w,
                    label=lbl.replace("\n", " "), color=col, alpha=0.85)
    for mi in MINORITY_IDX:
        axes[2].axvspan(mi - 0.5, mi + 0.5, color="#ffcccc", alpha=0.25, zorder=0)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(class_names, rotation=45, ha="right", fontsize=8)
    axes[2].set_ylim(0, 1.08)
    axes[2].set_title("Per-class F1 by variant\n(pink = minority classes)", fontweight="bold")
    axes[2].set_ylabel("F1")
    axes[2].legend(fontsize=7)

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    action_names = load_action_names(ROOT / "Sample_Code")
    class_names  = [action_names[a] for a in SUBSET]
    n_classes    = len(SUBSET)

    print("Loading paired samples...")
    all_paired = load_paired_samples()
    subjects   = sorted(set(row[1] for row in all_paired))
    print(f"  {len(all_paired)} paired sequences | {len(subjects)} subjects")

    from collections import Counter
    nat = Counter(row[3] for row in all_paired)
    print("\nNatural class distribution:")
    for i, name in enumerate(class_names):
        tag = "  ← minority" if i in MINORITY_IDX else ""
        print(f"  [{i}] {name:<28} {nat[i]:>3}{tag}")

    imb_paired = introduce_imbalance(all_paired, MINORITY_ACTS, KEEP_FRAC)
    imb = Counter(row[3] for row in imb_paired)
    ratio = max(imb.values()) / max(1, min(imb.values()))
    print(f"\nAfter imbalance: {len(imb_paired)}/{len(all_paired)} samples  |  ratio {ratio:.1f}:1")

    ckpt_v_root = ROOT / VCFG.checkpoint_dir
    ckpt_s_root = ROOT / SCFG.checkpoint_dir

    variant_data = {
        "balanced":  all_paired,
        "baseline":  imb_paired,
        "weighted":  imb_paired,
        "augmented": imb_paired,
    }
    results = {}

    print(f"\nDevice: {DEVICE}  |  FusionMLP epochs/fold: {N_EPOCHS}")

    for variant, data in variant_data.items():
        print(f"\n{'─'*60}\nVariant: {variant}")
        all_true, all_pred = [], []

        for test_subj in subjects:
            train_p = [row for row in data if row[1] != test_subj]
            test_p  = [row for row in data if row[1] == test_subj]
            if not test_p:
                continue

            ckpt_v = ckpt_v_root / f"fold_s{test_subj}" / "best.ckpt"
            ckpt_s = ckpt_s_root / f"fold_s{test_subj}" / "best.ckpt"
            if not ckpt_v.exists() or not ckpt_s.exists():
                print(f"  [skip s{test_subj}] missing checkpoint")
                continue

            v_mod = VideoLightningModule.load_from_checkpoint(str(ckpt_v)).to(DEVICE).eval()
            s_mod = SensorLightningModule.load_from_checkpoint(str(ckpt_s)).to(DEVICE).eval()

            e_v_tr, e_s_tr, y_tr = extract_train_embeddings(v_mod, s_mod, train_p, variant)
            e_v_te, e_s_te, y_te = extract_test_embeddings(v_mod, s_mod, test_p)

            weights = compute_class_weights(y_tr, n_classes) if variant == "weighted" else None

            torch.manual_seed(42)
            model = train_fusion_mlp(e_v_tr, e_s_tr, y_tr, n_classes, class_weights=weights)

            preds = predict_fusion(model, e_v_te, e_s_te)
            all_true.extend(y_te)
            all_pred.extend(preds)

        y_true = np.array(all_true)
        y_pred = np.array(all_pred)
        results[variant] = (y_true, y_pred)

        acc  = accuracy_score(y_true, y_pred)
        f1m  = f1_score(y_true, y_pred, average="macro")
        f1pc = f1_score(y_true, y_pred, average=None)
        minf1 = f1pc[list(MINORITY_IDX)].mean()
        print(f"  Accuracy: {acc:.3f}  |  Macro F1: {f1m:.3f}  |  Minority F1: {minf1:.3f}")
        print(classification_report(y_true, y_pred, target_names=class_names, digits=3))

        save_confusion_matrix(
            y_true, y_pred, class_names,
            title=f"Part5 Fusion [{variant}]  Acc={acc:.3f}  F1={f1m:.3f}",
            path=OUT_DIR / f"part5_fusion_{variant}_confusion.png",
        )

    # ── Summary ────────────────────────────────────────────────────────────────
    plot_summary(results, class_names, OUT_DIR / "part5_fusion_summary.png")
    print("\nSaved part5_fusion_summary.png")

    print(f"\n{'='*72}")
    print(f"{'Variant':<14} {'Accuracy':>9} {'Macro F1':>10} {'Minority F1':>13} {'vs baseline':>12}")
    print("─" * 72)
    base_f1 = f1_score(*results["baseline"], average="macro")
    for v in ["balanced", "baseline", "weighted", "augmented"]:
        y_true, y_pred = results[v]
        acc   = accuracy_score(y_true, y_pred)
        f1m   = f1_score(y_true, y_pred, average="macro")
        f1pc  = f1_score(y_true, y_pred, average=None)
        minf1 = f1pc[list(MINORITY_IDX)].mean()
        delta = f"+{f1m - base_f1:.3f}" if v != "baseline" else "—"
        print(f"  {v:<12} {acc:>9.3f} {f1m:>10.3f} {minf1:>13.3f} {delta:>12}")
    print("─" * 72)
    print("Minority F1 = mean F1 over the 3 rare action classes (label idx 0,1,2)")
    print("\nNote: encoders were trained on balanced data in Part 2 (60 epochs).")
    print("Only the FusionMLP head sees the imbalanced training distribution.")
    print(f"FusionMLP trained for {N_EPOCHS} epochs (Part 3 used 300).")


if __name__ == "__main__":
    main()
