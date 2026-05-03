"""
Part 3 (continued) — Feature-level and confidence-weighted fusion.

Why Bayesian fusion gave 0% gain (see part3_fusion.py):
  Video makes zero cross-group errors — all mistakes are within group
  (walking↔jogging, swipe_left↔swipe_right).  The Bayesian placement
  prior only resolves *cross-group* confusions, so it has nothing to fix.

Two complementary strategies that target within-group errors:

  1. Confidence-weighted fusion  (zero learnable parameters)
     α_i = conf_video_i / (conf_video_i + conf_sensor_i)
     Exploits complementary uncertainty: video is uncertain on locomotion,
     sensor is uncertain on arm direction.  No overfitting risk.

  2. Feature-MLP fusion  (frozen encoders, ~8 K params)
     [e_video(64) ∥ e_sensor(64)] → 2-layer MLP → 8 classes.
     Modality dropout (p = 0.3 per stream) + learned null embeddings
     let the same model run with one modality missing (Part 4).

Usage:
    python part3_fusion_feature.py
"""

import re
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score)
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from data.video_dataset import PoseDataset
from data.sensor_dataset import SensorDataset
from train import SensorLightningModule, VideoLightningModule
from utils import load_action_names, load_config, save_confusion_matrix
from data.sensor_dataset import extract_imu_features
from models.fusion import FusionMLP

warnings.filterwarnings("ignore")
torch.manual_seed(42)
np.random.seed(42)

# ── Config ──────────────────────────────────────────────────────────────────────

ROOT    = Path(__file__).parent
OUT_DIR = ROOT / "outputs"
OUT_DIR.mkdir(exist_ok=True)

VCFG = load_config(str(ROOT / "configs/video.yaml"))
SCFG = load_config(str(ROOT / "configs/sensor.yaml"))

SUBSET     = VCFG.subset                                             # [1,2,4,13,19,22,23,27]
LABEL_MAP  = {a: i for i, a in enumerate(SUBSET)}
KEEP_FEATS = [i for i in range(66) if i not in VCFG.drop_feats]
DEVICE = torch.device(
    "mps"  if torch.backends.mps.is_available() else
    "cuda" if torch.cuda.is_available()          else "cpu"
)


# ── Paired sample loading ───────────────────────────────────────────────────────

def load_paired_samples():
    """
    Index both modalities by (action, subject, trial) key and return only
    samples present in both.  Each element: (a, s, t, label, video_seq, sensor_raw, imu_feats).
    """
    fname_re   = re.compile(r"a(\d+)_s(\d+)_t(\d+)_")
    subset_set = set(SUBSET)

    video_idx = {}
    for p in sorted((ROOT / VCFG.kalman_cache).glob("*.npy")):
        m = fname_re.match(p.stem)
        if not m:
            continue
        a, s, t = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if a in subset_set:
            video_idx[(a, s, t)] = np.load(str(p))[:, KEEP_FEATS]

    sensor_idx = {}
    for p in sorted((ROOT / SCFG.inertial_dir).glob("*_inertial.mat")):
        m = fname_re.match(p.name)
        if not m:
            continue
        a, s, t = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if a in subset_set:
            sensor_idx[(a, s, t)] = scipy.io.loadmat(str(p))["d_iner"]

    common = sorted(set(video_idx) & set(sensor_idx))
    return [
        (a, s, t, LABEL_MAP[a],
         video_idx[(a, s, t)],
         sensor_idx[(a, s, t)],
         extract_imu_features(sensor_idx[(a, s, t)]))
        for a, s, t in common
    ]


# ── Embedding + probability extraction ──────────────────────────────────────────

@torch.no_grad()
def extract_emb_and_probs(
    video_ckpt: Path,
    sensor_ckpt: Path,
    video_samples: list,    # [(label, seq), ...]
    sensor_samples: list,   # [(label, raw), ...]
):
    """
    Load frozen checkpoints and extract, for each sample:
      - 64-dim embedding  (encoder output after mean-pool, before head)
      - softmax probability vector
    Returns: e_v (N,64), e_s (N,64), p_v (N,C), p_s (N,C), y (N,)
    """
    v_mod = VideoLightningModule.load_from_checkpoint(str(video_ckpt)).to(DEVICE).eval()
    s_mod = SensorLightningModule.load_from_checkpoint(str(sensor_ckpt)).to(DEVICE).eval()

    v_loader = DataLoader(
        PoseDataset(video_samples, augment=False),
        batch_size=32, shuffle=False, num_workers=0,
    )
    s_loader = DataLoader(
        SensorDataset(sensor_samples, max_len=SCFG.max_len, augment=False),
        batch_size=32, shuffle=False, num_workers=0,
    )

    e_v_l, p_v_l, y_l = [], [], []
    for x, y in v_loader:
        x  = x.to(DEVICE)
        vm = v_mod.model
        h  = vm.encoder(vm.pos_enc(vm.proj(x))).mean(dim=1)   # (B, 64)
        e_v_l.append(h.cpu())
        p_v_l.append(F.softmax(vm.head(h), dim=1).cpu())
        y_l.extend(y.numpy())

    e_s_l, p_s_l = [], []
    for x, _ in s_loader:
        x  = x.to(DEVICE)
        sm = s_mod.model
        h  = sm.encoder(sm.pos_enc(sm.proj(x))).mean(dim=1)
        e_s_l.append(h.cpu())
        p_s_l.append(F.softmax(sm.head(h), dim=1).cpu())

    return (
        torch.cat(e_v_l).numpy(), torch.cat(e_s_l).numpy(),
        torch.cat(p_v_l).numpy(), torch.cat(p_s_l).numpy(),
        np.array(y_l),
    )


# ── Confidence-weighted fusion ──────────────────────────────────────────────────

def confidence_weighted_fusion(p_v: np.ndarray, p_s: np.ndarray) -> np.ndarray:
    """
    Sample-wise weighted average: α_i = conf_video / (conf_video + conf_sensor).
    No parameters — exploits that each model is uncertain on different classes.
    """
    c_v = p_v.max(axis=1, keepdims=True)
    c_s = p_s.max(axis=1, keepdims=True)
    α   = c_v / (c_v + c_s + 1e-8)
    return α * p_v + (1 - α) * p_s


def train_fusion_mlp(
    e_v: np.ndarray,
    e_s: np.ndarray,
    y: np.ndarray,
    n_classes: int,
    n_epochs: int = 300,
    lr: float = 1e-3,
    weight_decay: float = 5e-3,
    p_drop: float = 0.3,
) -> FusionMLP:
    """Train FusionMLP on pre-extracted embeddings with modality dropout."""
    ev = torch.tensor(e_v, dtype=torch.float32).to(DEVICE)
    es = torch.tensor(e_s, dtype=torch.float32).to(DEVICE)
    yt = torch.tensor(y,   dtype=torch.long).to(DEVICE)

    model  = FusionMLP(n_classes).to(DEVICE)
    opt    = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loader = DataLoader(TensorDataset(ev, es, yt), batch_size=32, shuffle=True)

    for _ in range(n_epochs):
        model.train()
        for bv, bs, by in loader:
            B  = bv.size(0)
            dv = (torch.rand(B) < p_drop).to(DEVICE)
            ds = (torch.rand(B) < p_drop).to(DEVICE)
            ds[dv & ds] = False   # never drop both simultaneously
            loss = F.cross_entropy(model(bv, bs, dv, ds), by)
            opt.zero_grad(); loss.backward(); opt.step()

    return model.eval()


@torch.no_grad()
def predict_mlp(
    model: FusionMLP,
    e_v: np.ndarray,
    e_s: np.ndarray,
    miss_video: bool = False,
    miss_sensor: bool = False,
) -> np.ndarray:
    ev = torch.tensor(e_v, dtype=torch.float32).to(DEVICE)
    es = torch.tensor(e_s, dtype=torch.float32).to(DEVICE)
    B  = ev.size(0)
    dv = torch.ones(B, dtype=torch.bool, device=DEVICE) if miss_video  else None
    ds = torch.ones(B, dtype=torch.bool, device=DEVICE) if miss_sensor else None
    return model(ev, es, dv, ds).argmax(1).cpu().numpy()


# ── Main ────────────────────────────────────────────────────────────────────────

def main() -> None:
    action_names = load_action_names(ROOT / "Sample_Code")
    class_names  = [action_names[a] for a in SUBSET]
    n_classes    = len(SUBSET)

    print("Loading paired samples...")
    paired   = load_paired_samples()
    subjects = sorted(set(s for _, s, _, _, _, _, _ in paired))
    print(f"  {len(paired)} paired sequences | {len(subjects)} subjects\n")

    ckpt_v_root = ROOT / VCFG.checkpoint_dir
    ckpt_s_root = ROOT / SCFG.checkpoint_dir

    METHODS = [
        "video", "sensor_rf",
        "conf_weighted",
        "feat_mlp", "feat_mlp_video_only", "feat_mlp_sensor_only",
    ]
    res = {m: {"y_true": [], "y_pred": []} for m in METHODS}

    for test_subject in subjects:
        print(f"── Fold s{test_subject} " + "─" * 46)

        ckpt_v = ckpt_v_root / f"fold_s{test_subject}" / "best.ckpt"
        ckpt_s = ckpt_s_root / f"fold_s{test_subject}" / "best.ckpt"
        if not ckpt_v.exists() or not ckpt_s.exists():
            print("  [skip] missing checkpoint")
            continue

        train_p = [(a,s,t,lbl,vs,sr,fi) for (a,s,t,lbl,vs,sr,fi) in paired if s != test_subject]
        test_p  = [(a,s,t,lbl,vs,sr,fi) for (a,s,t,lbl,vs,sr,fi) in paired if s == test_subject]

        # ── RF sensor baseline ─────────────────────────────────────────────────
        X_tr = np.array([fi  for (*_, fi) in train_p])
        y_tr = np.array([lbl for (_,_,_,lbl,*_) in train_p])
        X_te = np.array([fi  for (*_, fi) in test_p])
        y_te = np.array([lbl for (_,_,_,lbl,*_) in test_p])
        sc   = StandardScaler().fit(X_tr)
        rf   = RandomForestClassifier(n_estimators=200, random_state=42)
        rf.fit(sc.transform(X_tr), y_tr)
        rf_preds = rf.predict(sc.transform(X_te))
        res["sensor_rf"]["y_true"].extend(y_te)
        res["sensor_rf"]["y_pred"].extend(rf_preds)

        # ── Frozen encoder embeddings + probabilities ──────────────────────────
        v_tr = [(lbl, vs) for (_,_,_,lbl,vs,sr,_) in train_p]
        v_te = [(lbl, vs) for (_,_,_,lbl,vs,sr,_) in test_p]
        s_tr = [(lbl, sr) for (_,_,_,lbl,vs,sr,_) in train_p]
        s_te = [(lbl, sr) for (_,_,_,lbl,vs,sr,_) in test_p]

        e_v_tr, e_s_tr, _, _, y_tr_emb = extract_emb_and_probs(ckpt_v, ckpt_s, v_tr, s_tr)
        e_v_te, e_s_te, p_v_te, p_s_te, y_te_emb = extract_emb_and_probs(ckpt_v, ckpt_s, v_te, s_te)

        # ── Video alone ────────────────────────────────────────────────────────
        res["video"]["y_true"].extend(y_te_emb)
        res["video"]["y_pred"].extend(p_v_te.argmax(1))

        # ── Confidence-weighted (no params) ────────────────────────────────────
        cw_preds = confidence_weighted_fusion(p_v_te, p_s_te).argmax(1)
        res["conf_weighted"]["y_true"].extend(y_te_emb)
        res["conf_weighted"]["y_pred"].extend(cw_preds)

        # ── Feature-MLP (train on this fold's train embeddings) ────────────────
        fusion_model = train_fusion_mlp(e_v_tr, e_s_tr, y_tr_emb, n_classes)

        mlp_both   = predict_mlp(fusion_model, e_v_te, e_s_te)
        mlp_v_only = predict_mlp(fusion_model, e_v_te, e_s_te, miss_sensor=True)
        mlp_s_only = predict_mlp(fusion_model, e_v_te, e_s_te, miss_video=True)

        for key, preds in [
            ("feat_mlp",             mlp_both),
            ("feat_mlp_video_only",  mlp_v_only),
            ("feat_mlp_sensor_only", mlp_s_only),
        ]:
            res[key]["y_true"].extend(y_te_emb)
            res[key]["y_pred"].extend(preds)

        print(
            f"  video: {accuracy_score(y_te_emb, p_v_te.argmax(1)):.3f}  |"
            f"  sensor_rf: {accuracy_score(y_te, rf_preds):.3f}  |"
            f"  conf_weighted: {accuracy_score(y_te_emb, cw_preds):.3f}  |"
            f"  feat_mlp: {accuracy_score(y_te_emb, mlp_both):.3f}"
        )

    # ── Summary table ──────────────────────────────────────────────────────────
    LABELS = {
        "video":               "Video alone (PoseTransformer)",
        "sensor_rf":           "Sensor alone (RF, classical)",
        "conf_weighted":       "Confidence-weighted fusion",
        "feat_mlp":            "Feature-MLP fusion (both modalities)",
        "feat_mlp_video_only": "  └─ video only  (sensor missing)",
        "feat_mlp_sensor_only":"  └─ sensor only (video missing)",
    }

    print(f"\n{'='*65}")
    print(f"{'Method':<38} {'Accuracy':>10} {'Macro F1':>10}")
    print("─" * 60)
    for key, label in LABELS.items():
        acc = accuracy_score(res[key]["y_true"], res[key]["y_pred"])
        f1  = f1_score(res[key]["y_true"], res[key]["y_pred"], average="macro")
        print(f"  {label:<36} {acc:>10.3f} {f1:>10.3f}")

    best_key = "feat_mlp"
    print()
    print(classification_report(
        res[best_key]["y_true"], res[best_key]["y_pred"],
        target_names=class_names, digits=3,
    ))

    # ── Confusion matrix grid (2×3) ────────────────────────────────────────────
    to_plot = [
        ("video",               "Video alone",                    "Greens"),
        ("sensor_rf",           "Sensor RF",                      "Blues"),
        ("conf_weighted",       "Confidence-weighted fusion",      "Oranges"),
        ("feat_mlp",            "Feature-MLP (both)",             "Purples"),
        ("feat_mlp_video_only", "Feature-MLP (video only)",       "RdPu"),
        ("feat_mlp_sensor_only","Feature-MLP (sensor only)",      "YlOrRd"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(27, 16))
    for ax, (key, title, cmap) in zip(axes.flat, to_plot):
        yt, yp = res[key]["y_true"], res[key]["y_pred"]
        cm     = confusion_matrix(yt, yp).astype(float)
        cm    /= cm.sum(axis=1, keepdims=True)
        acc    = accuracy_score(yt, yp)
        f1     = f1_score(yt, yp, average="macro")
        n      = len(class_names)
        im     = ax.imshow(cm, cmap=cmap, vmin=0, vmax=1)
        ax.set_xticks(range(n))
        ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(n))
        ax.set_yticklabels(class_names, fontsize=8)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"{title}\nAcc={acc:.3f}  F1={f1:.3f}", fontweight="bold", fontsize=11)
        for i in range(n):
            for j in range(n):
                ax.text(j, i, f"{cm[i,j]:.2f}", ha="center", va="center",
                        fontsize=7, color="white" if cm[i, j] > 0.5 else "black")
        plt.colorbar(im, ax=ax, fraction=0.046)

    plt.suptitle(
        "Part 3 — Feature-level Fusion  (frozen encoders + FusionMLP)\n"
        "Parts 4A/4B — Missing-modality inference via learned null embeddings",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    path = OUT_DIR / "part3_fusion_feature_confusion.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved {path}")


if __name__ == "__main__":
    main()
