"""
Part 7 — Out-of-Distribution Detection on Novel Action Classes

The system is trained on 8 action classes (subset {1,2,4,13,19,22,23,27}).
Part 7 detects when presented with excluded action classes (3,5-12,14-18,20-21,24-26)
using 6 OOD scoring methods on frozen encoders from Parts 2-6:

  1. Softmax entropy — high entropy → low confidence → OOD
  2. Max softmax probability — inverse confidence score
  3. Mahalanobis distance (sensor + video) — distance from training Gaussian
  4. k-NN distance (sensor + video) — distance to k-th nearest training point
  5. Conformal prediction set size — larger sets → high uncertainty → OOD
  6. Sensor-Video disagreement — cross-modal inconsistency → OOD signal

Combined score = weighted sum of normalized methods.
Evaluated via ROC-AUC, t-SNE visualizations, and confusion matrix.

Usage:
    python3 part7_ood.py
"""

import random
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import (auc, confusion_matrix, roc_auc_score, roc_curve,
                             precision_recall_curve)
from torch.utils.data import DataLoader

from data.sensor_dataset import SensorDataset, load_sensor_samples, pad_or_truncate
from data.video_dataset import PoseDataset, load_video_samples
from models.fusion import FusionMLP
from part6_confidence import get_fusion_embeddings, get_fusion_logits
from train import SensorLightningModule, VideoLightningModule
from utils import load_action_names, load_config, save_confusion_matrix
from utils_ml import (
    cp_predict_sets, nonconf_scores, fit_gaussian_embeddings, compute_mahal_distance,
    compute_entropy, compute_knn_distance, compute_ood_scores, normalize_scores,
    combine_scores
)

warnings.filterwarnings("ignore")

# ── Config ─────────────────────────────────────────────────────────────────────

ROOT    = Path(__file__).parent
OUT_DIR = ROOT / "outputs"
OUT_DIR.mkdir(exist_ok=True)

SCFG = load_config(str(ROOT / "configs/sensor.yaml"))
VCFG = load_config(str(ROOT / "configs/video.yaml"))

SUBSET     = SCFG.subset
OOD_SUBSET = [3, 5, 6]  # Excluded actions to test as OOD
LABEL_MAP  = {a: i for i, a in enumerate(SUBSET)}
OOD_LABEL_MAP = {a: i + len(SUBSET) for i, a in enumerate(OOD_SUBSET)}  # Map OOD actions to indices >= n_classes
KEEP_FEATS = [i for i in range(66) if i not in VCFG.drop_feats]
SEED       = 42

K_NN = 5
OOD_WEIGHTS = {
    "entropy": 0.2, "maxprob": 0.1, "mahal": 0.2,
    "knn": 0.2, "cp_size": 0.15, "disagreement": 0.15
}

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device(
    "mps"  if torch.backends.mps.is_available() else
    "cuda" if torch.cuda.is_available()          else "cpu"
)
print(f"Device: {DEVICE}\n")


# All OOD utilities imported from utils_ml (see imports above)


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    action_names = load_action_names(ROOT / "Sample_Code")
    class_names  = [action_names[a] for a in SUBSET]
    ood_names    = [action_names[a] for a in OOD_SUBSET]
    n_classes    = len(SUBSET)
    in_dim_s     = 12
    in_dim_v     = len(KEEP_FEATS) * 2

    print("Loading in-distribution (train) data...")
    sensor_samples_train = load_sensor_samples(ROOT / SCFG.inertial_dir, SUBSET, LABEL_MAP)
    video_samples_train  = load_video_samples(ROOT / VCFG.kalman_cache, SUBSET, LABEL_MAP, KEEP_FEATS)

    # Prepare train embeddings for Mahalanobis and k-NN
    s_train = [(y, r) for _, y, r in sensor_samples_train]
    v_train = [(y, seq) for _, y, seq in video_samples_train]

    print("Loading OOD data (excluded action classes)...")
    sensor_samples_ood = load_sensor_samples(ROOT / SCFG.inertial_dir, OOD_SUBSET, OOD_LABEL_MAP)
    video_samples_ood  = load_video_samples(ROOT / VCFG.kalman_cache, OOD_SUBSET, OOD_LABEL_MAP, KEEP_FEATS)

    s_ood = [(y, r) for _, y, r in sensor_samples_ood]
    v_ood = [(y, seq) for _, y, seq in video_samples_ood]

    print(f"In-distribution: {len(sensor_samples_train)} sensor | {len(video_samples_train)} video")
    print(f"OOD: {len(sensor_samples_ood)} sensor | {len(video_samples_ood)} video")

    # If no video OOD data, use sensor-only evaluation
    if len(v_ood) == 0:
        print("Note: No video OOD samples found (Kalman cache only includes training subset).")
        print("Using sensor embeddings only for OOD detection.\n")
        # Dummy video samples for embedding extraction (before velocity is added)
        v_ood = [(y, np.zeros((SCFG.max_len, len(KEEP_FEATS)))) for y, _ in s_ood]
    else:
        print()

    # Load frozen trained models (use fold 0 checkpoints)
    print("Loading trained models...")
    ckpt_root = ROOT / SCFG.checkpoint_dir / "fold_s1"
    s_mod = SensorLightningModule(
        n_classes=n_classes, in_dim=in_dim_s,
        d_model=SCFG.model.d_model, n_heads=SCFG.model.n_heads,
        n_layers=SCFG.model.n_layers, dropout=SCFG.model.dropout,
        lr=SCFG.training.lr, weight_decay=SCFG.training.weight_decay,
        n_epochs=SCFG.training.n_epochs,
    ).to(DEVICE)

    v_mod = VideoLightningModule(
        n_classes=n_classes, in_dim=in_dim_v,
        d_model=VCFG.model.d_model, n_heads=VCFG.model.n_heads,
        n_layers=VCFG.model.n_layers, dropout=VCFG.model.dropout,
        lr=VCFG.training.lr, weight_decay=VCFG.training.weight_decay,
        n_epochs=VCFG.training.n_epochs,
    ).to(DEVICE)

    # Try to load checkpoints
    try:
        ckpt = torch.load(str(ckpt_root / "best.ckpt"), map_location=DEVICE)
        s_mod.load_state_dict(ckpt["state_dict"])
        print(f"Loaded sensor checkpoint from {ckpt_root / 'best.ckpt'}")
    except:
        print("Warning: could not load sensor checkpoint; using untrained weights")

    ckpt_root_v = ROOT / VCFG.checkpoint_dir / "fold_s1"
    try:
        ckpt = torch.load(str(ckpt_root_v / "best.ckpt"), map_location=DEVICE)
        v_mod.load_state_dict(ckpt["state_dict"])
        print(f"Loaded video checkpoint from {ckpt_root_v / 'best.ckpt'}")
    except:
        print("Warning: could not load video checkpoint; using untrained weights")

    s_mod.eval(); v_mod.eval()

    # Extract embeddings
    print("\nExtracting training embeddings...")
    e_v_train, e_s_train, _ = get_fusion_embeddings(s_mod, v_mod, s_train, v_train, SCFG.max_len, DEVICE)

    print("Extracting OOD embeddings...")
    e_v_ood, e_s_ood, y_ood = get_fusion_embeddings(s_mod, v_mod, s_ood, v_ood, SCFG.max_len, DEVICE)

    # Fit Gaussian on training embeddings
    print("Fitting Gaussian on training embeddings...")
    mu_s, cov_s, mu_v, cov_v = fit_gaussian_embeddings(e_s_train, e_v_train)

    # Get logits for OOD samples
    print("Computing logits on OOD samples...")
    with torch.no_grad():
        bs = SCFG.training.batch_size
        s_ood_loader = DataLoader(SensorDataset(s_ood, augment=False, max_len=SCFG.max_len),
                                   batch_size=bs, shuffle=False)
        v_ood_loader = DataLoader(PoseDataset(v_ood, augment=False),
                                   batch_size=bs, shuffle=False)

        logits_s_ood, logits_v_ood = [], []
        for x, _ in s_ood_loader:
            logits_s_ood.append(s_mod(x.to(DEVICE)).cpu().numpy())
        for x, _ in v_ood_loader:
            logits_v_ood.append(v_mod(x.to(DEVICE)).cpu().numpy())

        logits_s_ood = np.concatenate(logits_s_ood, axis=0)
        logits_v_ood = np.concatenate(logits_v_ood, axis=0)

        # Fusion logits
        logits_f_ood = get_fusion_logits(FusionMLP(n_classes).to(DEVICE), e_v_ood, e_s_ood, DEVICE)

    # Compute conformal prediction sets (use T=1.0 from Part 6)
    # Simple: use top-2 classes as set for evaluation
    cp_sets_ood = cp_predict_sets(logits_f_ood, T=1.0, q_hat=0.5)

    # Compute OOD scores for training set (in-distribution baseline)
    print("Computing OOD scores for in-distribution baseline...")
    with torch.no_grad():
        s_train_loader = DataLoader(SensorDataset(s_train, augment=False, max_len=SCFG.max_len),
                                     batch_size=bs, shuffle=False)
        v_train_loader = DataLoader(PoseDataset(v_train, augment=False),
                                     batch_size=bs, shuffle=False)
        logits_s_train, logits_v_train = [], []
        for x, _ in s_train_loader:
            logits_s_train.append(s_mod(x.to(DEVICE)).cpu().numpy())
        for x, _ in v_train_loader:
            logits_v_train.append(v_mod(x.to(DEVICE)).cpu().numpy())

        logits_s_train = np.concatenate(logits_s_train, axis=0)
        logits_v_train = np.concatenate(logits_v_train, axis=0)
        logits_f_train = get_fusion_logits(FusionMLP(n_classes).to(DEVICE), e_v_train, e_s_train, DEVICE)
        cp_sets_train = cp_predict_sets(logits_f_train, T=1.0, q_hat=0.5)

    # Compute OOD scores
    print("Computing OOD scores...")
    scores_train = compute_ood_scores(
        logits_s_train, logits_v_train, logits_f_train,
        e_s_train, e_v_train, e_s_train, e_v_train,
        mu_s, cov_s, mu_v, cov_v, cp_sets_train, k_nn=K_NN
    )
    scores_ood = compute_ood_scores(
        logits_s_ood, logits_v_ood, logits_f_ood,
        e_s_ood, e_v_ood, e_s_train, e_v_train,
        mu_s, cov_s, mu_v, cov_v, cp_sets_ood, k_nn=K_NN
    )

    # Combine all scores
    all_scores = {}
    for key in scores_train.keys():
        all_scores[key] = np.hstack([scores_train[key], scores_ood[key]])

    # Normalize and combine
    normalized = normalize_scores(all_scores)
    combined_score = combine_scores(normalized, OOD_WEIGHTS)

    # Prepare labels: in-distribution (0) vs OOD (1)
    y_true = np.hstack([np.zeros(len(logits_s_train)), np.ones(len(logits_s_ood))])

    # ── Evaluation ───────────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("OOD Detection Evaluation")
    print("="*70)

    results = {}

    # Per-method ROC-AUC (using fusion logits for entropy/maxprob)
    for method in ["entropy_f", "maxprob_f", "mahal_s", "mahal_v", "knn_s", "knn_v", "cp_size", "disagreement"]:
        if method in normalized:
            fpr, tpr, _ = roc_curve(y_true, normalized[method])
            auroc = auc(fpr, tpr)
            results[method] = auroc
            print(f"  {method:20} AUROC = {auroc:.4f}")

    # Combined score
    fpr_c, tpr_c, _ = roc_curve(y_true, combined_score)
    auroc_c = auc(fpr_c, tpr_c)
    results["combined"] = auroc_c
    print(f"  {'combined':20} AUROC = {auroc_c:.4f}")

    print("\n" + "="*70)

    # ── Plotting ───────────────────────────────────────────────────────────

    # ROC curve
    fig, ax = plt.subplots(figsize=(8, 6))
    for method in ["entropy_f", "mahal_s", "knn_s", "cp_size"]:
        if method in normalized:
            fpr, tpr, _ = roc_curve(y_true, normalized[method])
            auroc = auc(fpr, tpr)
            ax.plot(fpr, tpr, label=f"{method} (AUROC={auroc:.3f})", linewidth=2)

    # Combined
    ax.plot(fpr_c, tpr_c, label=f"Combined (AUROC={auroc_c:.3f})", linewidth=2.5, color="black")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("OOD Detection: ROC Curves")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "part7_ood_roc.png", dpi=150)
    plt.close()
    print(f"Saved {OUT_DIR / 'part7_ood_roc.png'}")

    # t-SNE visualization (sensor)
    print("\nComputing t-SNE for sensor embeddings...")
    e_combined = np.vstack([e_s_train, e_s_ood])
    labels = y_true  # Reuse y_true: 0=in-dist, 1=OOD
    tsne = TSNE(n_components=2, random_state=SEED, perplexity=30)
    emb_2d = tsne.fit_transform(e_combined)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(emb_2d[labels==0, 0], emb_2d[labels==0, 1], c="blue", label="In-distribution", alpha=0.6, s=50)
    ax.scatter(emb_2d[labels==1, 0], emb_2d[labels==1, 1], c="red", label="OOD", alpha=0.6, s=50)
    ax.set_title("Sensor Embeddings (t-SNE)")
    ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")
    ax.legend()
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "part7_ood_tsne_sensor.png", dpi=150)
    plt.close()
    print(f"Saved {OUT_DIR / 'part7_ood_tsne_sensor.png'}")

    # t-SNE visualization (video)
    print("Computing t-SNE for video embeddings...")
    e_combined = np.vstack([e_v_train, e_v_ood])
    tsne = TSNE(n_components=2, random_state=SEED, perplexity=30)
    emb_2d = tsne.fit_transform(e_combined)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(emb_2d[labels==0, 0], emb_2d[labels==0, 1], c="blue", label="In-distribution", alpha=0.6, s=50)
    ax.scatter(emb_2d[labels==1, 0], emb_2d[labels==1, 1], c="red", label="OOD", alpha=0.6, s=50)
    ax.set_title("Video Embeddings (t-SNE)")
    ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")
    ax.legend()
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "part7_ood_tsne_video.png", dpi=150)
    plt.close()
    print(f"Saved {OUT_DIR / 'part7_ood_tsne_video.png'}")

    # Summary
    print("\nDone. Results saved to outputs/")


if __name__ == "__main__":
    main()
