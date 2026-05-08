"""Part 7: OOD Detection - Complete Two-Tier Implementation with Caching"""
import argparse, math, pickle, warnings
from pathlib import Path
import numpy as np
import torch, torch.nn.functional as F
from sklearn.covariance import LedoitWolf
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

def is_jupyter():
    """Auto-detect if running in Jupyter/IPython notebook."""
    try:
        from IPython import get_ipython
        return get_ipython() is not None
    except:
        return False

# Suppress numerical stability warnings (Ledoit-Wolf handles edge cases gracefully)
warnings.filterwarnings('ignore', category=RuntimeWarning)

from data.sensor_dataset import SensorDataset, load_sensor_samples
from data.video_dataset import PoseDataset, load_video_samples
from data.fusion_dataset import FusionDataset, extract_matched_keys
from models.model import TransformerClassifier
from models.fusion import FusionTransformerClassifier
from utils import load_config

ROOT = Path(__file__).parent
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description="OOD Detection: Two-Tier Mahalanobis + Conformal")
parser.add_argument("--modality", choices=["sensor", "video", "fusion"], default="sensor")
parser.add_argument("--config", type=Path, default=Path("configs/sensor.yaml"))
parser.add_argument("--ood-actions", nargs="+", type=int, default=[24, 9, 12])
parser.add_argument("--cache-dir", type=Path, default=Path("outputs/ood_stats"))
parser.add_argument("--force-recache", action="store_true", help="Ignore cache and recompute")
args = parser.parse_args()

cfg = load_config(args.config)
ID_SUBSET = cfg.subset
OOD_SUBSET = args.ood_actions
subjects = list(range(1, 9))
CHECKPOINT_DIR = ROOT / f"checkpoints/conformal/{args.modality}"
args.cache_dir.mkdir(parents=True, exist_ok=True)

print(f"Device: {DEVICE}\nModality: {args.modality}\nID: {ID_SUBSET}\nOOD: {OOD_SUBSET}\n")

if not CHECKPOINT_DIR.exists():
    print(f"❌ {CHECKPOINT_DIR} not found. Run: python3 train.py --modality {args.modality} --conformal")
    exit(1)

def compute_mahal(embeddings, centroids, covariances):
    """Minimum Mahalanobis distance to any class."""
    n, nc = len(embeddings), len(centroids)
    distances = np.zeros((n, nc))
    for c, (centroid, cov) in enumerate(zip(centroids.values(), covariances.values())):
        try:
            cov_inv = np.linalg.inv(cov)
        except:
            cov_inv = np.linalg.pinv(cov)
        for i in range(n):
            diff = embeddings[i] - centroid
            distances[i, c] = math.sqrt(diff @ cov_inv @ diff)
    return distances.min(axis=1)

def load_embs(subject, action_subset, dataset_cls, cfg, model, device, global_stats_fusion=None):
    """Load embeddings for subject's samples."""
    inertial_dir = getattr(cfg, "inertial_dir", "Inertial")
    kalman_cache = getattr(cfg, "kalman_cache", "outputs/pose_cache_kalman22")

    if args.modality == "sensor":
        samples = load_sensor_samples(ROOT / inertial_dir, action_subset,
                                     {a: i for i, a in enumerate(action_subset)})
    elif args.modality == "video":
        samples = load_video_samples(ROOT / kalman_cache, action_subset,
                                    {a: i for i, a in enumerate(action_subset)})
    else:
        s_samp = load_sensor_samples(ROOT / inertial_dir, action_subset,
                                     {a: i for i, a in enumerate(action_subset)})
        v_samp = load_video_samples(ROOT / kalman_cache, action_subset,
                                    {a: i for i, a in enumerate(action_subset)})
        samples = extract_matched_keys(s_samp, v_samp)

    samples = [(s, y, d) for s, y, d in samples if s == subject]
    if not samples:
        return np.array([]), np.array([])

    if args.modality == "fusion":
        # FusionDataset expects (label, paired_data), not (subject, label, paired_data)
        samples = [(y, d) for _, y, d in samples]
        kw = {"normalization_type": getattr(cfg, "normalization_type", "global"), "augment": False}
        if global_stats_fusion:
            kw["global_stats_sensor"], kw["global_stats_video"] = global_stats_fusion
    else:
        kw = {"max_len": getattr(cfg, "max_len", 256),
              "normalization_type": getattr(cfg, "normalization_type", "global"), "augment": False}
        if args.modality == "video":
            kw["landmark_set"] = getattr(cfg, "landmark_set", "hands_legs_hips")

    dataset = dataset_cls(samples, **kw)
    embs, labels = [], []
    model.eval()
    with torch.no_grad():
        for sample in dataset:
            if args.modality == "fusion":
                (x_s, x_v), label = sample
                x_s, x_v = x_s.unsqueeze(0).to(device), x_v.unsqueeze(0).to(device)
                _, emb = model(x_s, x_v, return_embedding=True)
            else:
                x, label = sample
                x = x.unsqueeze(0).to(device)
                _, emb = model(x, return_embedding=True)
            embs.append(emb.squeeze(0).cpu().numpy())
            labels.append(int(label) if isinstance(label, torch.Tensor) else int(label))

    return np.stack(embs), np.array(labels)

def load_probs(subject, action_subset, dataset_cls, cfg, model, device, global_stats_fusion=None):
    """Load softmax probabilities for subject."""
    inertial_dir = getattr(cfg, "inertial_dir", "Inertial")
    kalman_cache = getattr(cfg, "kalman_cache", "outputs/pose_cache_kalman22")

    if args.modality == "sensor":
        samples = load_sensor_samples(ROOT / inertial_dir, action_subset,
                                     {a: i for i, a in enumerate(action_subset)})
    elif args.modality == "video":
        samples = load_video_samples(ROOT / kalman_cache, action_subset,
                                    {a: i for i, a in enumerate(action_subset)})
    else:
        s_samp = load_sensor_samples(ROOT / inertial_dir, action_subset,
                                     {a: i for i, a in enumerate(action_subset)})
        v_samp = load_video_samples(ROOT / kalman_cache, action_subset,
                                    {a: i for i, a in enumerate(action_subset)})
        samples = extract_matched_keys(s_samp, v_samp)

    samples = [(s, y, d) for s, y, d in samples if s == subject]
    if not samples:
        return np.array([]), np.array([])

    if args.modality == "fusion":
        # FusionDataset expects (label, paired_data), not (subject, label, paired_data)
        samples = [(y, d) for _, y, d in samples]
        kw = {"normalization_type": getattr(cfg, "normalization_type", "global"), "augment": False}
        if global_stats_fusion:
            kw["global_stats_sensor"], kw["global_stats_video"] = global_stats_fusion
    else:
        kw = {"max_len": getattr(cfg, "max_len", 256),
              "normalization_type": getattr(cfg, "normalization_type", "global"), "augment": False}
        if args.modality == "video":
            kw["landmark_set"] = getattr(cfg, "landmark_set", "hands_legs_hips")

    dataset = dataset_cls(samples, **kw)
    probs, labels = [], []
    model.eval()
    with torch.no_grad():
        for sample in dataset:
            if args.modality == "fusion":
                (x_s, x_v), label = sample
                x_s, x_v = x_s.unsqueeze(0).to(device), x_v.unsqueeze(0).to(device)
                logits, _ = model(x_s, x_v, return_embedding=True)
            else:
                x, label = sample
                x = x.unsqueeze(0).to(device)
                logits, _ = model(x, return_embedding=True)
            p = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()
            probs.append(p)
            labels.append(int(label) if isinstance(label, torch.Tensor) else int(label))

    return np.stack(probs), np.array(labels)

def compute_global_stats_fusion(train_subjs, action_subset, cfg):
    """Compute global stats for sensor and video modalities for fusion."""
    from data.sensor_dataset import add_velocity as add_velocity_sensor
    from data.video_dataset import add_velocity as add_velocity_video, get_landmark_indices
    sensor_all, video_all = [], []

    # Get all config attributes with robust fallbacks
    landmark_set = getattr(cfg, "landmark_set", "hands_legs_hips")
    landmark_indices = get_landmark_indices(landmark_set)
    inertial_dir = getattr(cfg, "inertial_dir", "Inertial")
    kalman_cache = getattr(cfg, "kalman_cache", "outputs/pose_cache_kalman22")

    for train_s in train_subjs:
        s_samp = load_sensor_samples(ROOT / inertial_dir, action_subset,
                                     {a: i for i, a in enumerate(action_subset)})
        v_samp = load_video_samples(ROOT / kalman_cache, action_subset,
                                    {a: i for i, a in enumerate(action_subset)})
        s_samp = [(s, y, d) for s, y, d in s_samp if s == train_s]
        v_samp = [(s, y, d) for s, y, d in v_samp if s == train_s]

        for _, _, iner in s_samp:
            sensor_all.append(add_velocity_sensor(iner.astype(np.float32)))

        for _, _, pose in v_samp:
            pose = pose.astype(np.float32)
            # Filter to selected landmarks (matching FusionDataset)
            col_indices = []
            for i in landmark_indices:
                col_indices.extend([i * 2, i * 2 + 1])
            pose_filtered = pose[:, col_indices]
            # Add velocity on filtered data (matching FusionDataset)
            video_all.append(add_velocity_video(pose_filtered))

    if sensor_all:
        sensor_concat = np.vstack(sensor_all)
        sensor_stats = (sensor_concat.mean(axis=0), sensor_concat.std(axis=0) + 1e-5)
    else:
        sensor_stats = (np.zeros(12), np.ones(12))

    if video_all:
        video_concat = np.vstack(video_all)
        # video_concat has shape (N, len(landmarks)*4) at this point (no wrist deltas yet)
        video_stats = (video_concat.mean(axis=0), video_concat.std(axis=0) + 1e-5)
    else:
        # Default video size: hands_legs_hips = 12 landmarks * 4 = 48 dims
        video_stats = (np.zeros(48), np.ones(48))

    return (sensor_stats, video_stats)

# ============================================================================
# Main Evaluation Loop
# ============================================================================

print("="*70)
print("OOD DETECTION: MAHALANOBIS DISTANCE (8-Fold LOSO)")
print("="*70)

dataset_cls = {"sensor": SensorDataset, "video": PoseDataset, "fusion": FusionDataset}[args.modality]
results = {"far": [], "ood_tpr": [], "auroc": [], "test_d_all": [], "ood_d_all": [], "ood_actions_all": []}
fold_details = []  # Track per-fold metrics for table

for fold_idx, test_subj in enumerate(subjects, 1):
    # FIX #4: Use correct calibration subject (same as train.py)
    cal_subj = subjects[(fold_idx - 1) % 8]
    train_subjs = [s for s in subjects if s != test_subj and s != cal_subj]

    ckpt = CHECKPOINT_DIR / f"fold_s{test_subj}.pt"
    if not ckpt.exists():
        print(f"Fold {fold_idx}: ❌ missing")
        continue

    state = torch.load(str(ckpt), map_location=DEVICE, weights_only=True)

    # Create model (robust config access to handle different YAML structures)
    if args.modality == "sensor":
        in_dim = getattr(cfg, "in_dim_sensor", getattr(cfg, "in_dim", 12))
        model = TransformerClassifier(n_classes=len(ID_SUBSET), in_dim=in_dim,
                                     d_model=cfg.model.d_model, n_heads=cfg.model.n_heads,
                                     n_layers=cfg.model.n_layers, dropout=cfg.model.dropout)
    elif args.modality == "video":
        in_dim = getattr(cfg, "in_dim", 98)
        model = TransformerClassifier(n_classes=len(ID_SUBSET), in_dim=in_dim,
                                     d_model=cfg.model.d_model, n_heads=cfg.model.n_heads,
                                     n_layers=cfg.model.n_layers, dropout=cfg.model.dropout)
    else:
        in_dim_sensor = getattr(cfg, "in_dim_sensor", 12)
        in_dim_video = getattr(cfg, "in_dim_video", 98)
        model = FusionTransformerClassifier(n_classes=len(ID_SUBSET), in_dim_sensor=in_dim_sensor,
                                           in_dim_video=in_dim_video, d_model=cfg.model.d_model,
                                           n_heads=cfg.model.n_heads, n_layers=cfg.model.n_layers,
                                           dropout=cfg.model.dropout,
                                           d_fusion=getattr(cfg.model, "d_fusion", cfg.model.d_model))

    model = model.to(DEVICE)
    model.load_state_dict(state, strict=False)

    # Compute global stats for fusion if needed (outside cache check, needed for all phases)
    global_stats_fusion = None
    if args.modality == "fusion":
        global_stats_fusion = compute_global_stats_fusion(train_subjs, ID_SUBSET, cfg)

    # FIX #3: Cache logic - Phase A (ID Reference)
    cache_file = args.cache_dir / f"stats_{args.modality}_fold_s{test_subj}.pkl"

    if cache_file.exists() and not args.force_recache:
        print(f"Fold {fold_idx}: Loading from cache...", end="")
        with open(cache_file, 'rb') as f:
            cache = pickle.load(f)
        centroids, covs, dist_th = cache['centroids'], cache['covariances'], cache['dist_th']
        print(" ✓")
    else:
        print(f"Fold {fold_idx}: Computing ID reference...", end="")
        # Phase A: Compute centroids from training subjects
        train_embs_all, train_labels_all = [], []
        for train_s in train_subjs:
            e, l = load_embs(train_s, ID_SUBSET, dataset_cls, cfg, model, DEVICE, global_stats_fusion)
            if len(e) > 0:
                train_embs_all.append(e)
                train_labels_all.append(l)

        if not train_embs_all:
            print(" ❌ no train data")
            continue

        train_embs = np.vstack(train_embs_all)
        train_labels = np.concatenate(train_labels_all)

        centroids, covs = {}, {}
        for c in range(len(ID_SUBSET)):
            ce = train_embs[train_labels == c]
            if len(ce) > 0:
                centroids[c] = ce.mean(axis=0)
                lw = LedoitWolf()
                cov, _ = lw.fit(ce).covariance_, lw.shrinkage_
                covs[c] = cov

        # Phase B: Threshold calibration
        cal_e, cal_l = load_embs(cal_subj, ID_SUBSET, dataset_cls, cfg, model, DEVICE, global_stats_fusion)
        if len(cal_e) == 0:
            print(" ❌ no calibration data")
            continue

        cal_d = compute_mahal(cal_e, centroids, covs)
        dist_th = float(np.percentile(cal_d, 95.0))

        # Save cache
        with open(cache_file, 'wb') as f:
            pickle.dump({'centroids': centroids, 'covariances': covs, 'dist_th': dist_th}, f)
        print(" ✓")

    # Phase C: Test on ID samples
    test_e, test_l = load_embs(test_subj, ID_SUBSET, dataset_cls, cfg, model, DEVICE, global_stats_fusion)
    if len(test_e) == 0:
        print(f"Fold {fold_idx}: ❌ no test data")
        continue

    test_d = compute_mahal(test_e, centroids, covs)

    # Mahalanobis distance threshold
    test_alarm = test_d > dist_th
    far = test_alarm.mean()

    # OOD samples - FIX #2: Extract both embeddings AND softmax
    try:
        inertial_dir = getattr(cfg, "inertial_dir", "Inertial")
        kalman_cache = getattr(cfg, "kalman_cache", "outputs/pose_cache_kalman22")

        if args.modality == "sensor":
            ood_samp = load_sensor_samples(ROOT / inertial_dir, OOD_SUBSET,
                                          {a: i for i, a in enumerate(OOD_SUBSET)})
        elif args.modality == "video":
            ood_samp = load_video_samples(ROOT / kalman_cache, OOD_SUBSET,
                                         {a: i for i, a in enumerate(OOD_SUBSET)})
        else:
            s_s = load_sensor_samples(ROOT / inertial_dir, OOD_SUBSET,
                                     {a: i for i, a in enumerate(OOD_SUBSET)})
            v_s = load_video_samples(ROOT / kalman_cache, OOD_SUBSET,
                                    {a: i for i, a in enumerate(OOD_SUBSET)})
            ood_samp = extract_matched_keys(s_s, v_s)
    except Exception as e:
        print(f"  ❌ OOD sample loading failed: {e}")
        ood_samp = []

    if ood_samp and len(ood_samp) > 0:
        # Prepare OOD dataset
        if args.modality == "fusion":
            # FusionDataset expects (label, paired_data), not (subject, label, paired_data)
            ood_samp_fmt = [(y, d) for _, y, d in ood_samp]
            kw = {"normalization_type": getattr(cfg, "normalization_type", "global"), "augment": False}
            if global_stats_fusion:
                kw["global_stats_sensor"], kw["global_stats_video"] = global_stats_fusion
        else:
            ood_samp_fmt = ood_samp
            kw = {"max_len": getattr(cfg, "max_len", 256),
                  "normalization_type": getattr(cfg, "normalization_type", "global"), "augment": False}
            if args.modality == "video":
                kw["landmark_set"] = getattr(cfg, "landmark_set", "hands_legs_hips")

        # Store original action IDs (reverse-map from labels 0,1,2 back to 24,9,12)
        label_to_action = {i: a for i, a in enumerate(OOD_SUBSET)}
        ood_action_ids = np.array([label_to_action[label] for _, label, _ in ood_samp])

        ds = dataset_cls(ood_samp_fmt, **kw)
        ood_e_all = []
        model.eval()
        with torch.no_grad():
            for sample in ds:
                if args.modality == "fusion":
                    (x_s, x_v), _ = sample
                    x_s, x_v = x_s.unsqueeze(0).to(DEVICE), x_v.unsqueeze(0).to(DEVICE)
                    _, emb = model(x_s, x_v, return_embedding=True)
                else:
                    x, _ = sample
                    x = x.unsqueeze(0).to(DEVICE)
                    _, emb = model(x, return_embedding=True)
                ood_e_all.append(emb.squeeze(0).cpu().numpy())

        ood_e = np.stack(ood_e_all)

        ood_d = compute_mahal(ood_e, centroids, covs)

        # Mahalanobis distance alarm on OOD
        ood_alarm = ood_d > dist_th
        ood_tpr = ood_alarm.mean()

        # AUROC
        combined_d = np.concatenate([test_d, ood_d])
        combined_l = np.concatenate([np.zeros_like(test_l), np.ones(len(ood_d))])
        auroc = roc_auc_score(combined_l, combined_d)

        results["far"].append(far)
        results["ood_tpr"].append(ood_tpr)
        results["auroc"].append(auroc)
        results["test_d_all"].append(test_d)
        results["ood_d_all"].append(ood_d)
        results["ood_actions_all"].append(ood_action_ids)

        # Track per-fold and per-action metrics
        ood_tpr_by_action = {}
        for action_id in OOD_SUBSET:
            mask = ood_action_ids == action_id
            if mask.sum() > 0:
                ood_tpr_by_action[action_id] = (ood_alarm[mask].mean(), mask.sum())

        fold_details.append({
            'fold': fold_idx, 'test_subj': test_subj, 'far': far, 'ood_tpr': ood_tpr, 'auroc': auroc,
            'ood_tpr_by_action': ood_tpr_by_action, 'test_d': test_d, 'ood_d': ood_d, 'ood_actions': ood_action_ids
        })

        print(f"Fold {fold_idx}: FAR={far:.3f} | OOD TPR={ood_tpr:.3f} | AUROC={auroc:.3f}")

print("\n" + "="*70)
print("SUMMARY - MAHALANOBIS OOD DETECTION")
print("="*70)
if results["far"]:
    print(f"False Alarm Rate:  {np.mean(results['far']):.3f} ± {np.std(results['far']):.3f}  (5% target)")
    print(f"OOD Detection TPR: {np.mean(results['ood_tpr']):.3f} ± {np.std(results['ood_tpr']):.3f}  ✨")
    print(f"AUROC:             {np.mean(results['auroc']):.3f} ± {np.std(results['auroc']):.3f}")
else:
    print("No results collected. Check if OOD samples were loaded successfully.")
print("="*70)

# ============================================================================
# VISUALIZATION & ANALYSIS
# ============================================================================

if fold_details and results["far"]:
    from sklearn.metrics import roc_curve

    # 1. ROC CURVE
    all_test_d = np.concatenate(results["test_d_all"])
    all_ood_d = np.concatenate(results["ood_d_all"])
    all_labels = np.concatenate([np.zeros_like(all_test_d), np.ones_like(all_ood_d)])
    all_distances = np.concatenate([all_test_d, all_ood_d])

    fpr, tpr, thresholds = roc_curve(all_labels, all_distances)
    auroc_overall = roc_auc_score(all_labels, all_distances)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(fpr, tpr, 'b-', linewidth=2.5, label=f'ROC Curve (AUC={auroc_overall:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')

    # Mark the operating point (95th percentile threshold on calibration)
    first_op = True
    for detail in fold_details:
        test_d = detail['test_d']
        ood_d = detail['ood_d']
        combined_d = np.concatenate([test_d, ood_d])
        combined_l = np.concatenate([np.zeros_like(test_d), np.ones_like(ood_d)])
        dist_th = np.percentile(test_d, 95.0)

        fpr_pt = (test_d > dist_th).mean()
        tpr_pt = (ood_d > dist_th).mean()
        label = 'Operating Points (95th %ile)' if first_op else None
        ax.plot(fpr_pt, tpr_pt, 'ro', markersize=8, alpha=0.6, label=label)
        first_op = False

    ax.set_xlabel('False Positive Rate (ID samples wrongly flagged)', fontsize=11)
    ax.set_ylabel('True Positive Rate (OOD samples correctly detected)', fontsize=11)
    ax.set_title(f'OOD Detection ROC Curve ({args.modality.upper()})', fontsize=13, fontweight='bold')
    ax.grid(alpha=0.3)
    ax.legend(fontsize=11, loc='lower right')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    roc_path = Path("outputs") / f"ood_roc_{args.modality}.png"
    roc_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(roc_path), dpi=150, bbox_inches='tight')
    plt.show() if is_jupyter() else None
    plt.close()
    print(f"✓ ROC curve saved to {roc_path}")

    # 2. CONFUSION MATRIX (ID vs OOD Binary Classification)
    from sklearn.metrics import confusion_matrix

    # Aggregate across all folds
    binary_labels = np.concatenate([np.zeros_like(all_test_d), np.ones_like(all_ood_d)])
    # Use 95th percentile threshold on calibration set
    combined_threshold = np.percentile(all_test_d, 95.0)
    binary_predictions = (all_distances > combined_threshold).astype(int)

    cm = confusion_matrix(binary_labels, binary_predictions, labels=[0, 1])
    tn, fp = cm[0]
    fn, tp = cm[1]

    fig, ax = plt.subplots(figsize=(8, 8))
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    im = ax.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1)

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Predicted: ID', 'Predicted: OOD'], fontsize=11)
    ax.set_yticklabels(['True: ID', 'True: OOD'], fontsize=11)
    ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
    ax.set_title(f'OOD Detection Confusion Matrix ({args.modality.upper()})\nThreshold: 95th %ile',
                 fontsize=13, fontweight='bold')

    for i in range(2):
        for j in range(2):
            label_text = ['TN', 'FP', 'FN', 'TP'][i*2 + j]
            count = cm[i, j]
            rate = cm_norm[i, j]
            ax.text(j, i, f'{label_text}\n{count} ({rate:.1%})', ha='center', va='center',
                    fontsize=11, color='white' if rate > 0.5 else 'black', fontweight='bold')

    plt.colorbar(im, ax=ax, fraction=0.046, label='Normalized Rate')
    plt.tight_layout()

    cm_path = Path("outputs") / f"ood_confusion_{args.modality}.png"
    plt.savefig(str(cm_path), dpi=150, bbox_inches='tight')
    plt.show() if is_jupyter() else None
    plt.close()
    print(f"✓ Confusion matrix saved to {cm_path}")

    # 3. RESULTS TABLE
    # print("\n" + "="*100)
    # print("PER-FOLD RESULTS TABLE (detailed per fold + action breakdown)")
    # print("="*100)
    # print(f"{'Fold':<6} {'Test Subject':<15} {'FAR':<8} {'OOD TPR':<10} {'AUROC':<8} {'Action Breakdown':<50}")
    # print("-"*100)

    for detail in fold_details:
        action_str = ", ".join([f"a{a}:{detail['ood_tpr_by_action'].get(a, (0, 0))[0]:.2f}"
                               for a in OOD_SUBSET if a in detail['ood_tpr_by_action']])
        # print(f"{detail['fold']:<6} s{detail['test_subj']:<14} {detail['far']:<8.3f} {detail['ood_tpr']:<10.3f} "
              # f"{detail['auroc']:<8.3f} {action_str:<50}")

    # print("-"*100)
    # print(f"{'MEAN':<6} {'':<15} {np.mean(results['far']):<8.3f} {np.mean(results['ood_tpr']):<10.3f} "
    #       f"{np.mean(results['auroc']):<8.3f}")
    # print(f"{'STDEV':<6} {'':<15} {np.std(results['far']):<8.3f} {np.std(results['ood_tpr']):<10.3f} "
    #       f"{np.std(results['auroc']):<8.3f}")
    # print("="*100)

    # 3. PER-ACTION ANALYSIS
    # print("\n" + "="*70)
    # print("PER-ACTION OOD DETECTION ANALYSIS")
    # print("="*70)
    # print(f"{'Action':<10} {'Count':<8} {'Mean TPR':<12} {'Stdev TPR':<12} {'Misclass Dist':<15}")
    # print("-"*70)

    all_ood_actions = np.concatenate(results["ood_actions_all"])
    all_ood_dist = np.concatenate(results["ood_d_all"])

    for action_id in OOD_SUBSET:
        mask = all_ood_actions == action_id
        if mask.sum() > 0:
            action_dists = all_ood_dist[mask]
            # Get threshold from first fold (they're all the same)
            first_threshold = np.percentile(fold_details[0]['test_d'], 95.0)
            action_tprs = []
            misclass_dists = []

            for detail in fold_details:
                action_mask = detail['ood_actions'] == action_id
                if action_mask.sum() > 0:
                    action_tprs.append((detail['ood_d'][action_mask] > first_threshold).mean())
                    misclass_dists.extend(detail['ood_d'][action_mask & (detail['ood_d'] < first_threshold)])

            mean_tpr = np.mean(action_tprs) if action_tprs else 0
            std_tpr = np.std(action_tprs) if action_tprs else 0
            misclass_str = f"{np.mean(misclass_dists):.2f}" if misclass_dists else "N/A"

            # print(f"Action {action_id:<5} {mask.sum():<8} {mean_tpr:<12.3f} {std_tpr:<12.3f} {misclass_str:<15}")

    # print("="*70)
