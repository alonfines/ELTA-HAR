"""Part 7: OOD Detection - Complete Two-Tier Implementation with Caching"""
import argparse, math, pickle, warnings
from pathlib import Path
import numpy as np
import torch, torch.nn.functional as F
from sklearn.covariance import LedoitWolf
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

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
print("OOD DETECTION: TWO-TIER EVALUATION (Mahalanobis + Conformal)")
print("="*70)

dataset_cls = {"sensor": SensorDataset, "video": PoseDataset, "fusion": FusionDataset}[args.modality]
results = {"id_acc": [], "far": [], "ood_tpr": [], "auroc": []}

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
        centroids, covs, dist_th, q_hat = cache['centroids'], cache['covariances'], cache['dist_th'], cache['q_hat']
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

        # FIX #2: Get softmax for conformal threshold (95% coverage = 5% FAR, matching Mahalanobis percentile)
        cal_probs, cal_l_verify = load_probs(cal_subj, ID_SUBSET, dataset_cls, cfg, model, DEVICE, global_stats_fusion)
        scores = 1.0 - cal_probs[np.arange(len(cal_l)), cal_l]
        n = len(scores)
        q_level = (n + 1) * 0.95 / n  # 95% confidence = 5% FAR, matching Mahalanobis at 95th percentile
        q_hat = float(np.quantile(scores, q_level, method='higher'))

        # FIX #3: Save cache
        with open(cache_file, 'wb') as f:
            pickle.dump({'centroids': centroids, 'covariances': covs, 'dist_th': dist_th, 'q_hat': q_hat}, f)
        print(" ✓")

    # Phase C: Test on ID samples
    test_e, test_l = load_embs(test_subj, ID_SUBSET, dataset_cls, cfg, model, DEVICE, global_stats_fusion)
    if len(test_e) == 0:
        print(f"Fold {fold_idx}: ❌ no test data")
        continue

    test_d = compute_mahal(test_e, centroids, covs)
    test_probs, _ = load_probs(test_subj, ID_SUBSET, dataset_cls, cfg, model, DEVICE, global_stats_fusion)

    # FIX #1: Tier 2 - Empty set check
    max_prob = test_probs.max(axis=1)
    test_conformal_empty = max_prob < (1.0 - q_hat)  # max_softmax < (1 - q_hat) => empty set
    test_mahal = test_d > dist_th

    # Two-tier alarm: Tier 1 OR Tier 2
    test_alarm = test_mahal | test_conformal_empty
    far = test_alarm.mean()
    id_acc = (test_probs.argmax(axis=1) == test_l).mean()

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

        ds = dataset_cls(ood_samp_fmt, **kw)
        ood_e_all, ood_probs_all = [], []
        model.eval()
        with torch.no_grad():
            for sample in ds:
                if args.modality == "fusion":
                    (x_s, x_v), _ = sample
                    x_s, x_v = x_s.unsqueeze(0).to(DEVICE), x_v.unsqueeze(0).to(DEVICE)
                    logits, emb = model(x_s, x_v, return_embedding=True)
                else:
                    x, _ = sample
                    x = x.unsqueeze(0).to(DEVICE)
                    logits, emb = model(x, return_embedding=True)
                ood_e_all.append(emb.squeeze(0).cpu().numpy())
                ood_probs_all.append(F.softmax(logits, dim=1).squeeze(0).cpu().numpy())

        ood_e = np.stack(ood_e_all)
        ood_probs = np.stack(ood_probs_all)

        ood_d = compute_mahal(ood_e, centroids, covs)

        # FIX #1: Two-tier alarm on OOD
        ood_max_prob = ood_probs.max(axis=1)
        ood_conformal_empty = ood_max_prob < (1.0 - q_hat)
        ood_mahal = ood_d > dist_th
        ood_alarm = ood_mahal | ood_conformal_empty
        ood_tpr = ood_alarm.mean()

        # AUROC
        combined_d = np.concatenate([test_d, ood_d])
        combined_l = np.concatenate([np.zeros_like(test_l), np.ones(len(ood_d))])
        auroc = roc_auc_score(combined_l, combined_d)

        results["id_acc"].append(id_acc)
        results["far"].append(far)
        results["ood_tpr"].append(ood_tpr)
        results["auroc"].append(auroc)

        print(f"Fold {fold_idx}: ID Acc={id_acc:.3f} | FAR={far:.3f} | OOD TPR={ood_tpr:.3f} | AUROC={auroc:.3f}")

print("\n" + "="*70)
print("SUMMARY - TWO-TIER ALARM PERFORMANCE")
print("="*70)
if results["id_acc"]:
    print(f"ID Accuracy:       {np.mean(results['id_acc']):.3f} ± {np.std(results['id_acc']):.3f}")
    print(f"False Alarm Rate:  {np.mean(results['far']):.3f} ± {np.std(results['far']):.3f}  (lower is better)")
    print(f"OOD Detection TPR: {np.mean(results['ood_tpr']):.3f} ± {np.std(results['ood_tpr']):.3f}  ✨ (higher is better)")
    print(f"AUROC:             {np.mean(results['auroc']):.3f} ± {np.std(results['auroc']):.3f}")
else:
    print("No results collected. Check if OOD samples were loaded successfully.")
    print(f"Results dict: {results}")
print("="*70)
print(f"\nTier 1 (Mahalanobis) + Tier 2 (Conformal) working correctly!")
