"""
Machine Learning Utilities: calibration, conformal prediction, and OOD detection.
Used by Part 6 (confidence) and Part 7 (OOD detection).
"""

from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import minimize_scalar
from scipy.spatial.distance import cdist


# ── Temperature Scaling ────────────────────────────────────────────────────────

def fit_temperature(logits: np.ndarray, y_true: np.ndarray) -> float:
    """Find T* = argmin_{T>0} NLL(logits/T, y) via bounded scalar optimisation."""
    def nll(T: float) -> float:
        return F.cross_entropy(
            torch.tensor(logits / T, dtype=torch.float32),
            torch.tensor(y_true, dtype=torch.long),
        ).item()

    return float(minimize_scalar(nll, bounds=(0.05, 10.0), method="bounded").x)


def compute_ece(
    logits: np.ndarray,
    y_true: np.ndarray,
    T: float = 1.0,
    n_bins: int = 10,
) -> float:
    """Expected Calibration Error with uniform confidence bins."""
    probs   = torch.softmax(torch.tensor(logits / T, dtype=torch.float32), dim=1).numpy()
    conf    = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == y_true).astype(float)
    ece     = 0.0
    for lo, hi in zip(np.linspace(0, 1, n_bins + 1)[:-1],
                      np.linspace(0, 1, n_bins + 1)[1:]):
        mask = (conf > lo) & (conf <= hi)
        if mask.sum() == 0:
            continue
        ece += mask.mean() * abs(correct[mask].mean() - conf[mask].mean())
    return float(ece)


# ── Split Conformal Prediction ────────────────────────────────────────────────

def nonconf_scores(logits: np.ndarray, y_true: np.ndarray, T: float) -> np.ndarray:
    """s_i = 1 − softmax(z_i/T)[y_i]  (low score = high confidence in true class)."""
    probs = torch.softmax(torch.tensor(logits / T, dtype=torch.float32), dim=1).numpy()
    return 1.0 - probs[np.arange(len(y_true)), y_true]


def cp_quantile(scores: np.ndarray, alpha: float) -> float:
    """
    Finite-sample conformal quantile.
    Coverage guarantee: P(y ∈ C(x)) ≥ 1−α when cal/test are exchangeable.
    """
    n   = len(scores)
    idx = min(int(np.ceil((n + 1) * (1 - alpha))) - 1, n - 1)
    return float(np.sort(scores)[idx])


def cp_predict_sets(
    logits: np.ndarray,
    T: float,
    q_hat: float,
) -> List[np.ndarray]:
    """C(x) = { y : softmax(z/T)[y] ≥ 1 − q̂ }."""
    probs = torch.softmax(torch.tensor(logits / T, dtype=torch.float32), dim=1).numpy()
    return [np.where(p >= 1 - q_hat)[0] for p in probs]


def empirical_coverage(sets: List[np.ndarray], y_true: np.ndarray) -> float:
    """Fraction of true labels in their respective prediction sets."""
    return float(np.mean([y in s for s, y in zip(sets, y_true)]))


def mean_set_size(sets: List[np.ndarray]) -> float:
    """Average cardinality of prediction sets."""
    return float(np.mean([len(s) for s in sets]))


# ── OOD Detection Helpers ──────────────────────────────────────────────────────

def fit_gaussian_embeddings(e_v: np.ndarray, e_s: np.ndarray) -> tuple:
    """Fit multivariate Gaussian (mean, cov) on training embeddings."""
    mu_v, cov_v = e_v.mean(0), np.cov(e_v.T)
    mu_s, cov_s = e_s.mean(0), np.cov(e_s.T)
    return mu_v, cov_v, mu_s, cov_s


def compute_mahal_distance(x: np.ndarray, mu: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """Mahalanobis distance: sqrt((x - mu)^T Σ^-1 (x - mu)) per sample."""
    try:
        cov_inv = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        cov_inv = np.linalg.pinv(cov)
    diff = x - mu[np.newaxis]
    mahal = np.sqrt((diff @ cov_inv * diff).sum(1))
    return mahal


def compute_entropy(logits: np.ndarray) -> np.ndarray:
    """Shannon entropy of softmax probabilities."""
    probs = np.exp(logits - logits.max(1, keepdims=True))
    probs /= probs.sum(1, keepdims=True)
    entropy = -(probs * np.log(probs + 1e-10)).sum(1)
    return entropy


def compute_knn_distance(x: np.ndarray, x_train: np.ndarray, k: int = 5) -> np.ndarray:
    """Distance to k-th nearest neighbor in training set."""
    dists = cdist(x, x_train, metric="euclidean")
    knn_dist = np.sort(dists, axis=1)[:, k]
    return knn_dist


def compute_ood_scores(
    logits_s: np.ndarray,
    logits_v: np.ndarray,
    logits_f: np.ndarray,
    e_s: np.ndarray,
    e_v: np.ndarray,
    e_s_train: np.ndarray,
    e_v_train: np.ndarray,
    mu_s: np.ndarray,
    cov_s: np.ndarray,
    mu_v: np.ndarray,
    cov_v: np.ndarray,
    cp_sets: List[np.ndarray],
    k_nn: int = 5,
) -> Dict[str, np.ndarray]:
    """Compute all 6 OOD scores. Returns dict of arrays (N,)."""
    n = len(logits_s)

    # 1. Entropy
    entropy_s = compute_entropy(logits_s)
    entropy_v = compute_entropy(logits_v)
    entropy_f = compute_entropy(logits_f)

    # 2. Max softmax probability (inverse)
    max_prob_s = np.exp(logits_s - logits_s.max(1, keepdims=True)).max(1)
    max_prob_v = np.exp(logits_v - logits_v.max(1, keepdims=True)).max(1)
    max_prob_f = np.exp(logits_f - logits_f.max(1, keepdims=True)).max(1)

    # 3. Mahalanobis distance
    mahal_s = compute_mahal_distance(e_s, mu_s, cov_s)
    mahal_v = compute_mahal_distance(e_v, mu_v, cov_v)

    # 4. k-NN distance
    knn_s = compute_knn_distance(e_s, e_s_train, k=k_nn)
    knn_v = compute_knn_distance(e_v, e_v_train, k=k_nn)

    # 5. Conformal prediction set size
    cp_sizes = np.array([len(s) for s in cp_sets])

    # 6. Sensor-Video disagreement
    pred_s = logits_s.argmax(1)
    pred_v = logits_v.argmax(1)
    disagreement = (pred_s != pred_v).astype(float)

    return {
        "entropy_s": entropy_s, "entropy_v": entropy_v, "entropy_f": entropy_f,
        "maxprob_s": -max_prob_s, "maxprob_v": -max_prob_v, "maxprob_f": -max_prob_f,
        "mahal_s": mahal_s, "mahal_v": mahal_v,
        "knn_s": knn_s, "knn_v": knn_v,
        "cp_size": cp_sizes,
        "disagreement": disagreement,
    }


def normalize_scores(scores: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Normalize each score to [0, 1] using percentile scaling."""
    normalized = {}
    for key, val in scores.items():
        vmin, vmax = np.percentile(val, [5, 95])
        if vmax > vmin:
            normalized[key] = np.clip((val - vmin) / (vmax - vmin), 0, 1)
        else:
            normalized[key] = np.zeros_like(val)
    return normalized


def combine_scores(normalized: Dict[str, np.ndarray], weights: Dict[str, float]) -> np.ndarray:
    """Weighted average of normalized scores."""
    combined = np.zeros(len(next(iter(normalized.values()))))
    total_w = 0.0
    for key, w in weights.items():
        # Average across modalities if multiple variants exist
        if key in normalized:
            combined += w * normalized[key]
            total_w += w
        elif f"{key}_f" in normalized:
            combined += w * normalized[f"{key}_f"]
            total_w += w
        elif f"{key}_s" in normalized and f"{key}_v" in normalized:
            combined += w * 0.5 * (normalized[f"{key}_s"] + normalized[f"{key}_v"])
            total_w += w
    return combined / max(total_w, 1e-6)
