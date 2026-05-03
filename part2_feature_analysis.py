"""
Part 2 — Feature Analysis: Permutation Importance

Identifies which of the 36 hand-crafted features are most/least important
using permutation importance. Retrains Random Forest with reduced feature set
and compares performance.

Run: python3 part2_feature_analysis.py
"""

import re
import random
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.io
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
import yaml

from data.sensor_dataset import extract_imu_features as extract_features

warnings.filterwarnings("ignore")

# ── Load config ────────────────────────────────────────────────────────────────
with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

SEED = cfg["seed"]
SUBSET = cfg["subset"]

random.seed(SEED)
np.random.seed(SEED)

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
INERTIAL_DIR = ROOT / cfg["paths"]["inertial_dir"]
SAMPLE_DIR = ROOT / cfg["paths"]["sample_dir"]
OUT_DIR = ROOT / cfg["paths"]["output_dir"]
OUT_DIR.mkdir(exist_ok=True)

# ── File index ─────────────────────────────────────────────────────────────────
fname_re = re.compile(r"a(\d+)_s(\d+)_t(\d+)_")

inertial_files = {}
for p in sorted(INERTIAL_DIR.glob("*_inertial.mat")):
    m = fname_re.match(p.name)
    if m:
        inertial_files[tuple(int(x) for x in m.groups())] = p

# ── Action names ───────────────────────────────────────────────────────────────
action_re = re.compile(r"^\s*(\d+)\.\s+(.+?)\s{2,}\(")
ACTION_NAMES = {}
for line in (SAMPLE_DIR / "Action_List.txt").read_text().splitlines():
    m = action_re.match(line)
    if m:
        ACTION_NAMES[int(m.group(1))] = m.group(2).strip()

CLASS_NAMES = [ACTION_NAMES[a] for a in SUBSET]

# ── Load dataset ───────────────────────────────────────────────────────────────
print("Loading inertial features...")
subset_set = set(SUBSET)
label_map = {a: i for i, a in enumerate(SUBSET)}

samples = []
for (a, s, t), p in inertial_files.items():
    if a not in subset_set:
        continue
    iner = scipy.io.loadmat(str(p))["d_iner"]
    samples.append((s, label_map[a], extract_features(iner), iner))

subjects = sorted(set(s for s, _, _, _ in samples))
print(f"  {len(samples)} samples | {len(subjects)} subjects | {len(SUBSET)} classes")

# ── Compute permutation importance ─────────────────────────────────────────────
print("\n── Computing permutation importance (LOSO) ──")

all_importances = []

for test_subject in subjects:
    train = [(y, x) for s, y, x, _ in samples if s != test_subject]
    test = [(y, x) for s, y, x, _ in samples if s == test_subject]

    X_train = np.array([x for _, x in train])
    y_train = np.array([y for y, _ in train])
    X_test = np.array([x for _, x in test])
    y_test = np.array([y for y, _ in test])

    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    clf = RandomForestClassifier(n_estimators=200, random_state=SEED)
    clf.fit(X_train, y_train)

    # Permutation importance
    perm_importance = permutation_importance(
        clf, X_test, y_test, n_repeats=10, random_state=SEED
    )
    all_importances.append(perm_importance.importances_mean)

# Average importances across folds
avg_importances = np.mean(all_importances, axis=0)
feature_names = []
for ch in range(6):
    ch_name = ["Ax", "Ay", "Az", "Gx", "Gy", "Gz"][ch]
    for feat in ["mean", "std", "RMS", "energy", "dom_freq", "spec_entropy"]:
        feature_names.append(f"{ch_name}_{feat}")

# Sort by importance
sorted_idx = np.argsort(avg_importances)[::-1]

print("\nTop 10 most important features:")
for i in range(10):
    idx = sorted_idx[i]
    print(f"  {i+1}. {feature_names[idx]:15s} — {avg_importances[idx]:.4f}")

print("\nBottom 10 least important features:")
for i in range(10):
    idx = sorted_idx[-(i+1)]
    print(f"  {i+1}. {feature_names[idx]:15s} — {avg_importances[idx]:.4f}")

# ── Visualize feature importance ───────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 12))

sorted_names = [feature_names[i] for i in sorted_idx]
colors = ["green" if avg_importances[i] > np.percentile(avg_importances, 50) else "red"
          for i in sorted_idx]

ax.barh(range(len(sorted_idx)), avg_importances[sorted_idx], color=colors, alpha=0.7)
ax.set_yticks(range(len(sorted_idx)))
ax.set_yticklabels(sorted_names, fontsize=9)
ax.set_xlabel("Permutation Importance", fontsize=11)
ax.set_title("Feature Importance — Random Forest (LOSO)", fontweight="bold", fontsize=12)
ax.axvline(np.percentile(avg_importances, 50), color="blue", linestyle="--",
           linewidth=1, alpha=0.5, label="Median")
ax.legend()
ax.grid(axis="x", alpha=0.3)

plt.tight_layout()
path = OUT_DIR / "part2_feature_importance.png"
plt.savefig(path, dpi=150)
plt.show()
print(f"\n✓ Saved {path}")

# ── Compare: original vs. reduced features ─────────────────────────────────────
# Remove only top 5 most negative features
n_remove = 5
features_to_remove = sorted_idx[-n_remove:]  # Bottom 5 (most negative)
features_to_keep = sorted_idx[:-n_remove]    # Everything else

n_keep = len(features_to_keep)

print(f"\n── Removing top {n_remove} most negative features ––")
print(f"Features to remove ({n_remove}):")
for idx in sorted(features_to_remove, key=lambda i: avg_importances[i]):
    print(f"  - {feature_names[idx]:15s} ({avg_importances[idx]:+.4f})")

print(f"\nFeatures to keep ({n_keep}): {n_keep} features (36 → {n_keep})")

# LOSO with reduced features
y_true_all_original = []
y_pred_all_original = []
y_true_all_reduced = []
y_pred_all_reduced = []

for test_subject in subjects:
    train = [(y, x) for s, y, x, _ in samples if s != test_subject]
    test = [(y, x) for s, y, x, _ in samples if s == test_subject]

    X_train = np.array([x for _, x in train])
    y_train = np.array([y for y, _ in train])
    X_test = np.array([x for _, x in test])
    y_test = np.array([y for y, _ in test])

    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Original (all features)
    clf_orig = RandomForestClassifier(n_estimators=200, random_state=SEED)
    clf_orig.fit(X_train_scaled, y_train)
    y_pred_orig = clf_orig.predict(X_test_scaled)
    y_true_all_original.extend(y_test)
    y_pred_all_original.extend(y_pred_orig)

    # Reduced features
    clf_red = RandomForestClassifier(n_estimators=200, random_state=SEED)
    clf_red.fit(X_train_scaled[:, features_to_keep], y_train)
    y_pred_red = clf_red.predict(X_test_scaled[:, features_to_keep])
    y_true_all_reduced.extend(y_test)
    y_pred_all_reduced.extend(y_pred_red)

acc_orig = accuracy_score(y_true_all_original, y_pred_all_original)
f1_orig = f1_score(y_true_all_original, y_pred_all_original, average="macro")

acc_red = accuracy_score(y_true_all_reduced, y_pred_all_reduced)
f1_red = f1_score(y_true_all_reduced, y_pred_all_reduced, average="macro")

print(f"\n{'Metric':<15} {'Original (36)':<20} {'Reduced ({n_keep})':<20} {'Δ':<10}")
print("-" * 65)
print(f"{'Accuracy':<15} {acc_orig:.3f}{' ':<16} {acc_red:.3f}{' ':<16} {acc_red-acc_orig:+.3f}")
print(f"{'Macro F1':<15} {f1_orig:.3f}{' ':<16} {f1_red:.3f}{' ':<16} {f1_red-f1_orig:+.3f}")

print(f"\n✓ Conclusion: Removing {n_remove} negative-importance features (36 → {n_keep})")
if acc_red >= acc_orig - 0.01:  # within 1% tolerance
    print(f"  ✓ Performance maintained or improved. Safe to reduce!")
    print(f"\n  Features to keep (indices): {sorted(features_to_keep.tolist())}")
else:
    print(f"  ⚠ Significant performance drop ({acc_red-acc_orig:+.3f}). Keep original features.")
