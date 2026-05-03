"""
Part 2 — Random Forest Sensor Classifier

Trains a Random Forest classifier on 36-dim hand-crafted inertial features
using Leave-One-Subject-Out cross-validation.

Run: python3 part2_random_forest.py
"""

import re
import random
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.io
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
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

for d in [INERTIAL_DIR, SAMPLE_DIR]:
    if not d.exists():
        raise FileNotFoundError(f"Missing directory: {d}")

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

# ── LOSO cross-validation ─────────────────────────────────────────────────────
clf = RandomForestClassifier(n_estimators=200, random_state=SEED)

y_true_all = []
y_pred_all = []
fold_accuracies = []
fold_f1_scores = []

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

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_true_all.extend(y_test)
    y_pred_all.extend(y_pred)

    fold_acc = accuracy_score(y_test, y_pred)
    fold_f1 = f1_score(y_test, y_pred, average="macro")
    fold_accuracies.append(fold_acc)
    fold_f1_scores.append(fold_f1)

y_true_all = np.array(y_true_all)
y_pred_all = np.array(y_pred_all)

# ── Results summary ────────────────────────────────────────────────────────────
acc = accuracy_score(y_true_all, y_pred_all)
f1 = f1_score(y_true_all, y_pred_all, average="macro")
acc_std = np.std(fold_accuracies)
f1_std = np.std(fold_f1_scores)

print(f"\n✓ Random Forest — Accuracy: {acc:.3f} ± {acc_std:.3f} | Macro F1: {f1:.3f} ± {f1_std:.3f}")

# ── Confusion matrix ───────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 8))
cm = confusion_matrix(y_true_all, y_pred_all)
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
ax.set_xticks(range(len(SUBSET)))
ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right", fontsize=9)
ax.set_yticks(range(len(SUBSET)))
ax.set_yticklabels(CLASS_NAMES, fontsize=9)
ax.set_xlabel("Predicted", fontsize=11)
ax.set_ylabel("True", fontsize=11)
ax.set_title(f"Random Forest — Accuracy={acc:.3f}±{acc_std:.3f} | F1={f1:.3f}±{f1_std:.3f}", fontweight="bold", fontsize=12)

for i in range(len(SUBSET)):
    for j in range(len(SUBSET)):
        ax.text(j, i, f"{cm_norm[i,j]:.2f}", ha="center", va="center",
                fontsize=8, color="white" if cm_norm[i,j] > 0.5 else "black")

plt.colorbar(im, ax=ax, fraction=0.046)
plt.tight_layout()
path = OUT_DIR / "part2_classical_confusion.png"
plt.savefig(path, dpi=150)
plt.show()
print(f"\n✓ Saved {path}")

# ── Failure case: swipe left vs swipe right ────────────────────────────────────
print("\n── Failure case: knock on door (a19) vs boxing (a13) ––")

action_pair = {label_map[19]: "knock on door", label_map[13]: "boxing"}
misclassified = []

for test_subject in subjects:
    train = [(y, x) for s, y, x, _ in samples if s != test_subject]
    test = [(y, x, raw) for s, y, x, raw in samples
            if s == test_subject and y in action_pair]

    X_train = np.array([x for _, x in train])
    y_train = np.array([y for y, _ in train])
    scaler = StandardScaler().fit(X_train)

    clf_fold = RandomForestClassifier(n_estimators=200, random_state=SEED)
    clf_fold.fit(scaler.transform(X_train), y_train)

    for y_true, x, raw in test:
        y_pred = clf_fold.predict(scaler.transform(x.reshape(1, -1)))[0]
        if y_pred != y_true:
            misclassified.append((test_subject, y_true, y_pred, raw))

print(f"  Misclassified knock/boxing samples: {len(misclassified)}")

if misclassified:
    s, y_true, y_pred, raw = misclassified[0]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(
        f"Failure Case — Subject {s}: true={action_pair[y_true]}, predicted={action_pair[y_pred]}",
        fontweight="bold"
    )
    t = np.arange(len(raw))

    for i, lbl in enumerate(["Ax", "Ay", "Az"]):
        axes[0].plot(t, raw[:, i], label=lbl)
    axes[0].set_title("Accelerometer")
    axes[0].set_xlabel("Sample")
    axes[0].set_ylabel("g")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    for i, lbl in enumerate(["Gx", "Gy", "Gz"]):
        axes[1].plot(t, raw[:, 3 + i], label=lbl)
    axes[1].set_title("Gyroscope")
    axes[1].set_xlabel("Sample")
    axes[1].set_ylabel("°/s")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    path = OUT_DIR / "part2_classical_failure.png"
    plt.savefig(path, dpi=150)
    plt.show()
    print(f"✓ Saved {path}")
else:
    print("  No knock/boxing misclassifications found — classes fully separated.")

print("\n✓ Done.")
