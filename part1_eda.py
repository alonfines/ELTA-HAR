"""
Part 1 — Exploratory Data Analysis

Loads config from config.yaml. Visualizes one sample per action class
and applies t-SNE to hand-crafted features. Displays plots inline and saves to outputs/.

Run: python3 part1_eda.py
"""

import os
import re
import random
import warnings
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import yaml

from data.sensor_dataset import extract_imu_features

warnings.filterwarnings("ignore")

# ── Load config ───────────────────────────────────────────────────────────────
with open("configs/config.yaml") as f:
    cfg = yaml.safe_load(f)

SEED = cfg["seed"]
SUBSET = cfg["subset"]

random.seed(SEED)
np.random.seed(SEED)

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
INERTIAL_DIR = ROOT / cfg["paths"]["inertial_dir"]
RGB_DIRS = [ROOT / d for d in cfg["paths"]["rgb_dirs"]]
SAMPLE_DIR = ROOT / cfg["paths"]["sample_dir"]
OUT_DIR = ROOT / cfg["paths"]["output_dir"]
OUT_DIR.mkdir(exist_ok=True)

required_dirs = [INERTIAL_DIR, SAMPLE_DIR] + RGB_DIRS
missing = [str(d) for d in required_dirs if not d.exists()]
if missing:
    raise FileNotFoundError("Missing expected directories:\n" + "\n".join(missing))

# ── Build file indexes ─────────────────────────────────────────────────────────
fname_re = re.compile(r"a(\d+)_s(\d+)_t(\d+)_")

def _index(paths):
    out = {}
    for p in paths:
        m = fname_re.match(p.name)
        if m:
            out[tuple(int(x) for x in m.groups())] = p
    return out

inertial_files = _index(sorted(INERTIAL_DIR.glob("*_inertial.mat")))
rgb_files = _index(sorted(p for d in RGB_DIRS for p in d.glob("*_color.avi")))

assert len(inertial_files) == 861, f"Expected 861 inertial files, got {len(inertial_files)}"
assert len(rgb_files) == 861, f"Expected 861 RGB files, got {len(rgb_files)}"

# ── Parse action names ─────────────────────────────────────────────────────────
action_re = re.compile(r"^\s*(\d+)\.\s+(.+?)\s{2,}\(")
ACTION_NAMES = {}
for line in (SAMPLE_DIR / "Action_List.txt").read_text().splitlines():
    m = action_re.match(line)
    if m:
        ACTION_NAMES[int(m.group(1))] = m.group(2).strip()

assert len(ACTION_NAMES) == 27, f"Expected 27 actions, parsed {len(ACTION_NAMES)}"
print(f"Indexed {len(inertial_files)} paired samples across {len(ACTION_NAMES)} actions.")

# ── Helpers ────────────────────────────────────────────────────────────────────
def load_iner(a, s, t):
    return scipy.io.loadmat(str(inertial_files[(a, s, t)]))["d_iner"]

COLORS = plt.cm.tab10.colors

# ── Visualize one sample per action ────────────────────────────────────────────
def plot_samples(actions_to_plot=[1, 19, 22]):
    """One example per action class: RGB frame + accel + gyro."""
    fig, axes = plt.subplots(len(actions_to_plot), 3, figsize=(16, 4 * len(actions_to_plot)))
    fig.suptitle("Sample recording per action — RGB frame + inertial signals",
                 fontweight="bold", fontsize=13)

    for row, a in enumerate(actions_to_plot):
        key = (a, 1, 1)  # subject 1, trial 1
        sensor = "wrist" if a <= 21 else "thigh"

        # RGB
        cap = cv2.VideoCapture(str(rgb_files[key]))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.set(cv2.CAP_PROP_POS_FRAMES, total // 2)
        ret, frame = cap.read()
        cap.release()
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if ret else None

        # Inertial
        iner = load_iner(*key)
        t = np.arange(len(iner))

        if rgb_frame is not None:
            axes[row][0].imshow(rgb_frame)
        axes[row][0].set_title(f"a{a}: {ACTION_NAMES[a]}  ({sensor})", fontsize=9)
        axes[row][0].axis("off")

        # Accelerometer
        for i, lbl in enumerate(["Ax", "Ay", "Az"]):
            axes[row][1].plot(t, iner[:, i], label=lbl)
        axes[row][1].set_ylabel("Accel (g)", fontsize=8)
        axes[row][1].legend(fontsize=7)
        axes[row][1].grid(True, alpha=0.3)

        # Gyroscope
        for i, lbl in enumerate(["Gx", "Gy", "Gz"]):
            axes[row][2].plot(t, iner[:, 3 + i], label=lbl)
        axes[row][2].set_ylabel("Gyro (°/s)", fontsize=8)
        axes[row][2].legend(fontsize=7)
        axes[row][2].grid(True, alpha=0.3)

    axes[-1][1].set_xlabel("Sample")
    axes[-1][2].set_xlabel("Sample")
    plt.tight_layout()
    path = OUT_DIR / "part1_samples.png"
    plt.savefig(path, dpi=150)
    plt.show()
    print(f"✓ Saved {path}")

# ── t-SNE visualization ────────────────────────────────────────────────────────
def plot_tsne(X_feats, y_labels):
    """t-SNE colored by class."""
    X_scaled = StandardScaler().fit_transform(X_feats)
    tsne = TSNE(n_components=2, perplexity=15, random_state=SEED, n_iter=1000, verbose=0)
    X_2d = tsne.fit_transform(X_scaled)

    fig, ax = plt.subplots(figsize=(8, 6))
    fig.suptitle("t-SNE of Inertial Features", fontweight="bold", fontsize=13)

    for i, a in enumerate(SUBSET):
        mask = y_labels == a
        marker = "o" if a <= 21 else "^"
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                   color=COLORS[i], marker=marker, s=80, alpha=0.75,
                   edgecolors='black', linewidth=0.5,
                   label=f"a{a}: {ACTION_NAMES[a]}")

    ax.set_title("Coloured by Class  (○ wrist  △ thigh)", fontsize=12, fontweight='bold')
    ax.legend(fontsize=9, bbox_to_anchor=(1.01, 1), loc="upper left", framealpha=0.9)
    ax.set_xlabel("t-SNE 1", fontsize=10)
    ax.set_ylabel("t-SNE 2", fontsize=10)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    path = OUT_DIR / "part1_tsne.png"
    plt.savefig(path, dpi=150)
    plt.show()
    print(f"✓ Saved {path}")

# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n── Plot 1: sample recordings ──")
    plot_samples()

    print("\n── Building feature matrix ──")
    subset_set = set(SUBSET)
    X_feats = []
    y_labels = []

    for (a, s, t) in inertial_files:
        if a not in subset_set:
            continue
        iner = load_iner(a, s, t)
        X_feats.append(extract_imu_features(iner))
        y_labels.append(a)

    X_feats = np.array(X_feats)
    y_labels = np.array(y_labels)
    print(f"   Feature matrix: {X_feats.shape}")

    print("\n── Plot 2: t-SNE ──")
    plot_tsne(X_feats, y_labels)

    print("\n✓ Done. Outputs saved to outputs/")
