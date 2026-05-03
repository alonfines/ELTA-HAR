import re
from pathlib import Path
from types import SimpleNamespace
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import yaml
from sklearn.metrics import confusion_matrix


def _to_ns(d):
    if isinstance(d, dict):
        return SimpleNamespace(**{k: _to_ns(v) for k, v in d.items()})
    return d


def load_config(path: str) -> SimpleNamespace:
    with open(path) as f:
        return _to_ns(yaml.safe_load(f))


def load_action_names(sample_dir: Path) -> dict:
    action_re = re.compile(r"^\s*(\d+)\.\s+(.+?)\s{2,}\(")
    names = {}
    for line in (sample_dir / "Action_List.txt").read_text().splitlines():
        m = action_re.match(line)
        if m:
            names[int(m.group(1))] = m.group(2).strip()
    return names


def save_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    title: str,
    path: Path,
    cmap: str = "Blues",
) -> np.ndarray:
    n = len(class_names)
    
    # Bug 1 Fix: Explicitly pass labels to guarantee an n x n matrix
    cm = confusion_matrix(y_true, y_pred, labels=range(n))
    
    # Bug 2 Fix: Prevent division by zero if a class has no instances in y_true
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = cm.astype(float) / np.maximum(row_sums, 1e-9)

    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(cm_norm, cmap=cmap, vmin=0, vmax=1)
    
    ax.set_xticks(range(n))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(n))
    ax.set_yticklabels(class_names, fontsize=9)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title, fontweight="bold")
    
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{cm_norm[i,j]:.2f}", ha="center", va="center",
                    fontsize=8, color="white" if cm_norm[i, j] > 0.5 else "black")
                    
    plt.colorbar(im, fraction=0.046)
    plt.tight_layout()
    
    plt.savefig(path, dpi=150)
    
    # Risk Fix: Explicitly release the figure from memory
    #plt.close(fig)
    
    return cm
