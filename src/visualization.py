"""Visualization utilities for probabilistic edge detection."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def visualize_thresholds(
    image: np.ndarray,
    score01: np.ndarray,
    thresholds: Iterable[float] = (0.3, 0.5, 0.575, 0.7),
    title: Optional[str] = None,
    save_path: Optional[str] = None,
) -> None:
    """Display original image, probability score map, and binarized maps."""
    thresholds = list(thresholds)
    cols = 2 + len(thresholds)

    plt.figure(figsize=(3 * cols, 4))
    plt.subplot(1, cols, 1)
    plt.imshow(image, cmap="gray")
    plt.title("Input image")
    plt.axis("off")

    plt.subplot(1, cols, 2)
    plt.imshow(score01, cmap="gray")
    plt.title("Edge probability map")
    plt.axis("off")

    for index, threshold in enumerate(thresholds, start=3):
        plt.subplot(1, cols, index)
        plt.imshow(score01 >= threshold, cmap="gray")
        plt.title(f"Binary map t={threshold:.3f}")
        plt.axis("off")

    if title:
        plt.suptitle(title, y=1.02)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def plot_pr_curve(df: pd.DataFrame, save_path: Optional[str] = None) -> None:
    """Plot the precision-recall curve."""
    curve = df.sort_values("recall")
    plt.figure(figsize=(5, 4))
    plt.plot(curve["recall"], curve["precision"], marker="o")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall curve")
    plt.grid(True)
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def plot_f1_vs_threshold(df: pd.DataFrame, save_path: Optional[str] = None) -> None:
    """Plot F1-score as a function of the binarization threshold."""
    best = df.iloc[df["f1"].values.argmax()]
    plt.figure(figsize=(5, 4))
    plt.plot(df["threshold"], df["f1"], marker="o")
    plt.axvline(float(best["threshold"]), linestyle="--", label=f"t*={best['threshold']:.3f}")
    plt.xlabel("Threshold")
    plt.ylabel("F1-score")
    plt.title(f"F1 vs threshold — ODS={best['f1']:.3f}")
    plt.grid(True)
    plt.legend()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
