"""Core probabilistic edge detector based on local intensity differences."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter


def robust_normalize_01(score: np.ndarray, robust: bool = True) -> np.ndarray:
    """Normalize a score map into [0, 1].

    When robust=True, percentile normalization is used to reduce the influence
    of extreme values.
    """
    score = score.astype(np.float32)

    if robust:
        p1, p99 = np.percentile(score, 1.0), np.percentile(score, 99.0)
        denom = max(float(p99 - p1), 1e-6)
        return np.clip((score - p1) / denom, 0.0, 1.0).astype(np.float32)

    mn, mx = float(score.min()), float(score.max())
    denom = max(mx - mn, 1e-8)
    return ((score - mn) / denom).astype(np.float32)


def probabilistic_contours(image: np.ndarray, lambda_param: float = 5.0) -> np.ndarray:
    """Compute the probabilistic edge map.

    The method follows the article formulation:

        P_C(i,j) = 1 - exp(-lambda * sum_{(k,l) in N(i,j)} |I(i,j)-I(k,l)|)

    with N(i,j) given by the four direct neighbors: up, down, left, right.
    """
    if image.ndim != 2:
        raise ValueError("probabilistic_contours expects a 2D grayscale image.")
    if lambda_param <= 0:
        raise ValueError("lambda_param must be strictly positive.")

    img = image.astype(np.float32)
    local_variation = np.zeros_like(img, dtype=np.float32)

    center = img[1:-1, 1:-1]
    local_variation[1:-1, 1:-1] = (
        np.abs(center - img[:-2, 1:-1])
        + np.abs(center - img[2:, 1:-1])
        + np.abs(center - img[1:-1, :-2])
        + np.abs(center - img[1:-1, 2:])
    )

    return (1.0 - np.exp(-lambda_param * local_variation)).astype(np.float32)


def detect_edges(
    image: np.ndarray,
    lambda_param: float = 5.0,
    gaussian_sigma: float = 0.6,
    robust_norm: bool = True,
) -> np.ndarray:
    """End-to-end edge detection: optional smoothing, probability map, normalization."""
    img = image.astype(np.float32)
    if gaussian_sigma > 0:
        img = gaussian_filter(img, sigma=gaussian_sigma)

    score = probabilistic_contours(img, lambda_param=lambda_param)
    return robust_normalize_01(score, robust=robust_norm)
