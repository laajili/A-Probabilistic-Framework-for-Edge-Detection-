"""Input/output utilities for probabilistic edge detection."""

from __future__ import annotations

import glob
import os
from collections import defaultdict
from typing import Dict, List

import numpy as np
from skimage import color, io


def load_gray01(path: str) -> np.ndarray:
    """Load an image as a grayscale float32 array normalized in [0, 1].

    Parameters
    ----------
    path:
        Path to the input image.

    Returns
    -------
    np.ndarray
        Grayscale image normalized in [0, 1].
    """
    img = io.imread(path)
    if img.ndim == 3:
        gray = color.rgb2gray(img).astype(np.float32)
    else:
        gray = img.astype(np.float32)

    if gray.max() > 1.0:
        gray = gray / 255.0

    dynamic_range = float(np.ptp(gray))
    if dynamic_range < 1e-6:
        return np.zeros_like(gray, dtype=np.float32)

    return ((gray - gray.min()) / dynamic_range).astype(np.float32)


def build_image_to_labels(img_dir: str, gt_dir: str) -> Dict[str, List[str]]:
    """Build a mapping between images and their ground-truth edge labels.

    The function supports two naming conventions:
    - image_name_*.png for multiple labels per image;
    - image_name.png for a single label.
    """
    image_paths = sorted(glob.glob(os.path.join(img_dir, "*.*")))
    mapping: Dict[str, List[str]] = defaultdict(list)

    for image_path in image_paths:
        stem = os.path.splitext(os.path.basename(image_path))[0]
        labels = sorted(glob.glob(os.path.join(gt_dir, f"{stem}_*.png")))

        if not labels:
            alternative = os.path.join(gt_dir, f"{stem}.png")
            if os.path.exists(alternative):
                labels = [alternative]

        mapping[image_path].extend(labels)

    return dict(mapping)


def build_image_to_multicue_labels(img_dir: str, gt_root: str) -> Dict[str, List[str]]:
    """Build image-to-label mapping for the Multicue folder layout.

    Expected structure:
    - images are stored directly in ``img_dir``;
    - each annotator folder is nested under ``gt_root``;
    - labels use the same filename as the corresponding image.
    """
    image_paths = sorted(glob.glob(os.path.join(img_dir, "*.*")))
    mapping: Dict[str, List[str]] = defaultdict(list)

    for image_path in image_paths:
        filename = os.path.basename(image_path)
        labels = sorted(glob.glob(os.path.join(gt_root, "**", filename), recursive=True))
        mapping[image_path].extend(labels)

    return dict(mapping)
