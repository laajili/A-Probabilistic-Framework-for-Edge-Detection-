"""Run probabilistic edge detection on a folder of images."""

from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.edge_detection import detect_edges
from src.io_utils import load_gray01


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run probabilistic edge detection on an image folder.")
    parser.add_argument("--img_dir", type=str, required=True, help="Folder containing input images.")
    parser.add_argument("--out_dir", type=str, default="results/figures", help="Output folder for edge maps.")
    parser.add_argument("--lambda_param", type=float, default=5.0, help="Sensitivity parameter lambda.")
    parser.add_argument("--gaussian_sigma", type=float, default=0.6, help="Optional Gaussian smoothing sigma.")
    parser.add_argument("--threshold", type=float, default=None, help="Optional threshold for binary edge maps.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_paths = sorted(glob.glob(os.path.join(args.img_dir, "*.*")))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for image_path in tqdm(image_paths, desc="Probabilistic edge detection"):
        image = load_gray01(image_path)
        score = detect_edges(image, lambda_param=args.lambda_param, gaussian_sigma=args.gaussian_sigma)

        stem = Path(image_path).stem
        plt.imsave(out_dir / f"{stem}_edge_probability.png", score, cmap="gray")

        if args.threshold is not None:
            binary = score >= args.threshold
            plt.imsave(out_dir / f"{stem}_binary_t{args.threshold:.3f}.png", binary, cmap="gray")


if __name__ == "__main__":
    main()
