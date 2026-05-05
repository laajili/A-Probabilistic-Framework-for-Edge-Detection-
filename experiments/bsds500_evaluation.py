"""BSDS500-style evaluation script for the probabilistic edge detector."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.edge_detection import detect_edges
from src.evaluation import bsds_tolerance_px, compute_ods_ois_ap
from src.io_utils import build_image_to_labels, load_gray01


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate probabilistic edge detection with BSDS-style metrics.")
    parser.add_argument("--img_dir", type=str, required=True, help="Folder containing test images.")
    parser.add_argument("--gt_dir", type=str, required=True, help="Folder containing ground-truth labels.")
    parser.add_argument("--out_dir", type=str, default="results/metrics", help="Output folder for CSV metrics.")
    parser.add_argument("--lambda_param", type=float, default=5.0, help="Sensitivity parameter lambda.")
    parser.add_argument("--gaussian_sigma", type=float, default=0.6, help="Optional Gaussian smoothing sigma.")
    parser.add_argument("--n_thresholds", type=int, default=530, help="Number of thresholds between 0 and 1.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    image_to_labels = build_image_to_labels(args.img_dir, args.gt_dir)
    thresholds = np.linspace(0.0, 1.0, args.n_thresholds)

    scores = {}
    ground_truths = {}
    tolerances = {}

    for image_path, label_paths in tqdm(image_to_labels.items(), desc="Detection"):
        image = load_gray01(image_path)
        score = detect_edges(image, lambda_param=args.lambda_param, gaussian_sigma=args.gaussian_sigma)
        scores[image_path] = score
        tolerances[image_path] = bsds_tolerance_px(*image.shape)

        gt_list = []
        for label_path in label_paths:
            gt = load_gray01(label_path) > 0.5
            gt_list.append(gt)
        if gt_list:
            ground_truths[image_path] = gt_list

    df_ods, summary = compute_ods_ois_ap(scores, ground_truths, tolerances, thresholds)

    df_ods.to_csv(out_dir / "ods_pr_curve.csv", index=False)
    pd.DataFrame([summary]).to_csv(out_dir / "summary_metrics.csv", index=False)

    print("Evaluation summary:")
    for key, value in summary.items():
        print(f"  {key}: {value:.4f}")


if __name__ == "__main__":
    main()
