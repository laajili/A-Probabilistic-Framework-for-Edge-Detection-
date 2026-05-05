"""Grid search over lambda and threshold values for BSDS-style evaluation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.edge_detection import detect_edges
from src.evaluation import bsds_tolerance_px, compute_ods_ois_ap
from src.io_utils import build_image_to_labels, build_image_to_multicue_labels, load_gray01


def parse_float_list(value: str) -> List[float]:
    """Parse a comma-separated list of floats."""
    try:
        values = [float(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid float list: {value}") from exc

    if not values:
        raise argparse.ArgumentTypeError("At least one value is required.")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search best lambda and threshold on BSDS-style metrics.")
    parser.add_argument("--img_dir", type=str, required=True, help="Folder containing test or validation images.")
    parser.add_argument("--gt_dir", type=str, required=True, help="Folder containing ground-truth labels.")
    parser.add_argument("--out_dir", type=str, default="results/metrics/grid_search", help="Output folder.")
    parser.add_argument(
        "--label_mode",
        choices=("bsds", "multicue"),
        default="bsds",
        help="Ground-truth layout: flat BSDS-style labels or nested Multicue annotator folders.",
    )
    parser.add_argument(
        "--lambda_values",
        type=parse_float_list,
        default=parse_float_list("0.05,0.1,0.5,1,2,5,10,20,50"),
        help="Comma-separated lambda values, for example: 0.5,1,2,5,10.",
    )
    parser.add_argument(
        "--threshold_values",
        type=parse_float_list,
        default=None,
        help="Optional comma-separated threshold values in [0, 1]. Overrides --n_thresholds.",
    )
    parser.add_argument(
        "--n_thresholds",
        type=int,
        default=200,
        help="Number of evenly spaced thresholds in [0, 1] when --threshold_values is not provided.",
    )
    parser.add_argument("--gaussian_sigma", type=float, default=0.6, help="Optional Gaussian smoothing sigma.")
    parser.add_argument("--max_images", type=int, default=None, help="Optional maximum number of images.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.threshold_values is not None:
        thresholds = np.array(sorted(set(args.threshold_values)), dtype=float)
    else:
        if args.n_thresholds < 2:
            raise ValueError("--n_thresholds must be at least 2.")
        thresholds = np.linspace(0.0, 1.0, args.n_thresholds)

    if np.any((thresholds < 0.0) | (thresholds > 1.0)):
        raise ValueError("All threshold values must be in [0, 1].")

    if args.label_mode == "multicue":
        image_to_labels = build_image_to_multicue_labels(args.img_dir, args.gt_dir)
    else:
        image_to_labels = build_image_to_labels(args.img_dir, args.gt_dir)

    if args.max_images is not None:
        image_to_labels = dict(list(image_to_labels.items())[: args.max_images])
    images = {}
    ground_truths = {}
    tolerances = {}

    for image_path, label_paths in tqdm(image_to_labels.items(), desc="Loading data"):
        image = load_gray01(image_path)
        images[image_path] = image
        tolerances[image_path] = bsds_tolerance_px(*image.shape)

        gt_list = [load_gray01(label_path) > 0.5 for label_path in label_paths]
        if gt_list:
            ground_truths[image_path] = gt_list

    all_curves = []
    lambda_summaries = []

    for lambda_param in tqdm(args.lambda_values, desc="Lambda search"):
        scores = {
            image_path: detect_edges(image, lambda_param=lambda_param, gaussian_sigma=args.gaussian_sigma)
            for image_path, image in images.items()
        }

        curve, summary = compute_ods_ois_ap(scores, ground_truths, tolerances, thresholds)
        curve.insert(0, "lambda_param", float(lambda_param))
        all_curves.append(curve)

        lambda_summary = {"lambda_param": float(lambda_param), **summary}
        lambda_summaries.append(lambda_summary)

    grid_df = pd.concat(all_curves, ignore_index=True)
    summary_df = pd.DataFrame(lambda_summaries).sort_values("ODS", ascending=False)
    best_row = summary_df.iloc[[0]]

    grid_df.to_csv(out_dir / "grid_results.csv", index=False)
    summary_df.to_csv(out_dir / "lambda_summary.csv", index=False)
    best_row.to_csv(out_dir / "best_params.csv", index=False)

    best = best_row.iloc[0]
    print("Best parameters:")
    print(f"  lambda_param: {best['lambda_param']:.6g}")
    print(f"  threshold: {best['threshold_star']:.6g}")
    print(f"  ODS/F1: {best['ODS']:.4f}")
    print(f"  precision: {best['precision_at_ODS']:.4f}")
    print(f"  recall: {best['recall_at_ODS']:.4f}")
    print(f"  OIS_strict: {best['OIS_strict']:.4f}")
    print(f"  AP: {best['AP']:.4f}")


if __name__ == "__main__":
    main()
