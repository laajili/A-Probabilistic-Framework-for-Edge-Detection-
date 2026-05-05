"""Visualize edge probability maps under several binarization thresholds."""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.edge_detection import detect_edges
from src.io_utils import load_gray01
from src.visualization import visualize_thresholds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visual threshold analysis for probabilistic edge maps.")
    parser.add_argument("--img_dir", type=str, required=True)
    parser.add_argument("--lambda_param", type=float, default=5.0)
    parser.add_argument("--gaussian_sigma", type=float, default=0.6)
    parser.add_argument("--max_images", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_paths = sorted(glob.glob(str(Path(args.img_dir) / "*.*")))[: args.max_images]
    for image_path in image_paths:
        image = load_gray01(image_path)
        score = detect_edges(image, lambda_param=args.lambda_param, gaussian_sigma=args.gaussian_sigma)
        visualize_thresholds(image, score, title=Path(image_path).name)


if __name__ == "__main__":
    main()
