"""Benchmark runtime and memory usage of the probabilistic edge detector."""

from __future__ import annotations

import argparse
import ctypes
from ctypes import wintypes
import glob
import os
import platform
import statistics
import sys
import time
import tracemalloc
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.edge_detection import detect_edges
from src.io_utils import load_gray01


def process_rss_mb() -> Optional[float]:
    """Return current resident memory in MB when available."""
    if platform.system() == "Windows":
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)

        class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.c_ulong),
                ("PageFaultCount", ctypes.c_ulong),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = PROCESS_MEMORY_COUNTERS()
        counters.cb = ctypes.sizeof(counters)
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(PROCESS_MEMORY_COUNTERS),
            wintypes.DWORD,
        ]
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL

        handle = kernel32.GetCurrentProcess()
        ok = psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb)
        if ok:
            return counters.WorkingSetSize / (1024 * 1024)
        return None

    try:
        import resource

        rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if platform.system() == "Darwin":
            return rss_kb / (1024 * 1024)
        return rss_kb / 1024
    except Exception:
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure inference runtime and memory usage.")
    parser.add_argument("--img_dir", type=str, required=True, help="Folder containing input images.")
    parser.add_argument("--out_dir", type=str, default="results/metrics/runtime_memory", help="Output folder.")
    parser.add_argument("--lambda_param", type=float, default=2.0, help="Sensitivity parameter lambda.")
    parser.add_argument("--gaussian_sigma", type=float, default=0.6, help="Optional Gaussian smoothing sigma.")
    parser.add_argument("--repeats", type=int, default=5, help="Number of timed repetitions per image.")
    parser.add_argument("--warmup", type=int, default=1, help="Number of untimed warmup repetitions per image.")
    parser.add_argument("--max_images", type=int, default=None, help="Optional maximum number of images.")
    return parser.parse_args()


def benchmark_image(
    image: np.ndarray,
    lambda_param: float,
    gaussian_sigma: float,
    repeats: int,
    warmup: int,
) -> Dict[str, float]:
    """Benchmark one already-loaded image."""
    for _ in range(warmup):
        detect_edges(image, lambda_param=lambda_param, gaussian_sigma=gaussian_sigma)

    times_ms: List[float] = []
    peaks_mb: List[float] = []
    rss_deltas_mb: List[float] = []

    for _ in range(repeats):
        rss_before = process_rss_mb()
        tracemalloc.start()
        start = time.perf_counter()
        score = detect_edges(image, lambda_param=lambda_param, gaussian_sigma=gaussian_sigma)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        _, peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        rss_after = process_rss_mb()

        times_ms.append(elapsed_ms)
        peaks_mb.append(peak_bytes / (1024 * 1024))
        if rss_before is not None and rss_after is not None:
            rss_deltas_mb.append(max(0.0, rss_after - rss_before))

        if score.shape != image.shape:
            raise RuntimeError("Unexpected output shape from detect_edges.")

    pixels = int(image.shape[0] * image.shape[1])
    mean_time = statistics.mean(times_ms)
    median_time = statistics.median(times_ms)
    std_time = statistics.pstdev(times_ms) if len(times_ms) > 1 else 0.0
    mean_peak = statistics.mean(peaks_mb)
    mean_rss_delta = statistics.mean(rss_deltas_mb) if rss_deltas_mb else float("nan")

    return {
        "height": int(image.shape[0]),
        "width": int(image.shape[1]),
        "pixels": pixels,
        "mean_time_ms": mean_time,
        "median_time_ms": median_time,
        "std_time_ms": std_time,
        "min_time_ms": min(times_ms),
        "max_time_ms": max(times_ms),
        "fps": 1000.0 / mean_time if mean_time > 0 else float("inf"),
        "megapixels_per_second": (pixels / 1_000_000.0) / (mean_time / 1000.0) if mean_time > 0 else float("inf"),
        "tracemalloc_peak_mb": mean_peak,
        "rss_delta_mb": mean_rss_delta,
    }


def main() -> None:
    args = parse_args()
    if args.repeats < 1:
        raise ValueError("--repeats must be at least 1.")
    if args.warmup < 0:
        raise ValueError("--warmup must be non-negative.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(glob.glob(os.path.join(args.img_dir, "*.*")))
    if args.max_images is not None:
        image_paths = image_paths[: args.max_images]
    if not image_paths:
        raise FileNotFoundError(f"No images found in {args.img_dir}")

    rows = []
    for image_path in tqdm(image_paths, desc="Benchmark"):
        image = load_gray01(image_path)
        row = benchmark_image(
            image=image,
            lambda_param=args.lambda_param,
            gaussian_sigma=args.gaussian_sigma,
            repeats=args.repeats,
            warmup=args.warmup,
        )
        row["image"] = Path(image_path).name
        rows.append(row)

    per_image = pd.DataFrame(rows)
    summary = pd.DataFrame(
        [
            {
                "num_images": len(per_image),
                "lambda_param": args.lambda_param,
                "gaussian_sigma": args.gaussian_sigma,
                "repeats": args.repeats,
                "warmup": args.warmup,
                "python_version": platform.python_version(),
                "platform": platform.platform(),
                "processor": platform.processor(),
                "mean_time_ms": per_image["mean_time_ms"].mean(),
                "median_time_ms": per_image["median_time_ms"].median(),
                "std_time_ms": per_image["mean_time_ms"].std(ddof=0),
                "mean_fps": per_image["fps"].mean(),
                "mean_megapixels_per_second": per_image["megapixels_per_second"].mean(),
                "mean_tracemalloc_peak_mb": per_image["tracemalloc_peak_mb"].mean(),
                "max_tracemalloc_peak_mb": per_image["tracemalloc_peak_mb"].max(),
                "mean_rss_delta_mb": per_image["rss_delta_mb"].mean(),
            }
        ]
    )

    per_image.to_csv(out_dir / "per_image_runtime_memory.csv", index=False)
    summary.to_csv(out_dir / "summary_runtime_memory.csv", index=False)

    item = summary.iloc[0]
    print("Runtime and memory summary:")
    print(f"  images: {int(item['num_images'])}")
    print(f"  lambda_param: {item['lambda_param']:.6g}")
    print(f"  gaussian_sigma: {item['gaussian_sigma']:.6g}")
    print(f"  mean_time_ms: {item['mean_time_ms']:.4f}")
    print(f"  median_time_ms: {item['median_time_ms']:.4f}")
    print(f"  mean_fps: {item['mean_fps']:.2f}")
    print(f"  mean_tracemalloc_peak_mb: {item['mean_tracemalloc_peak_mb']:.4f}")
    print(f"  max_tracemalloc_peak_mb: {item['max_tracemalloc_peak_mb']:.4f}")
    print(f"  mean_rss_delta_mb: {item['mean_rss_delta_mb']:.4f}")


if __name__ == "__main__":
    main()
