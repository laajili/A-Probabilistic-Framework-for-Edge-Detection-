"""BSDS-style relaxed precision/recall evaluation utilities."""

from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt


def bsds_tolerance_px(height: int, width: int, k: float = 0.0075) -> int:
    """Compute the BSDS-like pixel tolerance from image size."""
    return int(np.rint(k * np.sqrt(height * height + width * width)))


def relaxed_confusion(pred: np.ndarray, gt: np.ndarray, tol: int) -> Tuple[int, int, int, int]:
    """Compute relaxed TP, FP, FN, TN using distance-transform tolerance."""
    pred = pred.astype(bool)
    gt = gt.astype(bool)

    dt_gt = distance_transform_edt(~gt)
    dt_pred = distance_transform_edt(~pred)

    match_pred = pred & (dt_gt <= tol)
    match_gt = gt & (dt_pred <= tol)

    tp = int(match_pred.sum())
    fp = int(pred.sum() - match_pred.sum())
    fn = int(gt.sum() - match_gt.sum())
    tn = int(pred.size - (tp + fp + fn))

    return tp, fp, fn, tn


def precision_recall_f(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    """Return precision, recall and F1-score."""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def ois_strict_for_image(
    score01: np.ndarray,
    gt_list: List[np.ndarray],
    thresholds: Iterable[float],
    tol: int,
) -> Tuple[float, Dict[str, float]]:
    """Compute strict OIS for one image by selecting the best label and threshold."""
    best_f_across_labels = 0.0
    best_info = {"label_idx": -1, "threshold": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}

    for label_idx, gt in enumerate(gt_list):
        for threshold in thresholds:
            pred = score01 >= threshold
            tp, fp, fn, _ = relaxed_confusion(pred, gt, tol)
            precision, recall, f1 = precision_recall_f(tp, fp, fn)

            if f1 > best_f_across_labels:
                best_f_across_labels = f1
                best_info = {
                    "label_idx": label_idx,
                    "threshold": float(threshold),
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                }

    return best_f_across_labels, best_info


def compute_ods_ap(
    scores: Dict[str, np.ndarray],
    ground_truths: Dict[str, List[np.ndarray]],
    tolerances: Dict[str, int],
    thresholds: Iterable[float],
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Compute micro-averaged ODS and AP over a dataset."""
    rows = []

    for threshold in thresholds:
        tp_sum = fp_sum = fn_sum = 0

        for image_path, score01 in scores.items():
            if image_path not in ground_truths:
                continue

            pred = score01 >= threshold
            tol = tolerances[image_path]

            best_tp = best_fp = best_fn = 0
            best_f = -1.0

            for gt in ground_truths[image_path]:
                tp, fp, fn, _ = relaxed_confusion(pred, gt, tol)
                precision, recall, f1 = precision_recall_f(tp, fp, fn)
                if f1 > best_f:
                    best_f = f1
                    best_tp, best_fp, best_fn = tp, fp, fn

            tp_sum += best_tp
            fp_sum += best_fp
            fn_sum += best_fn

        precision = tp_sum / (tp_sum + fp_sum) if (tp_sum + fp_sum) > 0 else 0.0
        recall = tp_sum / (tp_sum + fn_sum) if (tp_sum + fn_sum) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        rows.append({"threshold": float(threshold), "precision": precision, "recall": recall, "f1": f1})

    df = pd.DataFrame(rows)
    best = df.iloc[df["f1"].values.argmax()]
    curve = df.sort_values("recall")
    ap = float(np.trapz(curve["precision"].values, curve["recall"].values))

    summary = {
        "ODS": float(best["f1"]),
        "threshold_star": float(best["threshold"]),
        "precision_at_ODS": float(best["precision"]),
        "recall_at_ODS": float(best["recall"]),
        "AP": ap,
    }
    return df, summary


def compute_ods_ois_ap(
    scores: Dict[str, np.ndarray],
    ground_truths: Dict[str, List[np.ndarray]],
    tolerances: Dict[str, int],
    thresholds: Iterable[float],
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Compute ODS, OIS and AP while reusing distance transforms.

    This is equivalent to the relaxed BSDS-style protocol used by the helper
    functions above, but avoids recomputing ground-truth distance maps for
    every threshold and combines the ODS/OIS passes.
    """
    thresholds = list(thresholds)
    rows = []
    ois_best = {image_path: 0.0 for image_path in scores if image_path in ground_truths}

    gt_cache = {}
    for image_path, gt_list in ground_truths.items():
        gt_cache[image_path] = [
            {
                "gt": gt.astype(bool),
                "gt_sum": int(gt.astype(bool).sum()),
                "dt_gt": distance_transform_edt(~gt.astype(bool)),
            }
            for gt in gt_list
        ]

    for threshold in thresholds:
        tp_sum = fp_sum = fn_sum = 0

        for image_path, score01 in scores.items():
            if image_path not in gt_cache:
                continue

            pred = score01 >= threshold
            pred_sum = int(pred.sum())
            dt_pred = distance_transform_edt(~pred)
            tol = tolerances[image_path]

            best_tp = best_fp = best_fn = 0
            best_f = -1.0

            for item in gt_cache[image_path]:
                gt = item["gt"]
                tp = int((pred & (item["dt_gt"] <= tol)).sum())
                fp = pred_sum - tp
                matched_gt = int((gt & (dt_pred <= tol)).sum())
                fn = item["gt_sum"] - matched_gt

                precision, recall, f1 = precision_recall_f(tp, fp, fn)
                if f1 > best_f:
                    best_f = f1
                    best_tp, best_fp, best_fn = tp, fp, fn

            tp_sum += best_tp
            fp_sum += best_fp
            fn_sum += best_fn
            if best_f > ois_best[image_path]:
                ois_best[image_path] = best_f

        precision = tp_sum / (tp_sum + fp_sum) if (tp_sum + fp_sum) > 0 else 0.0
        recall = tp_sum / (tp_sum + fn_sum) if (tp_sum + fn_sum) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        rows.append({"threshold": float(threshold), "precision": precision, "recall": recall, "f1": f1})

    df = pd.DataFrame(rows)
    best = df.iloc[df["f1"].values.argmax()]
    curve = df.sort_values("recall")
    ap = float(np.trapz(curve["precision"].values, curve["recall"].values))

    summary = {
        "ODS": float(best["f1"]),
        "threshold_star": float(best["threshold"]),
        "precision_at_ODS": float(best["precision"]),
        "recall_at_ODS": float(best["recall"]),
        "AP": ap,
        "OIS_strict": float(np.mean(list(ois_best.values()))) if ois_best else 0.0,
    }
    return df, summary
