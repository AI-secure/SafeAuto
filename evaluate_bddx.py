#!/usr/bin/env python3
"""
Evaluate BDD-X captioning and control-signal predictions.

Usage examples
--------------
# Run both evaluations (caption + signal) — default behaviour
python eval_bddx.py --path xx

# Only caption evaluation
python eval_bddx.py --path xx --caption

# Only signal evaluation
python eval_bddx.py --path xx --signal
"""
import argparse
import json
import os
import re
from pathlib import Path

import numpy as np
from sklearn.metrics import mean_squared_error

from evalcap.utils_caption_evaluate import evaluate_on_coco_caption

def evaluate_captions(data_dir: Path) -> None:
    """Compute COCO-style caption metrics for Action / Justification splits."""
    caps = ["action", "justification"]
    gt_files = [Path("evalcap/BDDX_gt") / f"BDDX_Test_coco_{c}.json" for c in caps]
    pred_files = [data_dir / f"BDDX_Test_pred_{c}.json" for c in caps]

    # Try Action first, fall back to Justification if missing
    for pred, gt in zip(pred_files, gt_files):
        if pred.exists():
            print(f"→ Caption split: {pred.name}")
            result = evaluate_on_coco_caption(pred, gt)
            print(result)  # `result` is already nicely formatted by utils
            break
    else:
        print("[!] No caption prediction file found.")

SIGMA_THRESHOLDS = [0.1, 0.5, 1, 5, 10]

def extract_speed_course(text: str):
    """Parse `[speed]` and `[course]` lists from a prediction string."""
    speed_pat = re.compile(r"Speed:\s*\[?([+\-0-9., ]+)\]?")
    course_pat = re.compile(r"Course:\s*\[?([+\-0-9., ]+)\]?")

    try:
        speed_list = speed_pat.search(text).group(1)
        course_list = course_pat.search(text).group(1)
        speed = eval(speed_list)
        course = eval(course_list)
        return speed, course
    except AttributeError:
        return None, None  # Malformed string


def calculate_metrics(gt, pred, name: str):
    """Print RMSE and accuracy within multiple σ-thresholds."""
    gt = np.array(gt, dtype=np.float32)
    pred = np.array(pred, dtype=np.float32)
    valid = ~np.isnan(pred)

    rmse = np.sqrt(mean_squared_error(gt[valid], pred[valid]))
    print(f"{name} RMSE: {rmse:.2f}")

    for σ in SIGMA_THRESHOLDS:
        acc = np.mean(np.abs(gt[valid] - pred[valid]) < σ) * 100
        print(f"  • within {σ:>4}: {acc:5.2f} %")


def evaluate_signals(data_dir: Path) -> None:
    """Compute RMSE + threshold accuracy for Speed & Course signals."""
    # Load ground-truth conversations
    gt_path = Path("data/conversation/bddx/conversation_bddx_eval.json")
    with gt_path.open() as f:
        conv = json.load(f)

    # Human references (index 5 token in each conversation)
    refs = [item["conversations"][5]["value"].strip() for item in conv]
    gt_speed_course = [extract_speed_course(r) for r in refs]

    # Model predictions
    pred_path = data_dir / "BDDX_Test_pred_control_signal.json"
    if not pred_path.exists():
        print("[!] Signal prediction file not found.")
        return

    with pred_path.open() as f:
        preds = [c["caption"] for c in json.load(f)]
    pred_speed_course = [extract_speed_course(p) for p in preds]

    # Unzip and evaluate
    gt_speed, gt_course = zip(*gt_speed_course)
    pr_speed, pr_course = zip(*pred_speed_course)

    print("→ Control-signal metrics")
    calculate_metrics(gt_speed, pr_speed,  "Speed")
    calculate_metrics(np.array(gt_course) * 2, np.array(pr_course) * 2, "Course")  # Undo /2 pre-processing


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate BDD-X predictions for captions and/or control signals"
    )
    parser.add_argument(
        "--path", required=True,
        help="Directory containing prediction JSON files (relative to ./results/)"
    )
    parser.add_argument("--caption", action="store_true", help="Run caption metrics")
    parser.add_argument("--signal",  action="store_true", help="Run control-signal metrics")

    args = parser.parse_args()
    data_dir = Path("results") / args.path

    if not data_dir.exists():
        raise FileNotFoundError(f"Prediction directory not found: {data_dir}")

    # Default: run both if neither flag provided
    if not (args.caption or args.signal):
        args.caption = args.signal = True

    if args.caption:
        evaluate_captions(data_dir)

    if args.signal:
        evaluate_signals(data_dir)


if __name__ == "__main__":
    main()
