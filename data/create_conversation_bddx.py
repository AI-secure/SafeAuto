"""
Utility script to convert raw BDD-X JSON files into a conversation-style
dataset for training and evaluation.

This script processes BDD-X data, optionally augmenting it with retrieved
examples (RAG), and formats vehicle control signals into a consistent
numeric string representation.

Features:
* **Retrieval-Augmented Generation (RAG):** Can prepend retrieved
    driving scenarios to the context for in-context learning.
* **Configurable Numeric Formatting:** The string format for numeric values
    (like speed and course) is fully configurable via CLI arguments.

Example Usage:
# Default processing
python create_conversation_bddx.py

# Enable RAG with 2 examples and custom numeric formatting
# (2 digits for integer part, 3 for fractional part)
python create_conversation_bddx.py --rag --topk 2 --decimal 2 --fractional 3
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from tqdm import tqdm

# Input data paths
TRAIN_INFO_ROOT = Path("data/BDDX_Processed/info")
TEST_INFO_ROOT = Path("data/BDDX_Test/info")

# Output directory
OUTPUT_DIR = Path("data/conversation/bddx")

# Videos that are found to be corrupted or repeated
BROKEN_VIDEOS = {
    "10af6ba8-167c93c2_11906.mp4",
    "1036b362-92000280_11763.mp4",
    "22f325ef-1cafcfb4_21220.mp4",
    "231bdb08-2859bb11_21308.mp4",
    "23102cfb-b94e3ca0_21269.mp4",
    "2332ba2c-628e84e7_21390.mp4",
    "2332ba2c-628e84e7_21391.mp4",
    "22b4f5e6-a1859104_21148.mp4",
    "22b4f5e6-a1859104_21147.mp4",
    "231bdb08-2859bb11_21307.mp4",
    "22f325ef-1cafcfb4_21219.mp4"
}

# Conversation Constant Questions
# Though there are some grammatical errors, we retain the same question from the RAG-Driver repo for consistency.
QUESTION_1 = '''What is the action of ego car?'''
QUESTION_2 = '''Why does the ego car doing this?'''
QUESTION_3 = '''Predict the control signal for next frame.'''

def _format_float(
    value: float, *, decimal_digits: int, fractional_digits: int
) -> str:
    """Formats a float into a zero-padded string."""
    sign = "-" if value < 0 else ""
    formatted = f"{abs(value):.{fractional_digits}f}"
    integer_part, fractional_part = formatted.split(".")
    padded_integer = integer_part.zfill(decimal_digits)
    return f"{sign}{padded_integer}.{fractional_part}"

def format_value_string(
    data: np.ndarray | List[float] | float,
    *,
    decimal_digits: int,
    fractional_digits: int,
) -> str:
    """Wraps a single float or a list of floats into a formatted string."""
    if isinstance(data, (np.ndarray, list)):
        if len(data) == 1:
            return _format_float(
                data[0],
                decimal_digits=decimal_digits,
                fractional_digits=fractional_digits,
            )
        formatted_list = ", ".join(
            _format_float(
                v, decimal_digits=decimal_digits, fractional_digits=fractional_digits
            )
            for v in data
        )
        return f"[{formatted_list}]"
    return _format_float(
        data, decimal_digits=decimal_digits, fractional_digits=fractional_digits
    )

def sample_and_round(
    values: List[float], n: int, *, fractional_digits: int, half_course: bool = False
) -> np.ndarray:
    """Performs uniform stride sampling and rounds the results."""
    sample = np.array(values[::4][:n])
    if half_course:
        # Halve course values (originally [-180, 180]) to better fit
        # into a two-digit integer representation (e.g., 180 -> 90.0).
        sample /= 2
    return np.round(sample, fractional_digits)

def build_car_context(
    car_info: Dict[str, List[float]], *, decimal_digits: int, fractional_digits: int
) -> Tuple[str, str]:
    """Builds the context string and the target control signal string."""
    speed = sample_and_round(car_info["speed"], 8, fractional_digits=fractional_digits)
    course = sample_and_round(car_info["course"], 8, fractional_digits=fractional_digits, half_course=True)
    curvature = sample_and_round(car_info["curvature"], 8, fractional_digits=fractional_digits)
    acceleration = sample_and_round(car_info["acceleration"], 8, fractional_digits=fractional_digits)

    target_cs = (
        f"Speed: {format_value_string(speed[-1], decimal_digits=decimal_digits, fractional_digits=fractional_digits)} "
        f"Course: {format_value_string(course[-1], decimal_digits=decimal_digits, fractional_digits=fractional_digits)}"
    )

    context = (
        "The current video records driving scenario: <video>\n"
        f" Control Signal until current Frame Sequence is: Speed: {format_value_string(speed[:-1], decimal_digits=decimal_digits, fractional_digits=fractional_digits)}\n"
        f" Curvature: {format_value_string(curvature[:-1], decimal_digits=decimal_digits, fractional_digits=fractional_digits)}\n"
        f" Acceleration: {format_value_string(acceleration[:-1], decimal_digits=decimal_digits, fractional_digits=fractional_digits)}\n"
        f" Course: {format_value_string(course[:-1], decimal_digits=decimal_digits, fractional_digits=fractional_digits)}"
    )
    return context, target_cs

def build_conversation(
    context: str,
    action: str,
    justification: str,
    final_cs: str,
    *,
    icl_context: str = "",
) -> List[Dict[str, str]]:
    """Constructs the final list of conversation turns."""
    initial_prompt = f"{icl_context}\n{context}\n{QUESTION_1}" if icl_context else f"{context}\n{QUESTION_1}"
    return [
        {"from": "human", "value": initial_prompt.strip()},
        {"from": "gpt", "value": action},
        {"from": "human", "value": QUESTION_2},
        {"from": "gpt", "value": justification},
        {"from": "human", "value": QUESTION_3},
        {"from": "gpt", "value": final_cs},
    ]

def iter_info_files(info_root: Path):
    """Yields all JSON info files recursively from a root directory."""
    return info_root.rglob("*.json")


def generate_rag_info_mapping(
    train_info_root: Path, config: argparse.Namespace
) -> Dict[str, str]:
    """
    Generates the RAG info mapping on-the-fly from the training data.
    This map contains a formatted conversation string for each training video ID.
    """
    print("Generating RAG info mapping from training data...")
    id_info_match = {}
    file_iterator = iter_info_files(train_info_root)

    for info_path in tqdm(list(file_iterator), desc="Generating RAG info map"):
        with info_path.open("r") as fp:
            info = json.load(fp)

        if Path(info["video"]).name in BROKEN_VIDEOS:
            continue

        context, final_cs = build_car_context(
            info["car_info"],
            decimal_digits=config.decimal,
            fractional_digits=config.fractional,
        )
        conversation_turns = build_conversation(
            context,
            info["comment"][0]["action"],
            info["comment"][1]["justification"],
            final_cs,
        )

        # Format the conversation into a single string: "Human: ...\nAssistant: ..."
        convo_text_parts = []
        for turn in conversation_turns:
            role = "Human" if turn["from"] == "human" else "Assistant"
            convo_text_parts.append(f"{role}: {turn['value']}")
            id_info_match[info["video"].split('/')[-1]] = "\n".join(convo_text_parts)

    return id_info_match

def process_split(
    split: str,
    info_root: Path,
    config: argparse.Namespace,
    match_dict: Dict[str, List[str]],
    id_info_match: Dict[str, str],
) -> List[Dict[str, Any]]:
    """Processes a single data split (e.g., "train" or "test")."""
    processed_entries: List[Dict[str, Any]] = []
    file_iterator = iter_info_files(info_root)

    for info_path in tqdm(list(file_iterator), desc=f"Processing {split} split"):
        with info_path.open("r") as fp:
            info = json.load(fp)

        video_path = Path(info["video"])
        video_filename = video_path.name
        if video_filename in BROKEN_VIDEOS:
            continue
            
        context, final_cs = build_car_context(
            info["car_info"],
            decimal_digits=config.decimal,
            fractional_digits=config.fractional,
        )

        icl_context = ""
        rag_video_paths = []
        if config.rag and video_filename in match_dict:
            matched_ids = match_dict[video_filename][: config.topk]
            assert len(matched_ids) == config.topk
            if matched_ids:
                rag_video_paths = [
                    TRAIN_INFO_ROOT.parent / "videos" / Path(id).name for id in matched_ids
                ]
                samples = [id_info_match.get(sid, "") for sid in matched_ids]
                icl_lines = [
                    "Here are some historical driving experiences and corresponding question-and-answer examples:\n"
                ]
                for i, sample in enumerate(filter(None, samples)):
                    icl_lines.append(f"Experience {i}:\n{sample}\n")
                icl_context = "\n".join(icl_lines)

        conversation = build_conversation(
            context,
            info["comment"][0]["action"],
            info["comment"][1]["justification"],
            final_cs,
            icl_context=icl_context,
        )

        main_video_path = info_root.parent / "videos" / video_filename
        all_videos = [str(p) for p in rag_video_paths] + [str(main_video_path)]

        processed_entries.append(
            {"id": str(info["id"]), "video": all_videos, "conversations": conversation}
        )

    return processed_entries

def save_split(
    split: str, data: List[Dict[str, Any]], config: argparse.Namespace
) -> None:
    """Saves the processed data to a JSON file."""
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
    split_name = "train" if split == "train" else "eval"
    rag_suffix = f"_rag_top{config.topk}" if config.rag else ""
    filename = f"conversation_bddx_{split_name}{rag_suffix}.json"
    output_path = OUTPUT_DIR / filename

    print(f"Saving {len(data)} entries to {output_path}...")
    with output_path.open("w", encoding="utf-8") as fp:
        json.dump(data, fp, ensure_ascii=False, indent=2)


def parse_arguments() -> argparse.Namespace:
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Preprocess BDD-X info JSONs into conversation datasets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--rag", action="store_true", help="Enable retrieval-augmented context.")
    parser.add_argument("--topk", type=int, default=2, help="Top K retrieval matches to use when RAG is enabled.")
    parser.add_argument("--decimal", type=int, default=2, help="Minimum width of the integer part (zero-padded).")
    parser.add_argument("--fractional", type=int, default=3, help="Number of digits after the decimal point.")
    parser.add_argument("--rag_file", type=str, default="retrieval/bddx_rag_top2.json", help="Path to the RAG match data file.")
    return parser.parse_args()

def main() -> None:
    """Main execution function."""
    config = parse_arguments()

    config.topk = max(1, config.topk)
    config.decimal = max(1, config.decimal)
    config.fractional = max(0, config.fractional)
    
    print("Starting dataset creation with the following configuration:")
    print(f"  - RAG enabled: {config.rag}")
    if config.rag:
        print(f"  - Top-K examples: {config.topk}")
    print(f"  - Numeric format: {config.decimal} integer, {config.fractional} fractional digits")
    print("-" * 30)

    # If RAG is enabled, generate the info map and load the match dictionary.
    id_info_match = {}
    match_dict = {}
    if config.rag:
        id_info_match = generate_rag_info_mapping(TRAIN_INFO_ROOT, config)
        print(f"Loading RAG match data from {config.rag_file}...")
        with open(config.rag_file, "r") as f:
            match_dict = json.load(f)

    # Process both training and testing splits
    for split, root_path in [("train", TRAIN_INFO_ROOT), ("test", TEST_INFO_ROOT)]:
        processed_data = process_split(split, root_path, config, match_dict, id_info_match)
        if processed_data:
            save_split(split, processed_data, config)
        else:
            print(f"No data processed for the {split} split. Skipping save.")

    print("-" * 30)
    print("Processing complete.")

if __name__ == "__main__":
    main()