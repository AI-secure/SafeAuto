"""
Utility script to convert raw DriveLM JSON files into a conversation-style
dataset for training and evaluation.

This script processes DriveLM data, optionally augmenting it with retrieved
examples (RAG), and formats vehicle control signals into a consistent
numeric string representation for multi-view autonomous driving scenarios.

Features:
* **Retrieval-Augmented Generation (RAG):** Can prepend retrieved
    driving scenarios to the context for in-context learning.
* **Configurable Numeric Formatting:** The string format for numeric values
    (like position, speed, orientation) is fully configurable via CLI arguments.
* **Multi-view Image Support:** Handles 6-camera views from autonomous vehicles.

Example Usage:
# Default processing
python create_conversation_drivelm.py

# Enable RAG with 1 example and custom numeric formatting
# (2 digits for integer part, 2 for fractional part)
python create_conversation_drivelm.py --rag --topk 1 --decimal 2 --fractional 2 --history_len 6 --predict_len 6
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

from tqdm import tqdm

# Input data paths
TRAIN_INFO_ROOT = Path("data/DriveLM_train/info")
VAL_INFO_ROOT = Path("data/DriveLM_val/info")

# Output directory
OUTPUT_DIR = Path("data/conversation/drivelm")

def format_pos(num: float, *, decimal_digits: int, fractional_digits: int) -> str:
    """Formats a position value into a zero-padded string with consistent width."""
    rounded_num = round(num, fractional_digits)
    
    if rounded_num == 0:
        return f"{0:0{decimal_digits + fractional_digits + 1}.{fractional_digits}f}"
    
    if rounded_num < 0:
        return f"-{-rounded_num:0{decimal_digits + fractional_digits + 1}.{fractional_digits}f}"
    else:
        return f"{rounded_num:0{decimal_digits + fractional_digits + 1}.{fractional_digits}f}"


def round_values(data: List[List[float]], fractional_digits: int) -> List[List[float]]:
    """Rounds values to specified decimal places."""
    return [[round(x, fractional_digits) for x in item[:2]] for item in data]


def get_car_info(
    car_info: Dict[str, List[List[float]]],
    future_info: Dict[str, List[List[float]]],
    *,
    history_len: int,
    predict_len: int,
    decimal_digits: int,
    fractional_digits: int,
) -> Tuple[str, str, int]:
    """
    Extracts and formats car control information from historical and future data.
    
    Returns:
        Tuple of (context_string, final_control_signal, prediction_length)
    """
    # Current information (latest from future)
    cur_position = round_values([future_info['positions'][0]], fractional_digits)[0]
    cur_speed = round_values([future_info['speeds'][0]], fractional_digits)[0]
    cur_acceleration = round_values([future_info['accelerations'][0]], fractional_digits)[0]
    cur_orientation = round_values([future_info['orientations'][0]], fractional_digits)[0]

    # Historical and future data
    zero_position = cur_position
    his_positions = round_values(car_info['positions'][-history_len:], fractional_digits)
    fut_positions = round_values(future_info['positions'][1:predict_len+1], fractional_digits)
    his_speeds = round_values(car_info['speeds'][-history_len:], fractional_digits)
    his_accelerations = round_values(car_info['accelerations'][-history_len:], fractional_digits)
    his_orientations = round_values(car_info['orientations'][-history_len:], fractional_digits)

    # Append current values to historical data
    his_positions.append(cur_position)
    his_speeds.append(cur_speed)
    his_accelerations.append(cur_acceleration)
    his_orientations.append(cur_orientation)

    # Format relative positions and other metrics
    his_pos_rela = [
        [
            format_pos(his_positions[i][0] - zero_position[0], decimal_digits=decimal_digits, fractional_digits=fractional_digits),
            format_pos(his_positions[i][1] - zero_position[1], decimal_digits=decimal_digits, fractional_digits=fractional_digits)
        ]
        for i in range(len(his_positions))
    ]
    
    fur_pos_rela = [
        [
            format_pos(fut_positions[i][0] - zero_position[0], decimal_digits=decimal_digits, fractional_digits=fractional_digits),
            format_pos(fut_positions[i][1] - zero_position[1], decimal_digits=decimal_digits, fractional_digits=fractional_digits)
        ]
        for i in range(len(fut_positions))
    ]

    his_speeds = [[format_pos(x, decimal_digits=decimal_digits, fractional_digits=fractional_digits), format_pos(y, decimal_digits=decimal_digits, fractional_digits=fractional_digits)] for x, y in his_speeds]
    his_accelerations = [[format_pos(x, decimal_digits=decimal_digits, fractional_digits=fractional_digits), format_pos(y, decimal_digits=decimal_digits, fractional_digits=fractional_digits)] for x, y in his_accelerations]
    his_orientations = [[format_pos(x, decimal_digits=decimal_digits, fractional_digits=fractional_digits), format_pos(y, decimal_digits=decimal_digits, fractional_digits=fractional_digits)] for x, y in his_orientations]

    # Create formatted strings
    his_pos_str = ", ".join([f"({x}, {y})" for x, y in his_pos_rela])
    fur_pos_str = ", ".join([f"({x}, {y})" for x, y in fur_pos_rela])
    his_speeds_str = ", ".join([f"({x}, {y})" for x, y in his_speeds])
    his_accelerations_str = ", ".join([f"({x}, {y})" for x, y in his_accelerations])
    his_orientations_str = ", ".join([f"({x}, {y})" for x, y in his_orientations])

    cs_context = (
        f"The control signals until the current frame are: Position: [{his_pos_str}]\n"
        f"Speed: [{his_speeds_str}]\nOrientation: [{his_orientations_str}]"
    )
    final_cs = f'Position: [{fur_pos_str}]'
    pred_len = len(fut_positions)

    return cs_context, final_cs, pred_len


def get_qa_pair(qa_data: Dict[str, List[Dict[str, str]]]) -> List[Tuple[str, str]]:
    """Extracts question-answer pairs from QA data structure."""
    qa_pairs = []
    
    # Extract safe actions questions
    for q_type in qa_data:
        for q in qa_data[q_type]:
            if 'what are safe actions' in q['Q'].lower():
                qa_pairs.append((q['Q'], q['A']))
    
    # Add behavior questions
    behavior_pairs = [(q['Q'], q['A']) for q in qa_data.get('behavior', [])]
    return qa_pairs + behavior_pairs


def load_rag_data(
    rag_file: str,
    retrieval_pool_file: str = "data/conversation/drivelm/conversation_drivelm_train.json", # we use training data as the pool
) -> Dict[str, Dict[str, Any]]:
    """
    Convert the compact mapping produced by `new_rag.json`
    ──────────────────────────────────────────────────────────
        key → [id1, id2, ...]
    into the structure expected by `process_split`
    ──────────────────────────────────────────────────────────
        key → {"image": [[6-cam jpgs], …],
               "conversations": ["full_conv_str_1", …]}
    """
    with open(rag_file, "r") as f:
        rag_map: Dict[str, List[str]] = json.load(f)

    with open(retrieval_pool_file, "r") as f:
        pool_data = json.load(f)

    id2entry: Dict[str, Dict[str, Any]] = {}
    for item in pool_data:
        id2entry[item["id"]] = item

    expanded: Dict[str, Dict[str, Any]] = {}
    for key, id_list in rag_map.items():
        images, convs = [], []
        for rid in id_list:
            entry = id2entry.get(rid)
            if not entry:          # skip missing ids
                continue

            images.append(entry["image"])

            conv_str = "\n".join(
                f"{'Human' if c['from']=='human' else 'Assistant'}: {c['value']}"
                for c in entry["conversations"]
            )
            convs.append(conv_str)
        expanded[key] = {"image": images, "conversations": convs}
    return expanded

def process_split(
    split: str,
    info_root: Path,
    config: argparse.Namespace,
    match_dict: Dict[str, Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Processes a single data split (e.g., "train" or "val")."""
    processed_entries: List[Dict[str, Any]] = []
    
    if not info_root.exists():
        print(f"Warning: {info_root} does not exist. Skipping {split} split.")
        return processed_entries

    info_files = list(info_root.glob("*.json"))
    for info_file in tqdm(info_files, desc=f"Processing {split} split"):
        with info_file.open('r') as f:
            info = json.load(f)

        scene_token = info['scene_token']
        sample_token = info['sample_token']
        images = info['images']
        car_info = info['car_info']
        future_info = info['future_info']
        qa_data = info['QA']

        # Get car control information
        cs_context, final_cs, pred_len = get_car_info(
            car_info,
            future_info,
            history_len=config.history_len,
            predict_len=config.predict_len,
            decimal_digits=config.decimal,
            fractional_digits=config.fractional,
        )

        # Skip if prediction length doesn't match expected
        if pred_len != config.predict_len:
            continue

        qa_pairs = get_qa_pair(qa_data)
        conversation = []

        # Multi-view camera description
        camera_desc = (
            "CAM_FRONT: <image>, CAM_FRONT_LEFT: <image>, CAM_FRONT_RIGHT: <image>, "
            "CAM_BACK: <image>, CAM_BACK_LEFT: <image>, CAM_BACK_RIGHT: <image>\n"
        )

        # Motion prediction question
        motion_q = 'Predict the position of the ego vehicle.'
        motion_a = final_cs

        # Handle RAG context
        rag_image_context = []
        key = f'{scene_token}_{sample_token}'
        
        if config.rag and match_dict and key in match_dict:
            rag_images = match_dict[key]['image']
            rag_convs = match_dict[key]['conversations']
            rag_word_context = "Here are some historical driving experiences and corresponding question-and-answer examples:\n\n"
            
            for p in range(min(config.topk, len(rag_convs))):
                rag_word_context += f"Experience {p}:\n{rag_convs[p]}\n"
                if p < len(rag_images):
                    rag_image_context.extend(rag_images[p])
                    
            rag_word_context += '\nNow, the current multi-view images record the driving scenario: '
        else:
            rag_word_context = 'Following multi-view images record the driving scenario: '

        # Build conversation
        for i, (q, a) in enumerate(qa_pairs):
            if i == 0:
                conversation.append({'from': 'human', 'value': rag_word_context + camera_desc + q})
            else:
                conversation.append({'from': 'human', 'value': q})
            conversation.append({'from': 'gpt', 'value': a})

        # Add motion prediction
        conversation.append({'from': 'human', 'value': motion_q + ' ' + cs_context})
        conversation.append({'from': 'gpt', 'value': motion_a})

        info_dict = {
            'id': key,
            'image': rag_image_context + images,
            'conversations': conversation
        }
        processed_entries.append(info_dict)

    return processed_entries


def save_split(
    split: str, 
    data: List[Dict[str, Any]], 
    config: argparse.Namespace
) -> None:
    """Saves the processed data to a JSON file."""
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
    
    rag_suffix = f"_rag_top{config.topk}" if config.rag else ""
    filename = f"conversation_drivelm_{split}{rag_suffix}.json"
    output_path = OUTPUT_DIR / filename
    
    print(f"Saving {len(data)} entries to {output_path}...")
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def parse_arguments() -> argparse.Namespace:
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Preprocess DriveLM info JSONs into conversation datasets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # RAG configuration
    parser.add_argument("--rag", action="store_true", help="Enable retrieval-augmented context.")
    parser.add_argument("--topk", type=int, default=1, help="Top K retrieval matches to use when RAG is enabled.")
    parser.add_argument("--rag_file", type=str, default = 'retrieval/drivelm_rag_top2.json', help="Path to the RAG match data file.")
    
    # Numeric formatting
    parser.add_argument("--decimal", type=int, default=2, help="Minimum width of the integer part (zero-padded).")
    parser.add_argument("--fractional", type=int, default=2, help="Number of digits after the decimal point.")
    
    # Data processing configuration
    parser.add_argument("--history_len", type=int, default=6, help="Number of historical frames to include.")
    parser.add_argument("--predict_len", type=int, default=6, help="Number of future frames to predict.")
    
    return parser.parse_args()


def get_info_root(split: str) -> Path:
    """Returns the appropriate info root path for a given split."""
    if split == "train":
        return TRAIN_INFO_ROOT
    elif split == "val":
        return VAL_INFO_ROOT
    else:
        raise ValueError(f"Unknown split: {split}")


def main() -> None:
    """Main execution function."""
    config = parse_arguments()
    
    # Validate and adjust configuration
    config.topk = max(1, config.topk)
    config.decimal = max(1, config.decimal)
    config.fractional = max(0, config.fractional)
    config.history_len = max(1, config.history_len)
    config.predict_len = max(1, config.predict_len)
    
    print("Starting DriveLM dataset creation with the following configuration:")
    print(f"  - RAG enabled: {config.rag}")
    if config.rag:
        print(f"  - Top-K examples: {config.topk}")
    print(f"  - Numeric format: {config.decimal} integer, {config.fractional} fractional digits")
    print(f"  - History length: {config.history_len}")
    print(f"  - Prediction length: {config.predict_len}")
    print("-" * 50)
    
    # Load RAG data if enabled
    match_dict = {}
    if config.rag:
        rag_file = config.rag_file
        if os.path.exists(rag_file):
            match_dict = load_rag_data(rag_file)
        else:
            print(f"Warning: RAG file {rag_file} not found. Proceeding without RAG.")
            config.rag = False
    
    # Process each split
    for split in ['train', 'val']:
        try:
            info_root = get_info_root(split)
            processed_data = process_split(split, info_root, config, match_dict)
            
            if processed_data:
                save_split(split, processed_data, config)
            else:
                print(f"No data processed for the {split} split. Skipping save.")
                
        except ValueError as e:
            print(f"Error processing split '{split}': {e}")
            continue
    
    print("-" * 50)
    print("Processing complete.")


if __name__ == "__main__":
    main()