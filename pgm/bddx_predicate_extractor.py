import os
import re
import cv2
import json
import string
import pickle
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from ultralytics import YOLO
from pgm.config import BDDX
from pgm.gpt_utils import gpt_map_action, gpt_map_cs
from typing import Any, Dict, List, Tuple
from data.create_conversation_bddx import iter_info_files, build_car_context, build_conversation, BROKEN_VIDEOS

PREDICATE_NUM = BDDX().action_num + BDDX().condition_num
CONDITION_PREDICATE_NUM = BDDX().condition_num
ACTION_MAP = {k: BDDX().predicate[k.upper()] for k in BDDX().action_list}
CS_MAP = {k: BDDX().predicate[f"{k.upper()}_CS"] for k in BDDX().velocity_cs_list + BDDX().direction_cs_list}
LLM_ACTION_MAP = {k: BDDX().predicate[f"{k.upper()}_LLM"] for k in BDDX().action_list}
# environment predicate mapping (YOLO/scene classes to predicate index)
EP_MAP = {
    "stop traffic light": BDDX().predicate["SOLID_RED_LIGHT"],
    "warning": BDDX().predicate["SOLID_YELLOW_LIGHT"],
    "warning traffic light": BDDX().predicate["SOLID_YELLOW_LIGHT"],
    "warning left": BDDX().predicate["YELLOW_LEFT_ARROW_LIGHT"],
    "warning left traffic light": BDDX().predicate["YELLOW_LEFT_ARROW_LIGHT"],
    "stop left": BDDX().predicate["RED_LEFT_ARROW_LIGHT"],
    "stop left traffic light": BDDX().predicate["RED_LEFT_ARROW_LIGHT"],
    "merge": BDDX().predicate["MERGING_TRAFFIC_SIGN"],
    "noLeftTurn": BDDX().predicate["NO_LEFT_TURN_SIGN"],
    "noRightTurn": BDDX().predicate["NO_RIGHT_TURN_SIGN"],
    "pedestrianCrossing": BDDX().predicate["PEDESTRIAN_CROSSING_SIGN"],
    "stop": BDDX().predicate["STOP_SIGN"],
    "stopAhead": BDDX().predicate["STOP_SIGN"],
    "yield": BDDX().predicate["RED_YIELD_SIGN"],
    "yieldAhead": BDDX().predicate["RED_YIELD_SIGN"],
    "slow": BDDX().predicate["SLOW_SIGN"],
    "go": BDDX().predicate["SOLID_GREEN_LIGHT"],
    "go forward": BDDX().predicate["SOLID_GREEN_LIGHT"],
    "go forward traffic light": BDDX().predicate["SOLID_GREEN_LIGHT"],
}

# ---------------------- Utility Functions ----------------------
def update_action(action_list: List[str], action: str) -> Optional[str]:
    """
    Return the first action in action_list that appears as a substring in the action string (case-insensitive).
    """
    action_lower = action.lower() if action else ""
    return next((act for act in action_list if act.lower() in action_lower), None)

def update_action_set(action_list: List[str], ori_actions: str) -> List[str]:
    """
    Return all actions in action_list that appear as substrings in ori_actions string (case-insensitive).
    """
    ori_actions_lower = ori_actions.lower() if ori_actions else ""
    return [act for act in action_list if act.lower() in ori_actions_lower]

def cs_extract(id: str, cs_info: List[Dict]) -> Dict:
    """
    Extract control signal info for a given id from cs_info list.
    Returns a dict with keys: Speed, Curvature, Acceleration, Course.
    """
    item = next((item for item in cs_info if item.get('id') == id), None)
    if not item:
        return {}
    text = item['conversations'][0]['value']
    pattern = r'(Speed|Curvature|Acceleration|Course): \[([^\]]+)\]'
    matches = re.findall(pattern, text)
    return {key: [float(x) for x in values.split(', ')] for key, values in matches}

def get_vector_length(inference: bool) -> int:
    """Return the correct vector length for ep or full predicate vector."""
    return CONDITION_PREDICATE_NUM if inference else PREDICATE_NUM

def list2vector(items: List[str], mapping: dict, inference: bool) -> List[int]:
    """Generic function to map a list of names to a predicate vector."""
    vector = [0] * get_vector_length(inference)
    for item in items or []:
        idx = mapping.get(item)
        if idx is not None and idx < len(vector):
            vector[idx] = 1
    return vector

def cs2vector(velocity: Optional[str], direction: Optional[str], inference: bool) -> List[int]:
    """Map velocity and direction predicates to a predicate vector."""
    vector = [0] * get_vector_length(inference)
    for val in (velocity, direction):
        idx = CS_MAP.get(val)
        if idx is not None and idx < len(vector):
            vector[idx] = 1
    return vector

def combine_vectors(*vectors: List[int]) -> List[int]:
    """Element-wise max of multiple vectors."""
    return [max(values) for values in zip(*vectors)]

def segment2vector(segment: Dict, llm_prediction: List[str] = None, inference: bool = False) -> List[int]:
    """
    Convert a segment dict (and optional LLM prediction) to a combined predicate vector.
    If inference=True, only use action/class/cs; else, include llm_prediction.
    """
    action_vec = list2vector(segment.get('gt_action_predicate'), ACTION_MAP, inference)
    class_vec = list2vector(segment.get('classes'), EP_MAP, inference)
    cs_vec = cs2vector(segment.get('velocity_predicate'), segment.get('direction_predicate'), inference)
    if inference:
        return combine_vectors(action_vec, class_vec, cs_vec)
    llm_vec = list2vector(llm_prediction or [], LLM_ACTION_MAP, inference)
    return combine_vectors(action_vec, class_vec, cs_vec, llm_vec)

def vectorize_segments(extract_data: List[Dict], split_name: str, vector_save_dir: str = 'pgm/predicates/bddx/', inference: bool = False, llm_predicates: Dict[str, List[str]] = None) -> List[List[int]]:
    """
    Convert extraction results to predicate vectors and save as a pickle file.
    If inference=True, only use action/class/cs; else, include llm_predictions.
    """
    os.makedirs(vector_save_dir, exist_ok=True)
    vector_save_path = os.path.join(vector_save_dir, f'{split_name}_vectors.pkl')
    if os.path.exists(vector_save_path):
        print(f"[INFO] Vectors already exist, loading from {vector_save_path}")
        with open(vector_save_path, 'rb') as f:
            return pickle.load(f)
    print('[INFO] Begin to convert extraction results to vectors...')
    vectors = []
    for item in tqdm(extract_data, desc='Vectorizing segments'):
        llm_pred = None
        if not inference and llm_predicates is not None:
            llm_pred_dict = {entity["id"]: entity for entity in llm_predicates}
            llm_pred = llm_pred_dict.get(item['id'], {}).get('predicate', [])
        vector = segment2vector(item, llm_pred, inference)
        vectors.append(vector)
    with open(vector_save_path, 'wb') as f:
        pickle.dump(vectors, f)
    return vectors

def process_split(
    split: str,
    info_root: Path,
    decimal: int = 2,
    fractional: int = 3,
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
            decimal_digits=decimal,
            fractional_digits=fractional,
        )
        icl_context = ""
        conversation = build_conversation(
            context,
            info["comment"][0]["action"],
            info["comment"][1]["justification"],
            final_cs,
            icl_context=icl_context,
        )
        main_video_path = info_root.parent / "videos" / video_filename
        processed_entries.append(
            {"id": str(info["id"]), "video": [str(main_video_path)], "conversations": conversation}
        )
    return processed_entries

def create_segment_annotation(
    info_root: Path,
    split_name: str,
    segment_annotation_dir: str = 'data/segment_annotation',
    decimal: int = 2,
    fractional: int = 3,
) -> List[Dict]:
    """Convert conversation JSON to segment annotation with video/time info."""
    os.makedirs(segment_annotation_dir, exist_ok=True)
    out_path = os.path.join(segment_annotation_dir, f'segment_annotation_bddx_{split_name}.json')
    if os.path.exists(out_path):
        print(f"[INFO] Segment annotation already exists, loading from {out_path}")
        with open(out_path) as f:
            return json.load(f)
    processed_conversations = process_split(split_name, info_root, decimal, fractional)
    print (len(processed_conversations), "processed conversations")
    result = []
    for item in tqdm(processed_conversations, desc='Segment Annotation'):
        video = item['video'][0]
        try:
            action = next((conv['value'] for conv in item['conversations'] if conv['from'] == 'gpt' and 'What is the action of ego car?' in item['conversations'][0]['value']), '')
        except Exception:
            action = ''
        try:
            cs_data = cs_extract(item['id'], processed_conversations)
        except Exception:
            cs_data = None
        result.append({
            'id': item.get('id'),
            'original_video': video,
            'ori_action': action,
            'control_signal': cs_data,
        })
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=4)
    print(f"[INFO] Segment Annotation saved to {out_path}, {len(result)} samples.")
    return result

# ---------------------- YOLO Detection and Predicate Extraction ----------------------
class YOLODetector:
    def __init__(self, yolo_ckpt_path, segment_annotation_dict: List[Dict]):
        self.model = YOLO(yolo_ckpt_path)
        self.segment_annotation_dict = segment_annotation_dict

    def load_video(self, video_path: str):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video {video_path}")
        return cap

    def detect_objects_yolo(self, frame) -> List[str]:
        return [self.model.names[int(box.cls)] for result in self.model(frame) for box in result.boxes]

    def yolo_detection_for_last_frame(self, video_path: str) -> List[str]:
        cap = self.load_video(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0:
            print(f"[WARN] Video {video_path} has no frames.")
            cap.release()
            return []
        # extract from last frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames - 1)
        ret, frame = cap.read()
        if not ret:
            print(f"[WARN] Failed to read last frame in {video_path}")
            cap.release()
            return []
        yolo_results = set(self.detect_objects_yolo(frame))
        cap.release()
        return list(yolo_results)

    def extract_environment_predicate(self, split_name, extraction_dir: str = 'data/extraction/bddx') -> List[Dict]:
        os.makedirs(extraction_dir, exist_ok=True)
        extraction_path = os.path.join(extraction_dir, f'extraction_bddx_{split_name}.json')
        if os.path.exists(extraction_path):
            print(f"[INFO] Extraction already exists, loading from {extraction_path}")
            with open(extraction_path) as f:
                return json.load(f)
        print(f"[INFO] Extracting YOLO classes for bddx_{split_name}...")
        action_list = BDDX().action_list
        extracted_data = []
        for item in tqdm(self.segment_annotation_dict, desc='Extracting YOLO classes'):
            video_path = item['original_video']
            yolo_results = self.yolo_detection_for_last_frame(video_path)
            ori_action = item.get('ori_action', '')
            cs = item.get('control_signal', {})
            gpt_response = gpt_map_action(ori_action).strip(string.whitespace + string.punctuation)
            gt_action_predicate = update_action_set(action_list, gpt_response)
            cs_pred = gpt_map_cs(cs.get('Speed', []), cs.get('Curvature', []), cs.get('Acceleration', []), cs.get('Course', []))
            extracted_data.append({
                'id': item['id'],
                'video': video_path,
                'ori_action': ori_action,
                'gt_action_predicate': gt_action_predicate,
                'velocity_predicate': update_action(BDDX().velocity_cs_list, cs_pred),
                'direction_predicate': update_action(BDDX().direction_cs_list, cs_pred),
                'classes': yolo_results,
                'cs': cs,
            })
        with open(extraction_path, 'w') as f:
            json.dump(extracted_data, f, indent=4)
        print(f"[INFO] Extraction completed, saved to {extraction_path}.")
        return extracted_data

def main(
    inference: bool,
    info_root: str,
    decimal: int,
    fractional: int,
    yolo_ckpt_path: str,
) -> None:
    """
    Main pipeline for BDDX predicate extraction and vectorization.
    """
    split_name = 'eval' if 'BDDX_Test' in info_root else 'train'
    info_root = Path(info_root)
    # 1. Generate annotation mapping
    segment_annotation_dict = create_segment_annotation(info_root=info_root, 
                                                        split_name=split_name, 
                                                        decimal=decimal, 
                                                        fractional=fractional)
    # 2. YOLO detection and class extraction
    yolo_detector = YOLODetector(yolo_ckpt_path, segment_annotation_dict)
    extraction_data = yolo_detector.extract_environment_predicate(split_name)
    # 3. Vectorization
    vectorize_segments(
        extraction_data,
        split_name,
        inference=inference,
    )
    print('[INFO] BDDX predicate extraction pipeline completed.')

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="BDDX Predicate Extraction Pipeline")
    parser.add_argument('--inference', action='store_true', help='Whether to extract mllm predicates (use only at inference time)')
    parser.add_argument('--info_root', type=str, default="data/BDDX_Test/info", help='Path to BDD-X info directory')
    parser.add_argument('--decimal', type=int, default=2, help='Decimal places for rounding')
    parser.add_argument('--fractional', type=int, default=3, help='Fractional places for rounding')
    parser.add_argument('--yolo', type=str, default="pgm/ckpts/YOLO/Lisa_finetuned.pt", help='Path to YOLO checkpoint')
    args = parser.parse_args()
    main(
        args.inference,
        args.info_root,
        args.decimal,
        args.fractional,
        args.yolo,
    )