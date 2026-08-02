import os
import re
import json
import pickle
from tqdm import tqdm
from typing import Dict, List, Optional
from pgm.config import DriveLM

_CFG = DriveLM()
PREDICATE_NUM = _CFG.action_num + _CFG.condition_num
ACTION_NUM = _CFG.action_num
ACTION_MAP = {k: _CFG.predicate[k.upper()] for k in _CFG.action_list}
CS_MAP = {k: _CFG.predicate[f"{k.upper()}_CS"] for k in _CFG.action_list}
LLM_ACTION_MAP = {k: _CFG.predicate[f"{k.upper()}_LLM"] for k in _CFG.action_list}
# environment predicate mapping (YOLO traffic-sign classes and NuScenes map
# lane-divider types to predicate index)
EP_MAP = {
    "stop traffic light": _CFG.predicate["SOLID_RED_LIGHT"],
    "warning": _CFG.predicate["SOLID_YELLOW_LIGHT"],
    "warning traffic light": _CFG.predicate["SOLID_YELLOW_LIGHT"],
    "warning left": _CFG.predicate["YELLOW_LEFT_ARROW_LIGHT"],
    "warning left traffic light": _CFG.predicate["YELLOW_LEFT_ARROW_LIGHT"],
    "stop left": _CFG.predicate["RED_LEFT_ARROW_LIGHT"],
    "stop left traffic light": _CFG.predicate["RED_LEFT_ARROW_LIGHT"],
    "merge": _CFG.predicate["MERGING_TRAFFIC_SIGN"],
    "noLeftTurn": _CFG.predicate["NO_LEFT_TURN_SIGN"],
    "noRightTurn": _CFG.predicate["NO_RIGHT_TURN_SIGN"],
    "pedestrianCrossing": _CFG.predicate["PEDESTRIAN_CROSSING_SIGN"],
    "stop": _CFG.predicate["STOP_SIGN"],
    "stopAhead": _CFG.predicate["STOP_SIGN"],
    "yield": _CFG.predicate["RED_YIELD_SIGN"],
    "yieldAhead": _CFG.predicate["RED_YIELD_SIGN"],
    "slow": _CFG.predicate["SLOW_SIGN"],
    "go": _CFG.predicate["SOLID_GREEN_LIGHT"],
    "go forward": _CFG.predicate["SOLID_GREEN_LIGHT"],
    "go forward traffic light": _CFG.predicate["SOLID_GREEN_LIGHT"],
    "DOUBLE_DASHED_WHITE_LEFT": _CFG.predicate["DOUBLE_DASHED_WHITE_LINE_LEFT"],
    "DOUBLE_DASHED_WHITE_RIGHT": _CFG.predicate["DOUBLE_DASHED_WHITE_LINE_RIGHT"],
    "SINGLE_SOLID_WHITE_LEFT": _CFG.predicate["SINGLE_SOLID_WHITE_LINE_LEFT"],
    "SINGLE_SOLID_WHITE_RIGHT": _CFG.predicate["SINGLE_SOLID_WHITE_LINE_RIGHT"],
    "DOUBLE_SOLID_WHITE_LEFT": _CFG.predicate["DOUBLE_SOLID_WHITE_LINE_LEFT"],
    "DOUBLE_SOLID_WHITE_RIGHT": _CFG.predicate["DOUBLE_SOLID_WHITE_LINE_RIGHT"],
    "SINGLE_ZIGZAG_WHITE_LEFT": _CFG.predicate["SINGLE_ZIGZAG_WHITE_LINE_LEFT"],
    "SINGLE_ZIGZAG_WHITE_RIGHT": _CFG.predicate["SINGLE_ZIGZAG_WHITE_LINE_RIGHT"],
    "SINGLE_SOLID_YELLOW_LEFT": _CFG.predicate["SINGLE_SOLID_YELLOW_LINE_LEFT"],
    "SINGLE_SOLID_YELLOW_RIGHT": _CFG.predicate["SINGLE_SOLID_YELLOW_LINE_RIGHT"],
}

# ---------------------- Behavior-option Parsing ----------------------
def behavior_to_actions(sentence: Optional[str]) -> List[str]:
    """Map a DriveLM behavior-option description to action predicates.

    e.g. "The ego vehicle is steering to the left. The ego vehicle is driving
    slowly." -> ['Slow', 'Left']. Slight steering counts as Straight.
    """
    mapping_rules = {
        r'going straight': 'Straight',
        r'driving fast': 'Fast',
        r'driving very fast': 'Fast',
        r'driving slowly': 'Slow',
        r'driving with normal speed': 'Normal',
        r'not moving': 'Stop',
        r'slightly steering to the left': 'Straight',
        r'slightly steering to the right': 'Straight',
        r'steering to the left': 'Left',
        r'steering to the right': 'Right',
    }
    actions, matched_patterns = [], set()
    for pattern, action in mapping_rules.items():
        if sentence and re.search(pattern, sentence, re.IGNORECASE):
            if 'steering' in pattern:
                if 'slightly' in pattern or 'steering' not in matched_patterns:
                    matched_patterns.add('steering')
                    actions.append(action)
            else:
                actions.append(action)
    return actions

def question2option(question: str) -> List[tuple]:
    """Parse the A-D options of a behavior question into (letter, description)."""
    pattern = r"([A-D])\. (.*?)(?= [A-D]\.|$)"
    return [(m[0], m[1].strip()) for m in re.findall(pattern, question)]

def get_option(text: str, option_letter: str) -> Optional[str]:
    """Return the description of the given option letter inside a question."""
    match = re.search(rf"{option_letter}\.\s(.+?)(?=\s[A-Z]\.|$)", text, re.DOTALL)
    return match.group(1).strip() if match else None

# ---------------------- Vectorization ----------------------
def list2vector(items: List[str], mapping: dict) -> List[int]:
    vector = [0] * PREDICATE_NUM
    for item in items or []:
        idx = mapping.get(item)
        if idx is not None:
            vector[idx] = 1
    return vector

def cs2vector(velocity: Optional[str], direction: Optional[str]) -> List[int]:
    vector = [0] * PREDICATE_NUM
    for val in (velocity, direction):
        idx = CS_MAP.get(val)
        if idx is not None:
            vector[idx] = 1
    return vector

def combine_vectors(*vectors: List[int]) -> List[int]:
    return [max(values) for values in zip(*vectors)]

def segment2vector(segment: Dict, llm_actions: Optional[List[str]] = None) -> List[int]:
    """Convert an extraction entry (and MLLM-predicted actions) to a full
    predicate vector: [actions | environment | lane | control signals | MLLM]."""
    action_vec = list2vector(segment.get('gt_action_predicate'), ACTION_MAP)
    class_vec = list2vector(segment.get('classes'), EP_MAP)
    cs_vec = cs2vector(segment.get('velocity_predicate'), segment.get('direction_predicate'))
    llm_vec = list2vector(llm_actions or [], LLM_ACTION_MAP)
    return combine_vectors(action_vec, class_vec, cs_vec, llm_vec)

def condition_vector(segment: Dict, llm_actions: Optional[List[str]] = None) -> List[int]:
    """The condition part of the predicate vector, fed to PGM inference."""
    return segment2vector(segment, llm_actions)[ACTION_NUM:]

def vectorize_extraction(
    extraction: List[Dict],
    vector_save_path: str,
    llm_predicates: Optional[Dict[str, List[str]]] = None,
) -> List[List[int]]:
    """Convert extraction entries to full predicate vectors for PGM training.

    The MLLM action block is filled from llm_predicates (a mapping from sample
    id to predicted action predicates); entries without a prediction fall back
    to the ground-truth actions (teacher forcing, as in the original released
    checkpoint's training data).
    """
    vectors = []
    for item in tqdm(extraction, desc='Vectorizing segments'):
        llm_actions = (llm_predicates or {}).get(item['id'], item.get('gt_action_predicate'))
        vectors.append(segment2vector(item, llm_actions))
    os.makedirs(os.path.dirname(vector_save_path), exist_ok=True)
    with open(vector_save_path, 'wb') as f:
        pickle.dump(vectors, f)
    print(f"[INFO] Saved {len(vectors)} predicate vectors to {vector_save_path}.")
    return vectors

# ---------------------- Raw NuScenes/YOLO Extraction ----------------------
class DriveLMExtractor:
    """Extract environment predicates for DriveLM keyframes.

    Traffic signs/lights come from a fine-tuned YOLOv8 detector on the front
    cameras; lane-divider types and pedestrian crossings come from the
    NuScenes map expansion. Requires the `nuscenes-devkit` package and the
    NuScenes dataset (with map expansion) — only needed to regenerate
    `data/extraction/drivelm/*.json`, which are already released.
    """

    def __init__(self, nuscenes_root: str, yolo_ckpt_path: str = 'pgm/ckpts/YOLO/Lisa_finetuned.pt', version: str = 'v1.0-trainval'):
        from nuscenes.nuscenes import NuScenes
        from nuscenes.map_expansion.map_api import NuScenesMap
        from ultralytics import YOLO
        self.nusc = NuScenes(version=version, dataroot=nuscenes_root, verbose=False)
        self.maps = {name: NuScenesMap(dataroot=nuscenes_root, map_name=name)
                     for name in ('singapore-onenorth', 'singapore-hollandvillage',
                                  'boston-seaport', 'singapore-queenstown')}
        self.yolo = YOLO(yolo_ckpt_path)

    def _map_for_scene(self, scene_token: str):
        scene_info = self.nusc.get('scene', scene_token)
        location = self.nusc.get('log', scene_info['log_token'])['location']
        return self.maps[location]

    def _ego_pose(self, frame_token: str):
        sample_info = self.nusc.get('sample', frame_token)
        cam_front = self.nusc.get('sample_data', sample_info['data']['CAM_FRONT'])
        return self.nusc.get('ego_pose', cam_front['ego_pose_token'])['translation']

    def _closest_divider_type(self, ego_x, ego_y, map_instance, divider_segments):
        nodes = {n['token']: n for n in map_instance.node}
        best, best_dist = None, float('inf')
        for seg in divider_segments:
            node = nodes[seg['node_token']]
            dist = (ego_x - node['x']) ** 2 + (ego_y - node['y']) ** 2
            if dist < best_dist:
                best, best_dist = seg, dist
        return best

    def extract_conditions(self, image_paths: List[str], scene_token: str, frame_token: str) -> List[str]:
        """Detect YOLO classes on the given camera images and read lane
        information around the ego pose from the NuScenes map."""
        classes = set()
        for img_path in image_paths:
            for detection in self.yolo(img_path, verbose=False):
                for box in detection.boxes:
                    classes.add(self.yolo.names[int(box.cls)])
        map_instance = self._map_for_scene(scene_token)
        ego_x, ego_y, _ = self._ego_pose(frame_token)
        road_on_point = map_instance.layers_on_point(ego_x, ego_y)
        if road_on_point.get('ped_crossing'):
            classes.add('pedestrianCrossing')
        closest_lane = map_instance.get_closest_lane(ego_x, ego_y, radius=3)
        lane_info = next((l for l in map_instance.lane if l['token'] == closest_lane), None)
        if lane_info:
            for side in ('left', 'right'):
                segments = lane_info[f'{side}_lane_divider_segments']
                if segments:
                    node = self._closest_divider_type(ego_x, ego_y, map_instance, segments)
                    classes.add(f"{node['segment_type']}_{side.upper()}")
        return list(classes)

def gpt_map_cs(speed, course) -> str:
    """Map raw speed/course sequences to one velocity and one direction
    predicate with GPT (requires OPENAI_API_KEY)."""
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    prompt = """
    Given the current speed and course of the car, use one velocity predicate and one directional predicate to best describe the behavior of the car.
    The velocity predicates are: Normal, Fast, Slow, Stop.
    The directional predicates are: Straight, Left, Right.
    Output the predicates directly without any additional information.
    Here are some examples:
    # Speed: [(4.54, 0.0), (5.34, 0.0), (5.67, 0.0), (5.7, 0.0), (6.46, 0.0), (6.63, 0.0)]
    # Course: [(1.0, 0.0), (1.0, 0.0), (1.0, 0.0), (1.0, 0.0), (1.0, 0.0), (1.0, 0.0)]
    # Predicate: Fast, Straight
    # Speed: [(10.01, 0.0), (9.88, 0.0), (9.52, 0.0), (9.39, 0.0), (9.15, 0.0), (8.94, 0.0)]
    # Course: [(0.84, 0.0), (0.84, 0.0), (0.86, 0.0), (0.89, 0.0), (0.93, 0.0), (0.95, 0.0)]
    # Predicate: Fast, Right
    # Speed: [(2.51, 0.0), (2.49, 0.0), (2.45, 0.0), (2.43, 0.0), (2.43, 0.0), (2.37, 0.0)]
    # Course: [(0.85, 0.0), (0.85, 0.0), (0.86, 0.0), (0.85, 0.0), (0.82, 0.0), (0.75, 0.0)]
    # Predicate: Slowly, Left
    # Speed: [(1.65, 0.0), (1.37, 0.0), (0.73, 0.0), (0.09, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0)]
    # Course: [(0.86, 0.0), (0.86, 0.0), (0.87, 0.0), (0.86, 0.0), (0.86, 0.0), (0.86, 0.0), (0.85, 0.0), (0.84, 0.0)]
    # Predicate: Stop, Straight
    # Speed: {speed}
    # Course: {course}
    # Predicate: """.format(speed=speed, course=course)
    response = client.chat.completions.create(
        model='gpt-4o',
        messages=[{"role": "system", "content": "You are a helpful assistant"},
                  {"role": "user", "content": prompt}],
        temperature=0,
    )
    return response.choices[0].message.content

# ---------------------- Pipeline ----------------------
def main(split: str, extraction_path: str, llm_predicate_path: Optional[str], vector_save_dir: str) -> None:
    with open(extraction_path) as f:
        extraction = json.load(f)
    llm_predicates = None
    if llm_predicate_path:
        with open(llm_predicate_path) as f:
            llm_predicates = {x['id']: x['predicate'] for x in json.load(f)}
    vectorize_extraction(
        extraction,
        os.path.join(vector_save_dir, f'{split}_pgm_vectors.pkl'),
        llm_predicates=llm_predicates,
    )
    print('[INFO] DriveLM predicate vectorization completed.')

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="DriveLM Predicate Vectorization Pipeline")
    parser.add_argument('--split', type=str, default='train', choices=['train', 'eval'])
    parser.add_argument('--extraction_path', type=str, default=None,
                        help='Path to the extraction JSON (defaults to the released one for the split)')
    parser.add_argument('--llm_predicate_path', type=str, default=None,
                        help='Optional JSON with MLLM-predicted action predicates [{id, predicate}]; '
                             'defaults to teacher forcing with ground-truth actions')
    parser.add_argument('--vector_save_dir', type=str, default='pgm/predicates/drivelm')
    args = parser.parse_args()
    extraction_path = args.extraction_path or f'data/extraction/drivelm/extraction_drivelm_{args.split}.json'
    main(args.split, extraction_path, args.llm_predicate_path, args.vector_save_dir)
