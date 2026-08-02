import argparse
import json
import re

from pgm.drivelm_predicate_extractor import question2option, get_option

BEHAVIOR_QUESTION_KEY = "Predict the behavior of the ego vehicle"

SPEED_DESCRIPTIONS = ['driving fast', 'driving very fast', 'driving slowly',
                      'driving with normal speed', 'not moving']
COURSE_DESCRIPTIONS = ['slightly steering to the left', 'slightly steering to the right',
                       'steering to the left', 'steering to the right', 'going straight']

def split_speed_course(description):
    speed = next((d for d in SPEED_DESCRIPTIONS if d in description), None)
    course = next((d for d in COURSE_DESCRIPTIONS if d in description), None)
    return speed, course

def parse_option_letter(answer):
    match = re.search(r'\b([A-D])\b', answer)
    return match.group(1) if match else None

def main(args):
    with open(f"results/{args.path}/DrivingLM_Test_pred.json") as f:
        predictions = json.load(f)
    with open(args.conversation) as f:
        conversations = {item['id']: item for item in json.load(f)}
    with open(args.answers) as f:
        answers = json.load(f)

    total = correct = speed_correct = course_correct = 0
    for pred_list in predictions:
        vid = pred_list[0]['image_id']
        scene_id, frame_id = vid.split('_')
        item = conversations.get(vid)
        if item is None:
            continue
        questions = [c['value'] for c in item['conversations'][::2]]
        beh_idx = next((i for i, q in enumerate(questions) if BEHAVIOR_QUESTION_KEY in q), None)
        try:
            gt_letter = answers[scene_id]["key_frames"][frame_id]["QA"]["behavior"][0]["A"]
        except KeyError:
            continue
        if beh_idx is None or beh_idx >= len(pred_list):
            continue
        question = questions[beh_idx]
        pred_letter = parse_option_letter(pred_list[beh_idx]['caption'])
        if pred_letter is None:
            continue
        total += 1
        correct += (pred_letter == gt_letter)
        gt_desc = get_option(question, gt_letter) or ''
        pred_desc = get_option(question, pred_letter) or ''
        gt_speed, gt_course = split_speed_course(gt_desc)
        pred_speed, pred_course = split_speed_course(pred_desc)
        speed_correct += (pred_speed == gt_speed)
        course_correct += (pred_course == gt_course)

    print(f"Samples evaluated: {total}")
    print(f"Behavior accuracy: {correct / total:.4f}")
    print(f"Speed accuracy   : {speed_correct / total:.4f}")
    print(f"Steering accuracy: {course_correct / total:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate DriveLM behavior predictions")
    parser.add_argument('--path', type=str, required=True,
                        help='Result folder name under results/, e.g. drivelm_0.01-0.35_rag_top1')
    parser.add_argument('--conversation', type=str,
                        default='data/conversation/drivelm/conversation_drivelm_val.json')
    parser.add_argument('--answers', type=str, default='data/drivelm_val_answers.json')
    args = parser.parse_args()
    main(args)
