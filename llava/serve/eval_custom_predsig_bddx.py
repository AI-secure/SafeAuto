import argparse
import torch

from llava.constants import X_TOKEN_INDEX, DEFAULT_X_TOKEN, DEFAULT_X_START_TOKEN, DEFAULT_X_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import process_images, tokenizer_X_token, get_model_name_from_path, KeywordsStoppingCriteria
import requests
from PIL import Image
from io import BytesIO
from transformers import TextStreamer

import os
import json
from tqdm import tqdm
import numpy as np
import string
from pgm.pgm import PGM
from pgm.config import BDDX
from pgm.bddx_predicate_extractor import update_action_set, list2vector, cs2vector, combine_vectors, EP_MAP, LLM_ACTION_MAP
from pgm.gpt_utils import gpt_map_action

def load_image(image_file):
    if image_file.startswith('http://') or image_file.startswith('https://'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image


def main(args):
    # Suggestion Mapping:
    suggestion_mapping = {
        'Keep': 'keep moving',
        'Accelerate': 'accelerate',
        'Decelerate': 'decelerate',
        'Stop': 'stop',
        'Reverse': 'reverse',
        'MakeLeftTurn': 'turn left',
        'MakeRightTurn': 'turn right',
        'MakeUTurn': 'make a U-turn',
        'Merge': 'merge into the traffic',
        'LeftPass': 'pass the car on the left side',
        'RightPass': 'pass the car on the right side',
        'Yield': 'yield to the other car',
        'ChangeToRightLane': 'change to the right lane',
        'ChangeToLeftLane': 'change to the left lane',
        'Park': 'park the car',
        'PullOver': 'pull over',
    }
    
    # Questions:

    json_file = args.input
    os.makedirs(args.output, exist_ok=True)
    out_json_paths = [f"{args.output}/BDDX_Test_pred_{cap}.json" for cap in ['action','justification','control_signal']]
    
    # Model
    disable_torch_init()
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name,
                                                                     args.load_8bit, args.load_4bit, device=args.device)
    
    # PGM
    config = BDDX()
    pgm = PGM(config, weights_path=np.load(args.pgm_path))
    extraction = json.load(open(args.extraction_path, 'r'))
    action_list = BDDX().action_list
    # print(model, tokenizer, processor)
    # image_processor = processor['image']
    video_processor = processor['video']


    conv_mode = "driving"

    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print('[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}'.format(conv_mode, args.conv_mode, args.conv_mode))
    else:
        args.conv_mode = conv_mode

    conv = conv_templates[args.conv_mode].copy()


    # gt
    with open(json_file, 'r') as file:
        data = json.load(file)

    # Pred
    out_jsons = [[],[],[]]
    
    for item in tqdm(data):
        q1, q2, q3 = item["conversations"][0]["value"], item["conversations"][2]["value"], item["conversations"][4]["value"]
        conv.messages.clear()
        if "mpt" in model_name.lower():
            roles = ('user', 'assistant')
        else:
            roles = conv.roles
            
        vps, vid = item["video"], item['id']
        
        video_paths = [os.path.join("./data",vp) for vp in vps]

        
        video_tensor = [video_processor(video_path, return_tensors='pt')['pixel_values'] for video_path in video_paths]
        if type(video_tensor) is list:
            tensor = [[video.to(model.device, dtype=torch.float16) for video in video_tensor]]
        else:
            tensor = video_tensor.to(model.device, dtype=torch.float16)
            
        key = ['video']
        
        inst_answers = []
        for qid, question in enumerate([q1, q2, q3]):
            if qid == 0:
                ask_q1 = question
                # only verify once
                for attempt in range(2):
                    conv.messages.clear()
                    conv.append_message(conv.roles[0], ask_q1)
                    conv.append_message(conv.roles[1], None)
                    prompt = conv.get_prompt()
                    input_ids = tokenizer_X_token(prompt, tokenizer, X_TOKEN_INDEX['VIDEO'], return_tensors='pt').unsqueeze(0).cuda()
                    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
                    keywords = [stop_str]
                    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
                    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
                    with torch.inference_mode():
                        output_ids = model.generate(
                            input_ids,
                            images=[tensor, key],
                            do_sample=False,
                            temperature=args.temperature,
                            max_new_tokens=args.max_new_tokens,
                            streamer=streamer,
                            use_cache=True,
                            stopping_criteria=[stopping_criteria])
                    outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
                    conv.messages[-1][-1] = outputs
                    inst_pred = {
                        "image_id": vid,
                        "caption": outputs.replace("</s>", "")
                    }
                    # PGM safety verification
                    extraction_inst = next((inst for inst in extraction if inst['id'] == vid), None)
                    gt_action_predicates = extraction_inst['gt_action_predicate']
                    pred_action = outputs.replace("</s>", "")
                    gpt_response = gpt_map_action(pred_action).strip(string.whitespace + string.punctuation)
                    pred_action_predicates = update_action_set(action_list, gpt_response)
                    illegal_predicates = [p for p in pred_action_predicates if p not in gt_action_predicates]
                    legal_predicates = [p for p in pred_action_predicates if p in gt_action_predicates]
                    if illegal_predicates and attempt == 0:
                        ep_vector = list2vector(extraction_inst, EP_MAP, inference=True)
                        cs_vector = cs2vector(extraction_inst['velocity_predicate'], extraction_inst['direction_predicate'], inference=True)
                        mllm_vector = list2vector(pred_action_predicates, LLM_ACTION_MAP, inference=True)
                        condition_vector = combine_vectors(ep_vector, cs_vector, mllm_vector)
                        pgm_probs, index = pgm.infer_action_probability(condition_vector)
                        pgm_suggest_predicate = action_list[index]
                        legal_predicates.append(pgm_suggest_predicate)
                        pgm_suggestion = "The car should " + ", ".join([suggestion_mapping[p] for p in legal_predicates if p in suggestion_mapping]) + "."
                        last_position = ask_q1.rfind("What is the action of ego car?")
                        ask_q1 = ask_q1[:last_position] + pgm_suggestion + ask_q1[last_position:]
                        continue 
                    out_jsons[qid].append(inst_pred)
                    break
            else:
                conv.append_message(conv.roles[0], question)
                conv.append_message(conv.roles[1], None)
                prompt = conv.get_prompt()
                input_ids = tokenizer_X_token(prompt, tokenizer, X_TOKEN_INDEX['VIDEO'], return_tensors='pt').unsqueeze(0).cuda()
                stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
                keywords = [stop_str]
                stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
                streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
                with torch.inference_mode():
                    output_ids = model.generate(
                        input_ids,
                        images=[tensor, key],
                        do_sample=False,
                        temperature=args.temperature,
                        max_new_tokens=args.max_new_tokens,
                        streamer=streamer,
                        use_cache=True,
                        stopping_criteria=[stopping_criteria])
                outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
                conv.messages[-1][-1] = outputs
                inst_pred = {
                    "image_id": vid,
                    "caption": outputs.replace("</s>", "")
                }
                out_jsons[qid].append(inst_pred)
        # import pdb; pdb.set_trace()
        # break
        for i in range(3):
            with open(out_json_paths[i],"w") as of:
                json.dump(out_jsons[i], of, indent=4)

    # Save separate json for action and justification
    for i in range(3):
        with open(out_json_paths[i],"w") as of:
            json.dump(out_jsons[i], of, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--pgm-path", type=str, default="./pgm/ckpts/pgm/bddx_weights.npy")
    parser.add_argument("--extraction-path", type=str, default="./data/extraction/bddx/extraction_bddx_eval.json")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--image-aspect-ratio", type=str, default='pad')
    parser.add_argument("--input", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()
    main(args)
