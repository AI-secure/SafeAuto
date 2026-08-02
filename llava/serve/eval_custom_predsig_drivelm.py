import argparse
import re
import torch
import numpy as np

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
from pgm.pgm import DriveLMPGM
from pgm.config import DriveLM
from pgm.drivelm_predicate_extractor import behavior_to_actions, question2option, condition_vector

BEHAVIOR_QUESTION_KEY = "Predict the behavior of the ego vehicle"

def load_image(image_file):
    if image_file.startswith('http://') or image_file.startswith('https://'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image


def parse_option_letter(answer):
    match = re.search(r'\b([A-D])\b', answer)
    return match.group(1) if match else None


def pgm_verify_behavior(pgm, config, segment, question, answer):
    """Post-safety verification of a behavior answer (multiple choice).

    Ranks (velocity, direction) action pairs by PGM probability given the
    observed conditions and the MLLM-suggested actions, then finds the options
    matching the most probable pair. The MLLM answer is kept when it is among
    them; otherwise it is replaced by the top-ranked legal option.
    """
    options = question2option(question)
    letter = parse_option_letter(answer)
    if not options or letter is None:
        return answer
    descriptions = dict(options)
    llm_actions = behavior_to_actions(descriptions.get(letter))
    cond = np.array(condition_vector(segment, llm_actions))
    (velo_probs, dire_probs), _ = pgm.infer_action_probability(cond)
    option_actions = [(l, set(behavior_to_actions(desc))) for l, desc in options]
    v_num = config.velocity_action_num
    probable_answers = []
    for d_idx in np.argsort(-dire_probs):
        for v_idx in np.argsort(-velo_probs):
            pair = {config.action_list[v_idx], config.action_list[d_idx + v_num]}
            probable_answers = [l for l, acts in option_actions if acts == pair]
            if probable_answers:
                break
        if probable_answers:
            break
    if not probable_answers or letter in probable_answers:
        return answer
    return probable_answers[0]


def main(args):
    # Questions:

    json_file = args.input
    os.makedirs(args.output, exist_ok=True)
    out_json_paths = [f"{args.output}/DrivingLM_Test_pred.json"]

    # Model
    disable_torch_init()
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name,
                                                                     args.load_8bit, args.load_4bit, device=args.device)

    # PGM safety verification
    config = DriveLM()
    pgm = DriveLMPGM(config, weights=np.load(args.pgm_path))
    with open(args.extraction_path, 'r') as f:
        extraction = {inst['id']: inst for inst in json.load(f)}

    # print(model, tokenizer, processor)
    # image_processor = processor['image']
    # use image processor to handle the six images from video
    video_processor = processor['image']


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
    out_jsons = [[]]

    for item in tqdm(data):
        prediction = []
        qs = [q["value"] for q in item["conversations"][::2]]
        conv.messages.clear()
        if "mpt" in model_name.lower():
            roles = ('user', 'assistant')
        else:
            roles = conv.roles

        vps, vid = item["image"], item['id']

        video_paths = [vp for vp in vps]


        video_tensor = [video_processor(video_path, return_tensors='pt')['pixel_values'] for video_path in video_paths]
        if type(video_tensor) is list:
            tensor = [[video.to(model.device, dtype=torch.float16) for video in video_tensor]]
        else:
            tensor = video_tensor.to(model.device, dtype=torch.float16)

        key = ['image']

        inst_answers = []
        for qid, question in enumerate(qs):
            # print(question)
            inp = question

            if vps is not None:
                # First Message
                # inp = DEFAULT_X_TOKEN['VIDEO'] + '\n' + inp
                inp = inp
                conv.append_message(conv.roles[0], inp)
                video = None
            else:
                # later messages
                conv.append_message(conv.roles[0], inp)

            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            input_ids = tokenizer_X_token(prompt, tokenizer, X_TOKEN_INDEX['VIDEO'], return_tensors='pt').unsqueeze(0).cuda()

            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            keywords = [stop_str]
            stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
            streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)


            # import pdb;pdb.set_trace()
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
            caption = outputs.replace("</s>", "")

            # PGM safety verification on the behavior question; the corrected
            # answer replaces the model output in the conversation context so
            # that follow-up questions condition on it.
            if BEHAVIOR_QUESTION_KEY in question and vid in extraction:
                verified = pgm_verify_behavior(pgm, config, extraction[vid], question, caption)
                if verified != caption:
                    print(f"[PGM] {vid}: {caption!r} -> {verified!r}")
                    caption = verified
                    conv.messages[-1][-1] = verified + "</s>"

            if args.debug:
                print("\n", {"prompt": prompt, "outputs": outputs}, "\n")

            inst_pred = {
                "image_id":vid,
                "caption":caption
            }
            prediction.append(inst_pred)
        out_jsons[0].append(prediction)
        # import pdb; pdb.set_trace()
        # break
        for i in range(1):
            with open(out_json_paths[i],"w") as of:
                json.dump(out_jsons[i], of, indent=4)

    # Save separate json for action and justification
    for i in range(1):
        with open(out_json_paths[i],"w") as of:
            json.dump(out_jsons[i], of, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--pgm-path", type=str, default="./pgm/ckpts/pgm/drivelm_weights.npy")
    parser.add_argument("--extraction-path", type=str, default="./data/extraction/drivelm/extraction_drivelm_eval.json")
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
