import os
import json
import string
from typing import List, Dict
from tqdm import tqdm
from openai import OpenAI
from pgm.config import BDDX

def gpt_map_action(action: str) -> str:
    """
    Use GPT to map a natural language action description to one or more BDDX predicates.
    Args:
        action (str): The action description.
    Returns:
        str: The mapped predicate(s) as a string.
    """
    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    system_prompt = "You are a helpful assistant"
    prompt = """Given the current behavior of the car, please use one or two predicates below to best describe the behavior of the car. The predicates are: 
    Keep, Accelerate, Decelerate, Stop, Reverse, MakeLeftTurn, MakeRightTurn, MakeUTurn, Merge, LeftPass, RightPass, Yield, ChangeToLeftLane, ChangeToRightLane, Park, PullOver.
    Here are some examples:
    #Current Behavior#: The car is travelling down the road.
    #Predicates#: Keep\n
    #Current Behavior#: The car is making left turn.
    #Predicates#: MakeLeftTurn\n
    #Current Behavior#: The car is slowing down and then comes to a stop.
    #Predicates#: Decelerate, Stop\n
    #Current Behavior#: The car is accelerating and then turns right.
    #Predicates#: Accelerate, MakeRightTurn\n
    #Current Behavior#: The car is making a left turn and accelerates.
    #Predicates#: MakeLeftTurn, Accelerate\n
    #Current Behavior#: The car decelerates and stops.
    #Predicates#: Decelerate, Stop\n
    
    Now the current behavior of the car is described, provide the predicates that best describe the behavior of the car.:
    
    #Current Behavior#: {action}
    #Predicates#: """.format(action=action)
    
    messages=[{"role": "system", "content": system_prompt},
              {"role": "user", "content": prompt}]
    response = client.chat.completions.create(
        model='gpt-4o',
        messages=messages,
        temperature=0.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
    ).choices[0]
    return response.message.content

def gpt_map_cs(speed, curvature, acceleration, course) -> str:
    """
    Use GPT to map control signal arrays to velocity and direction predicates.
    Args:
        speed, curvature, acceleration, course: List or str representations of control signals.
    Returns:
        str: The mapped velocity and direction predicates as a string.
    """
    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    system_prompt = "You are a helpful assistant"
    prompt = """
    Given the current speed, curvature, acceleration, and course of the car, use one velocity predicate and one directional predicate to best describe the behavior of the car. 
    The velocity predicates are: Keep, Accelerate, Decelerate, Stop, Reverse.
    The directional predicates are: Straight, Left, Right. 
    Output the predicates directly without any additional information.
    Here are some examples:
    # Speed: [7.18, 5.76, 4.45, 3.30, 2.24, 1.20, 0.36]
    # Curvature: [1.32, 0.88, 0.58, 1.85, 2.74, 1.61, 0.64]
    # Acceleration: [-1.22, -1.85, -2.39, -2.22, -2.01, -1.46, -0.87]
    # Course: [0.00, -10.03, -8.33, -3.23, -0.97, -0.32, -0.08]
    # Predicate: Stop, Left
    # Speed: [12.31, 9.51, 7.24, 5.38, 3.67, 2.76, 3.00]
    # Curvature: [-0.00, 0.00, 0.00, -0.05, -0.18, -0.67, -0.79]
    # Acceleration: [-1.85, -2.79, -2.73, -2.23, -1.67, -0.47, 0.71]
    # Course: [0.00, 0.00, 0.00, 0.00, -20.26, -60.78, 7.17]
    # Predicate: Decelerate, Right
    # Speed: [1.27, 4.18, 6.83, 8.87, 10.44, 12.22, 14.45]
    # Curvature: [0.00, 0.00, 0.00, -0.00, -0.01, -0.00, -0.00]
    # Acceleration: [2.27, 2.15, 1.81, 1.35, 1.28, 1.56, 1.45]
    # Course: [0.00, -0.09, 0.00, 0.00, 0.20, 0.00, 0.00]
    # Predicate: Accelerate, Straight
    # Speed: {speed}
    # Curvature: {curvature}
    # Acceleration: {acceleration}
    # Course: {course}
    # Predicate: """.format(speed=speed, curvature=curvature, acceleration=acceleration, course=course)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]
    response = client.chat.completions.create(
        model='gpt-4o',
        messages=messages,
        temperature=0.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
    ).choices[0]
    return response.message.content

def map_LLM_pred(llm_prediction_path: str, llm_predicate_path: str) -> None:
    """
    Map LLM-generated captions to BDDX predicates using GPT and save the results.
    Args:
        llm_result_path (str): Path to the LLM result JSON file.
        save_path (str): Path to save the mapped predicate JSON file.
    """
    action_list = BDDX().action_list
    llm_predictions = json.load(open(llm_prediction_path))
    llm_predicates = []
    for item in tqdm(llm_predictions, desc='Mapping LLM predictions'):
        id = item['image_id']
        action = item['caption']
        answer = gpt_map_action(action)
        characters_to_remove = string.whitespace + string.punctuation
        answer = answer.strip(characters_to_remove)
        predicate = [act for act in action_list if act.lower() in answer.lower()]
        llm_predicates.append({'id': id, 'action': action, 'predicate': predicate})
    with open(llm_predicate_path, 'w') as f:
        json.dump(llm_predicates, f, indent=4) 
    return llm_predicates