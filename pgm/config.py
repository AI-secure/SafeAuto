from typing import Callable, Dict, List

class BDDX:
    """
    PGM configuration for the BDDX dataset, including actions, predicate mapping, and rule formulas.
    """
    def __init__(self):
        # High-level action list
        self.action_list: List[str] = [
            'Keep', 'Accelerate', 'Decelerate', 'Stop', 'Reverse',
            'MakeLeftTurn', 'MakeRightTurn', 'MakeUTurn', 'Merge',
            'LeftPass', 'RightPass', 'Yield', 'ChangeToLeftLane',
            'ChangeToRightLane', 'Park', 'PullOver'
        ]
        # Low-level velocity control signals
        self.velocity_cs_list: List[str] = ['Keep', 'Accelerate', 'Decelerate', 'Stop', 'Reverse']
        # Low-level direction control signals
        self.direction_cs_list: List[str] = ['Straight', 'Left', 'Right']

        # Predicate mapping table
        self.predicate: Dict[str, int] = {
            # Unobserved Predicates
            'KEEP': 0, 'ACCELERATE': 1, 'DECELERATE': 2, 'STOP': 3, 'REVERSE': 4,
            'MAKELEFTTURN': 5, 'MAKERIGHTTURN': 6, 'MAKEUTURN': 7, 'MERGE': 8,
            'LEFTPASS': 9, 'RIGHTPASS': 10, 'YIELD': 11, 'CHANGETOLEFTLANE': 12,
            'CHANGETORIGHTLANE': 13, 'PARK': 14, 'PULLOVER': 15,
            # Environment Predicates
            'SOLID_RED_LIGHT': 16, 'SOLID_YELLOW_LIGHT': 17, 'YELLOW_LEFT_ARROW_LIGHT': 18,
            'RED_LEFT_ARROW_LIGHT': 19, 'MERGING_TRAFFIC_SIGN': 20, 'NO_LEFT_TURN_SIGN': 21,
            'NO_RIGHT_TURN_SIGN': 22, 'PEDESTRIAN_CROSSING_SIGN': 23, 'STOP_SIGN': 24,
            'RED_YIELD_SIGN': 25, 'SLOW_SIGN': 26, 'SOLID_GREEN_LIGHT': 27,
            'KEEP_CS': 28, 'ACCELERATE_CS': 29, 'DECELERATE_CS': 30, 'STOP_CS': 31,
            'REVERSE_CS': 32, 'STRAIGHT_CS': 33, 'LEFT_CS': 34, 'RIGHT_CS': 35,
            # MLLM Action Predicates
            'KEEP_LLM': 36, 'ACCELERATE_LLM': 37, 'DECELERATE_LLM': 38, 'STOP_LLM': 39,
            'REVERSE_LLM': 40, 'MAKELEFTTURN_LLM': 41, 'MAKERIGHTTURN_LLM': 42,
            'MAKEUTURN_LLM': 43, 'MERGE_LLM': 44, 'LEFTPASS_LLM': 45, 'RIGHTPASS_LLM': 46,
            'YIELD_LLM': 47, 'CHANGETOLEFTLANE_LLM': 48, 'CHANGETORIGHTLANE_LLM': 49,
            'PARK_LLM': 50, 'PULLOVER_LLM': 51
        }

        self.action_num: int = 16
        self.condition_num: int = 36
        self.hardrule_num: int = 10

        # List of rule formulas, each formula is a lambda expression
        self.formulas: List[Callable[[List[int]], float]] = [
            lambda args: 1 - args[self.predicate["SOLID_RED_LIGHT"]] + args[self.predicate["SOLID_RED_LIGHT"]] * \
                ((1 - args[self.predicate["ACCELERATE"]]) * \
                (1 - args[self.predicate["LEFTPASS"]]) * \
                (1 - args[self.predicate["YIELD"]])), # SolidRedLight → ¬Accelerate ∧ ¬LeftPass ∧ ¬Yield
      
            lambda args: 1 - args[self.predicate["SOLID_YELLOW_LIGHT"]] + args[self.predicate["SOLID_YELLOW_LIGHT"]] * \
                            ((args[self.predicate["MAKELEFTTURN"]] + \
                                args[self.predicate["MAKERIGHTTURN"]] + \
                                args[self.predicate["KEEP"]] + \
                                args[self.predicate["STOP"]] + \
                                args[self.predicate["DECELERATE"]] - \
                                args[self.predicate["MAKELEFTTURN"]] * args[self.predicate["MAKERIGHTTURN"]] - \
                                args[self.predicate["MAKELEFTTURN"]] * args[self.predicate["KEEP"]] - \
                                args[self.predicate["MAKELEFTTURN"]] * args[self.predicate["STOP"]] - \
                                args[self.predicate["MAKELEFTTURN"]] * args[self.predicate["DECELERATE"]] - \
                                args[self.predicate["MAKERIGHTTURN"]] * args[self.predicate["KEEP"]] - \
                                args[self.predicate["MAKERIGHTTURN"]] * args[self.predicate["STOP"]] - \
                                args[self.predicate["MAKERIGHTTURN"]] * args[self.predicate["DECELERATE"]] - \
                                args[self.predicate["KEEP"]] * args[self.predicate["STOP"]] - \
                                args[self.predicate["KEEP"]] * args[self.predicate["DECELERATE"]] - \
                                args[self.predicate["STOP"]] * args[self.predicate["DECELERATE"]] + \
                                args[self.predicate["MAKELEFTTURN"]] * args[self.predicate["MAKERIGHTTURN"]] * \
                                args[self.predicate["KEEP"]] * args[self.predicate["STOP"]] * \
                                args[self.predicate["DECELERATE"]]) * \
                                (1 - args[self.predicate["ACCELERATE"]])), #SolidYellowLight → MakeLeftTurn ∨ MakeRightTurn∨ Keep ∨ Stop ∨ Decelerate ∧ ¬Accelerate

            lambda args: 1 - args[self.predicate["YELLOW_LEFT_ARROW_LIGHT"]] + args[self.predicate["YELLOW_LEFT_ARROW_LIGHT"]] * \
                            (args[self.predicate["STOP"]] + args[self.predicate["DECELERATE"]] - \
                            args[self.predicate["STOP"]] * args[self.predicate["DECELERATE"]]),  # YellowLeftArrowLight → Stop ∨ Decelerate

            lambda args: 1 - args[self.predicate["RED_LEFT_ARROW_LIGHT"]] + args[self.predicate["RED_LEFT_ARROW_LIGHT"]] * \
                            (1 - (args[self.predicate["MAKELEFTTURN"]] + args[self.predicate["MAKEUTURN"]] - \
                            args[self.predicate["MAKELEFTTURN"]] * args[self.predicate["MAKEUTURN"]])),  # RedLeftArrowLight → ¬(MakeLeftTurn ∨ MakeUTurn)

            lambda args: 1 - args[self.predicate["MERGING_TRAFFIC_SIGN"]] + args[self.predicate["MERGING_TRAFFIC_SIGN"]] * \
                            args[self.predicate["DECELERATE"]],  # MergingTrafficSign → Decelerate

            lambda args: 1 - args[self.predicate["NO_LEFT_TURN_SIGN"]] + args[self.predicate["NO_LEFT_TURN_SIGN"]] * \
                            (1 - args[self.predicate["MAKELEFTTURN"]]),  # NoLeftTurnSign → ¬MakeLeftTurn

            lambda args: 1 - args[self.predicate["NO_RIGHT_TURN_SIGN"]] + args[self.predicate["NO_RIGHT_TURN_SIGN"]] * \
                            (1 - args[self.predicate["MAKERIGHTTURN"]]),  # NoRightTurnSign → ¬MakeRightTurn

            lambda args: 1 - args[self.predicate["RED_YIELD_SIGN"]] + args[self.predicate["RED_YIELD_SIGN"]] * \
                            args[self.predicate["DECELERATE"]],  # RedYieldSign → Decelerate

            lambda args: 1 - args[self.predicate["SLOW_SIGN"]] + args[self.predicate["SLOW_SIGN"]] * \
                            (1 - args[self.predicate["ACCELERATE"]]),  # SlowSign → ¬Accelerate
                            
            lambda args: 1 - args[self.predicate["STOP_SIGN"]] + args[self.predicate["STOP_SIGN"]] * \
                            (1 - args[self.predicate["PULLOVER"]]),  # StopSign → ¬PULLOVER
            
            lambda args: 1 - args[self.predicate["KEEP_CS"]] + args[self.predicate["KEEP_CS"]] * \
                            (args[self.predicate["KEEP"]] + args[self.predicate["ACCELERATE"]]),  # KEEP_CS → KEEP ∨ ACCELERATE
                            
            lambda args: 1 - args[self.predicate["ACCELERATE_CS"]] + args[self.predicate["ACCELERATE_CS"]] * \
                            (args[self.predicate["KEEP"]] + args[self.predicate["ACCELERATE"]]),  # ACCELERATE_CS → KEEP ∨ ACCELERATE

            lambda args: 1 - args[self.predicate["DECELERATE_CS"]] + args[self.predicate["DECELERATE_CS"]] * \
                            (args[self.predicate["DECELERATE"]] + args[self.predicate["STOP"]]),  # DECELERATE_CS → DECELERATE ∨ STOP
            
            lambda args: 1 - args[self.predicate["STOP_CS"]] + args[self.predicate["STOP_CS"]] * \
                            (args[self.predicate["DECELERATE"]] + args[self.predicate["STOP"]]),  # STOP_CS → DECELERATE ∨ STOP
                            
            lambda args: 1 - args[self.predicate["REVERSE_CS"]] + args[self.predicate["REVERSE_CS"]] * \
                            args[self.predicate["REVERSE"]],  # REVERSE_CS → REVERSE
            
            lambda args: 1 - args[self.predicate["LEFT_CS"]] + args[self.predicate["LEFT_CS"]] * \
                            (args[self.predicate["MAKELEFTTURN"]] + args[self.predicate["CHANGETOLEFTLANE"]] - \
                            args[self.predicate["MAKELEFTTURN"]] * args[self.predicate["CHANGETOLEFTLANE"]]),  # LEFT_CS → MakeLeftTurn ∨ ChangeToLeftLane
            
            lambda args: 1 - args[self.predicate["RIGHT_CS"]] + args[self.predicate["RIGHT_CS"]] * \
                            (args[self.predicate["MAKERIGHTTURN"]] + args[self.predicate["CHANGETORIGHTLANE"]] - \
                            args[self.predicate["MAKERIGHTTURN"]] * args[self.predicate["CHANGETORIGHTLANE"]]),  # RIGHT_CS → MakeRightTurn ∨ ChangeToRightLane   
                
            lambda args: 1 - (args[self.predicate["LEFT_CS"]] * args[self.predicate["CHANGETORIGHTLANE_LLM"]]) + \
                    (args[self.predicate["LEFT_CS"]] * args[self.predicate["CHANGETORIGHTLANE_LLM"]] * args[self.predicate["CHANGETOLEFTLANE"]]), # LEFT_CS ∧ ChangeToRightLane_llm → ChangeToLeftLane   
                        
            lambda args: 1 - (args[self.predicate["RIGHT_CS"]] * args[self.predicate["CHANGETOLEFTLANE_LLM"]]) + \
                    (args[self.predicate["RIGHT_CS"]] * args[self.predicate["CHANGETOLEFTLANE_LLM"]] * args[self.predicate["CHANGETORIGHTLANE"]]),  # RIGHT_CS ∧ ChangeToLeftLane_llm → ChangeToRightLane   

            lambda args: 1 - args[self.predicate["KEEP_LLM"]] + args[self.predicate["KEEP_LLM"]] * \
                            args[self.predicate["KEEP"]],  # KEEP_LLM → KEEP
            
            lambda args: 1 - args[self.predicate["ACCELERATE_LLM"]] + args[self.predicate["ACCELERATE_LLM"]] * \
                            args[self.predicate["ACCELERATE"]],  # ACCELERATE_LLM → ACCELERATE
                            
            lambda args: 1 - args[self.predicate["DECELERATE_LLM"]] + args[self.predicate["DECELERATE_LLM"]] * \
                            args[self.predicate["DECELERATE"]],  # DECELERATE_LLM → DECELERATE  
                            
            lambda args: 1 - args[self.predicate["STOP_LLM"]] + args[self.predicate["STOP_LLM"]] * \
                            args[self.predicate["STOP"]],  # STOP_LLM → STOP  
                            
            lambda args: 1 - args[self.predicate["REVERSE_LLM"]] + args[self.predicate["REVERSE_LLM"]] * \
                            args[self.predicate["REVERSE"]],  # REVERSE_LLM → REVERSE
                            
            lambda args: 1 - args[self.predicate["MAKELEFTTURN_LLM"]] + args[self.predicate["MAKELEFTTURN_LLM"]] * \
                            args[self.predicate["MAKELEFTTURN"]],  # MAKELEFTTURN_LLM → MAKELEFTTURN
                            
            lambda args: 1 - args[self.predicate["MAKERIGHTTURN_LLM"]] + args[self.predicate["MAKERIGHTTURN_LLM"]] * \
                            args[self.predicate["MAKERIGHTTURN"]],  # MAKERIGHTTURN_LLM → MAKERIGHTTURN
                            
            lambda args: 1 - args[self.predicate["MAKEUTURN_LLM"]] + args[self.predicate["MAKEUTURN_LLM"]] * \
                            args[self.predicate["MAKEUTURN"]],  # MAKEUTURN_LLM → MAKEUTURN
                            
            lambda args: 1 - args[self.predicate["MERGE_LLM"]] + args[self.predicate["MERGE_LLM"]] * \
                            args[self.predicate["MERGE"]],  # MERGE_LLM → MERGE
                            
            lambda args: 1 - args[self.predicate["LEFTPASS_LLM"]] + args[self.predicate["LEFTPASS_LLM"]] * \
                            args[self.predicate["LEFTPASS"]],  # LEFTPASS_LLM → LEFTPASS
            
            lambda args: 1 - args[self.predicate["RIGHTPASS_LLM"]] + args[self.predicate["RIGHTPASS_LLM"]] * \
                            args[self.predicate["RIGHTPASS"]],  # RIGHTPASS_LLM → RIGHTPASS
                            
            lambda args: 1 - args[self.predicate["YIELD_LLM"]] + args[self.predicate["YIELD_LLM"]] * \
                            args[self.predicate["YIELD"]],  # YIELD_LLM → YIELD
                            
            lambda args: 1 - args[self.predicate["CHANGETOLEFTLANE_LLM"]] + args[self.predicate["CHANGETOLEFTLANE_LLM"]] * \
                            args[self.predicate["CHANGETOLEFTLANE"]],  # CHANGETOLEFTLANE_LLM → CHANGETOLEFTLANE
                            
            lambda args: 1 - args[self.predicate["CHANGETORIGHTLANE_LLM"]] + args[self.predicate["CHANGETORIGHTLANE_LLM"]] * \
                            args[self.predicate["CHANGETORIGHTLANE"]],  # CHANGETORIGHTLANE_LLM → CHANGETORIGHTLANE
                            
            lambda args: 1 - args[self.predicate["PARK_LLM"]] + args[self.predicate["PARK_LLM"]] * \
                            args[self.predicate["PARK"]],  # PARK_LLM → PARK
                            
            lambda args: 1 - args[self.predicate["PULLOVER_LLM"]] + args[self.predicate["PULLOVER_LLM"]] * \
                            args[self.predicate["PULLOVER"]]  # PULLOVER_LLM → PULLOVER          
        ]


class DriveLM:
    """
    PGM configuration for the DriveLM dataset, including actions, predicate mapping, and rule formulas.
    """
    def __init__(self):
        self.action_list: List[str] = [
            'Normal', 'Fast', 'Slow', 'Stop', 'Left', 'Right', 'Straight'
        ]

        self.predicate: Dict[str, int] = {
            # Actions
            'NORMAL': 0, 'FAST': 1, 'SLOW': 2, 'STOP': 3, 'LEFT': 4, 'RIGHT': 5, 'STRAIGHT': 6,
            # Environment conditions
            'SOLID_RED_LIGHT': 7, 'SOLID_YELLOW_LIGHT': 8, 'YELLOW_LEFT_ARROW_LIGHT': 9,
            'RED_LEFT_ARROW_LIGHT': 10, 'MERGING_TRAFFIC_SIGN': 11, 'NO_LEFT_TURN_SIGN': 12,
            'NO_RIGHT_TURN_SIGN': 13, 'PEDESTRIAN_CROSSING_SIGN': 14, 'STOP_SIGN': 15,
            'RED_YIELD_SIGN': 16, 'SLOW_SIGN': 17, 'SOLID_GREEN_LIGHT': 18,
            # Lane markings
            'DOUBLE_DASHED_WHITE_LINE_LEFT': 19, 'DOUBLE_DASHED_WHITE_LINE_RIGHT': 20,
            'SINGLE_SOLID_WHITE_LINE_LEFT': 21, 'SINGLE_SOLID_WHITE_LINE_RIGHT': 22,
            'DOUBLE_SOLID_WHITE_LINE_LEFT': 23, 'DOUBLE_SOLID_WHITE_LINE_RIGHT': 24,
            'SINGLE_ZIGZAG_WHITE_LINE_LEFT': 25, 'SINGLE_ZIGZAG_WHITE_LINE_RIGHT': 26,
            'SINGLE_SOLID_YELLOW_LINE_LEFT': 27, 'SINGLE_SOLID_YELLOW_LINE_RIGHT': 28,
            # Low-level control signals
            'NORMAL_CS': 29, 'FAST_CS': 30, 'SLOW_CS': 31, 'STOP_CS': 32,
            'LEFT_CS': 33, 'RIGHT_CS': 34, 'STRAIGHT_CS': 35,
            # LLM predicted actions
            'NORMAL_LLM': 36, 'FAST_LLM': 37, 'SLOW_LLM': 38, 'STOP_LLM': 39,
            'LEFT_LLM': 40, 'RIGHT_LLM': 41, 'STRAIGHT_LLM': 42
        }

        self.velocity_action_num: int = 4
        self.direction_action_num: int = 3
        self.action_num: int = 7
        self.condition_num: int = 36
        self.hardrule_num: int = 15

        # List of rule formulas, each formula is a lambda expression
        self.formulas: List[Callable[[List[int]], float]] = [
            lambda args: 1 - args[self.predicate["SOLID_RED_LIGHT"]] + args[self.predicate["FAST"]] * \
                      (1 - args[self.predicate["FAST"]]), # SolidRedLight → ¬Fast
                      
            lambda args: 1 - args[self.predicate["SOLID_YELLOW_LIGHT"]] + args[self.predicate["FAST"]] * \
                            (1 - args[self.predicate["FAST"]]), # SolidYellowLight → ¬Fast
                            
            lambda args: 1 - args[self.predicate["YELLOW_LEFT_ARROW_LIGHT"]] + args[self.predicate["YELLOW_LEFT_ARROW_LIGHT"]] * \
                            (args[self.predicate["STOP"]] + args[self.predicate["SLOW"]] - \
                            args[self.predicate["STOP"]] * args[self.predicate["SLOW"]]),  # YellowLeftArrowLight → Stop ∨ Slow

            lambda args: 1 - args[self.predicate["RED_LEFT_ARROW_LIGHT"]] + args[self.predicate["RED_LEFT_ARROW_LIGHT"]] * \
                            (1 - args[self.predicate["LEFT"]]),  # RedLeftArrowLight → ¬Left
            
            lambda args: 1 - args[self.predicate["MERGING_TRAFFIC_SIGN"]] + args[self.predicate["MERGING_TRAFFIC_SIGN"]] * \
                            (1- args[self.predicate["FAST"]]),  # MergingTrafficSign → ¬Fast

            lambda args: 1 - args[self.predicate["NO_LEFT_TURN_SIGN"]] + args[self.predicate["NO_LEFT_TURN_SIGN"]] * \
                            (1 - args[self.predicate["LEFT"]]),  # NoLeftTurnSign → ¬Left

            lambda args: 1 - args[self.predicate["NO_RIGHT_TURN_SIGN"]] + args[self.predicate["NO_RIGHT_TURN_SIGN"]] * \
                            (1 - args[self.predicate["RIGHT"]]),  # NoRightTurnSign → ¬Right
            
            lambda args: 1 - args[self.predicate["RED_YIELD_SIGN"]] + args[self.predicate["RED_YIELD_SIGN"]] * \
                            (1-args[self.predicate["FAST"]]),  # RedYieldSign → ¬Fast  

            lambda args: 1 - args[self.predicate["SLOW_SIGN"]] + args[self.predicate["SLOW_SIGN"]] * \
                            (1 - args[self.predicate["FAST"]]),  # SlowSign → ¬Fast  
                            
            lambda args: 1 - args[self.predicate["SINGLE_SOLID_WHITE_LINE_LEFT"]] + args[self.predicate["SINGLE_SOLID_WHITE_LINE_LEFT"]] * \
                            (1 - args[self.predicate["LEFT"]]),  # SingleSolidWhiteLineLeft → ¬Left
                            
            lambda args: 1 - args[self.predicate["SINGLE_SOLID_WHITE_LINE_RIGHT"]] + args[self.predicate["SINGLE_SOLID_WHITE_LINE_RIGHT"]] * \
                            (1 - args[self.predicate["RIGHT"]]),  # SingleSolidWhiteLineRight → ¬Right
                            
            lambda args: 1 - args[self.predicate["DOUBLE_SOLID_WHITE_LINE_LEFT"]] + args[self.predicate["DOUBLE_SOLID_WHITE_LINE_LEFT"]] * \
                            (1 - args[self.predicate["LEFT"]]),  # DOUBLE_SOLID_WHITE_LINE_LEFT → ¬Left
                            
            lambda args: 1 - args[self.predicate["DOUBLE_SOLID_WHITE_LINE_RIGHT"]] + args[self.predicate["DOUBLE_SOLID_WHITE_LINE_RIGHT"]] * \
                            (1 - args[self.predicate["RIGHT"]]),  # DOUBLE_SOLID_WHITE_LINE_RIGHT → ¬Right

            lambda args: 1 - args[self.predicate["SINGLE_ZIGZAG_WHITE_LINE_LEFT"]] + args[self.predicate["SINGLE_ZIGZAG_WHITE_LINE_LEFT"]] * \
                            (1 - args[self.predicate["STOP"]]),  # SingleZigzagWhiteLineLeft → ¬Stop

            lambda args: 1 - args[self.predicate["SINGLE_ZIGZAG_WHITE_LINE_RIGHT"]] + args[self.predicate["SINGLE_ZIGZAG_WHITE_LINE_RIGHT"]] * \
                            (1 - args[self.predicate["STOP"]]),  # SingleZigzagWhiteLineRight → ¬Stop 
                            
            lambda args: 1 - args[self.predicate["NORMAL_CS"]] + args[self.predicate["NORMAL_CS"]] * \
                            args[self.predicate["NORMAL"]],  # NORMAL_CS → NORMAL
                            
            lambda args: 1 - args[self.predicate["FAST_CS"]] + args[self.predicate["FAST_CS"]] * \
                            args[self.predicate["FAST"]],  # FAST_CS → FAST
                            
            lambda args: 1 - args[self.predicate["SLOW_CS"]] + args[self.predicate["SLOW_CS"]] * \
                            args[self.predicate["SLOW"]],  # SLOW_CS → SLOW
                            
            lambda args: 1 - args[self.predicate["STOP_CS"]] + args[self.predicate["STOP_CS"]] * \
                            args[self.predicate["STOP"]],  # STOP_CS → STOP 
                            
            lambda args: 1 - args[self.predicate["LEFT_CS"]] + args[self.predicate["LEFT_CS"]] * \
                            args[self.predicate["LEFT"]],  # LEFT_CS → LEFT
                            
            lambda args: 1 - args[self.predicate["RIGHT_CS"]] + args[self.predicate["RIGHT_CS"]] * \
                            args[self.predicate["RIGHT"]],  # RIGHT_CS → RIGHT
                            
            lambda args: 1 - args[self.predicate["STRAIGHT_CS"]] + args[self.predicate["STRAIGHT_CS"]] * \
                            args[self.predicate["STRAIGHT"]],  # STRAIGHT_CS → STRAIGHT
                            
            lambda args: 1 - args[self.predicate["NORMAL_LLM"]] + args[self.predicate["NORMAL_LLM"]] * \
                            args[self.predicate["NORMAL"]],  # NORMAL_LLM → NORMAL
                            
            lambda args: 1 - args[self.predicate["FAST_LLM"]] + args[self.predicate["FAST_LLM"]] * \
                            args[self.predicate["FAST"]],  # FAST_LLM → FAST
                            
            lambda args: 1 - args[self.predicate["SLOW_LLM"]] + args[self.predicate["SLOW_LLM"]] * \
                            args[self.predicate["SLOW"]],  # SLOW_LLM → SLOW
                            
            lambda args: 1 - args[self.predicate["STOP_LLM"]] + args[self.predicate["STOP_LLM"]] * \
                            args[self.predicate["STOP"]],  # STOP_LLM → STOP
                            
            lambda args: 1 - args[self.predicate["LEFT_LLM"]] + args[self.predicate["LEFT_LLM"]] * \
                            args[self.predicate["LEFT"]],  # LEFT_LLM → LEFT
                            
            lambda args: 1 - args[self.predicate["RIGHT_LLM"]] + args[self.predicate["RIGHT_LLM"]] * \
                            args[self.predicate["RIGHT"]],  # RIGHT_LLM → RIGHT
                            
            lambda args: 1 - args[self.predicate["STRAIGHT_LLM"]] + args[self.predicate["STRAIGHT_LLM"]] * \
                            args[self.predicate["STRAIGHT"]]  # STRAIGHT_LLM → STRAIGHT 
        ] 