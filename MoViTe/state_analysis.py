import numpy as np
import pandas as pd
import argparse
import matplotlib.pyplot as plt
import re

from enum import Enum

class State(Enum):
    POSITION_X = 0
    POSITION_Y = 1
    POSITION_Z = 2
    RAIN = 3
    FOG = 4
    WETNESS = 5
    TIMEOFDAY = 6
    SIGNAL = 7
    ROTATION_X = 8
    ROTATION_Y = 9
    ROTATION_Z = 10
    SPEED = 11
    NUM_OBS = 12
    NUM_NPC = 13
    MIN_OBS_DIST = 14
    SPEED_MIN_OBS_DIST = 15
    VOL_MIN_OBS_DIST = 16
    DIST_TO_MAX_SPEED_OBS = 17
    LOCAL_DIFF = 18
    LOCAL_ANGLE = 19
    DIS_DIFF = 20
    THETA_DIFF = 21
    VEL_DIFF = 22
    SIZE_DIFF = 23
    MLP_EVAL = 24
    COST_EVAL = 25
    CRUISE_MLP_EVAL = 26
    JUNCTION_MLP_EVAL = 27
    CYCLIST_KEEP_LANE_EVAL = 28
    LANE_SCANNING_EVAL = 29
    PEDESTRIAN_INTERACTION_EVAL = 30
    JUNCTION_MAP_EVAL = 31
    LANE_AGGREGATING_EVAL = 32
    SEMANTIC_LSTM_EVAL = 33
    JOINTLY_PREDICTION_PLANNING_EVAL = 34
    VECTORNET_EVAL = 35
    UNKNOWN = 36
    THROTTLE = 37
    BRAKE = 38
    STEERING_RATE = 39
    STEERING_TARGET = 40
    ACCELERATION = 41
    GEAR = 42
    

obs_time = 6.0

def action_analysis(exp_file):
    
    df = pd.read_csv(exp_file)
    state = []
    prob = []
    
    for index, row in df.iterrows():
        state_ = re.split('|'.join(map(re.escape, [' ', '\n'])), row['State'][1:-1])
        state_c = []
        for state_i in state_:
            if state_i != '':
                state_c.append(float(state_i))
        state.append(state_c[State.BRAKE.value])
        cp = float(row["Collision_Probability"])
        # state.append(state_)
        prob.append(cp)
        # break
        
    print(state)
                
    plt.scatter(state, prob, marker='o', color='b')
    plt.xlabel('Brake')
    plt.ylabel('Collision Probability')
    plt.title('Brake vs Collision Probability')
    plt.show()
    
    # plt.plot(action_by_model)
    # plt.xlabel('Action')
    # plt.ylabel('Freq')
    # plt.title('Action x Freq')
    # plt.show()
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process input and output files.")
    parser.add_argument("exp_file", help="Experiment File")
    # parser.add_argument("n_actions", help="Number of actions")

    args = parser.parse_args()

    action_analysis(args.exp_file)