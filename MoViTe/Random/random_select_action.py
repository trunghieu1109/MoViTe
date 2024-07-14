from utils import get_action_space
import random
import requests
import pandas as pd
import time
import numpy as np
import json
import os 


# Execute action
def execute_action(action_id):
    api = action_space[str(action_id)]
    print("Start action: ", api)
    response = requests.post(api)
    obstacle_uid = None
    print(response)
    try:
        vioRate_list = response.json()['vioRate']
        obstacle_uid = response.json()['collision_uid']
    except Exception as e:
        print(e)
        vioRate_list = [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0]
        
    return vioRate_list, obstacle_uid

latitude_space = []
check_num = 5  # here depend on observation time: 2s->10, 4s->6, 6s->
latitude_position = 0

position_space = []
position_space_size = 0


def judge_done():
    global latitude_position
    global position_space_size
    global position_space
    judge = False
    position = requests.get("http://localhost:8933/LGSVL/Status/EGOVehicle/Position").json()
    position_space.append((position['x'], position['y'], position['z']))
    position_space_size = (position_space_size + 1) % check_num
    if len(position_space) == 5:
        start_pos = position_space[0]
        end_pos = position_space[4]
        position_space = []
        dis = pow(
            pow(start_pos[0] - end_pos[0], 2) + pow(start_pos[1] - end_pos[1], 2) + pow(start_pos[2] - end_pos[2], 2),
            0.5)
        
        dis2 = start_pos[1] - end_pos[1]
        
        if dis < 0.15:
            judge = True
            
        if dis2 > 25:
            judge = True
    return judge


def calculate_reward(action_id):
    """
    Function for reward calculating.
    First, interpret action id to real RESTful API and call function -execute_action- to execute current RESTful API;
    then after the execution of RESTful API, the collision information are collected and based on it, we can calculate the reward.
    :param action_id:
    :return:
    """
    vioRate_list, obstacle_uid = execute_action(action_id)
    observation = get_environment_state()
    action_reward = 0
    violation_rate = 0
    episode_done = judge_done()
    
    violation_rate = round(float(
        (requests.get("http://localhost:8933/LGSVL/Status/ViolationRate")).content.decode(
            encoding='utf-8')), 6)
            
    return observation, violation_rate, episode_done, vioRate_list, obstacle_uid


def get_environment_state():
    global per_confi
    global pred_confi
    
    r = requests.get("http://localhost:8933/LGSVL/Status/Environment/State")
    a = r.json()
    state = np.zeros(43)
    state[0] = a['x']
    state[1] = a['y']
    state[2] = a['z']
    state[3] = a['rain']
    state[4] = a['fog']
    state[5] = a['wetness']
    state[6] = a['timeofday']
    state[7] = a['signal']
    state[8] = a['rx']
    state[9] = a['ry']
    state[10] = a['rz']
    state[11] = a['speed']
    
    # add advanced external states 
    state[12] = a['num_obs']
    state[13] = a['num_npc']
    state[14] = a['min_obs_dist']
    state[15] = a['speed_min_obs_dist']
    state[16] = a['vol_min_obs_dist']
    state[17] = a['dist_to_max_speed_obs']
    
    # add localization option
    
    state[18] = a['local_diff']
    state[19] = a['local_angle']
    
    # add perception option
    state[20] = a['dis_diff']
    state[21] = a['theta_diff']
    state[22] = a['vel_diff']
    state[23] = a['size_diff']
    
    # add prediction option
    state[24] = a['mlp_eval']
    state[25] = a['cost_eval']
    state[26] = a['cruise_mlp_eval']
    state[27] = a['junction_mlp_eval']
    state[28] = a['cyclist_keep_lane_eval']
    state[29] = a['lane_scanning_eval']
    state[30] = a['pedestrian_interaction_eval']
    state[31] = a['junction_map_eval']
    state[32] = a['lane_aggregating_eval']
    state[33] = a['semantic_lstm_eval']
    state[34] = a['jointly_prediction_planning_eval']
    state[35] = a['vectornet_eval']
    state[36] = a['unknown']
    
    # add control option
    state[37] = a['throttle']
    state[38] = a['brake']
    state[39] = a['steering_rate']
    state[40] = a['steering_target']
    state[41] = a['acceleration']
    state[42] = a['gear']
    
    return state

requests.post("http://localhost:8933/LGSVL/LoadScene?scene=bd77ac3b-fbc3-41c3-a806-25915c777022&road_num=" + '1')
requests.post("http://localhost:8933/LGSVL/SetObTime?observation_time=6")

action_space = get_action_space()['command']
action_space_size = action_space['num']

for loop in range(0, 5):
    title = ["Episode", "Step", "State", "Action", "Violation Rate", "Collision_uid", "Violation Rate List" "Done"]
    df_title = pd.DataFrame([title])
    file_name = str(int(time.time()))
    
    file_path = '../ExperimentData/Random-or-Non-random Analysis/Data_Random/'
    
    if not os.path.isdir(file_path):
        print("Create dir", file_path)
        os.makedirs(file_path)
    
    df_title.to_csv(file_path + 'random_6s_road1_' + file_name + '_dis_1' + '.csv', mode='w', header=False, index=None)

    iteration = 0
    step = 0
    print("Start episode 0")
    while True:
        # Random select action to execute

        s = get_environment_state()
        
        retry = True
        
        while(retry):
        
            retry = False
        
            try:
                api_id = random.randint(0, action_space_size - 1)
                _, violation_rate, done, vioRate_list, collision_uid, = calculate_reward(api_id)
            except json.JSONDecodeError as e:
                print(e)
                retry = True

        print('api_id, violation_rate, violation_rate_list, done: ', api_id, violation_rate, vioRate_list, done)

        pd.DataFrame([[iteration, step, s, api_id, violation_rate, collision_uid, vioRate_list, done]]).to_csv(
            '../ExperimentData/Random-or-Non-random Analysis/Data_Random_Opt_Min_Dis/random_6s_road1_' + file_name + '_dis_1' + '.csv',
            mode='a',
            header=False, index=None)

        step += 1
        if done:
            # break
            requests.post("http://localhost:8933/LGSVL/LoadScene?scene=bd77ac3b-fbc3-41c3-a806-25915c777022&road_num=" + '1')
            iteration += 1
            step = 0
            print("Start episode ", iteration)
            if iteration == 16:
                break
