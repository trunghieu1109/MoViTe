from utils import get_action_space
import random
import requests
import pandas as pd
import time
import numpy as np
import json
import math
import os 

ETTC_threshold = 7 # (s)
DTO_threshold = 10 # (m)
JERK_threshold = 5 # (m/s^2)

# Execute action
def execute_action(action_id):
    api = action_space[str(action_id)]
    print("Start action: ", api)
    response = requests.post(api)
    obstacle_uid = None
    print(response)
    try:
        proC_list = response.json()['probability']
        DTO_list = response.json()['distance']
        ETTC_list = response.json()['ETTC']
        JERK_list = response.json()['JERK']
        obstacle_uid = response.json()['collision_uid']
    except Exception as e:
        print(e)
        proC_list = [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0]
        DTO_list = [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0]
        ETTC_list = [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0]
        JERK_list = [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0]
        
    return proC_list, DTO_list, ETTC_list, JERK_list, obstacle_uid

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


# Execute action and get return
def calculate_reward(action_id):
    """
    Function for reward calculating.
    First, interpret action id to real RESTful API and call function -execute_action- to execute current RESTful API;
    then after the execution of RESTful API, the collision information are collected and based on it, we can calculate the reward.
    :param action_id:
    :return:
    """
    
    global DTO_threshold
    global ETTC_threshold
    global JERK_threshold
    
    proC_list, DTO_list, ETTC_list, JERK_list, obstacle_uid = execute_action(action_id)
    observation = get_environment_state()
    action_reward = 0
    
    # Collision Probability Reward
    
    collision_probability = 0
    collision_reward = 0
    
    collision_info = (requests.get("http://localhost:8933/LGSVL/Status/CollisionInfo")).content.decode(
        encoding='utf-8')

    episode_done = judge_done()

    if collision_info != 'None':
        collision_reward = 1
        collision_probability = 1
        # episode_done = True
    elif collision_info == "None":
        collision_probability = round(float(
            (requests.get("http://localhost:8933/LGSVL/Status/CollisionProbability")).content.decode(
                encoding='utf-8')), 6)
        
        collision_reward = collision_probability
        
    print("Collision Probability: ", collision_probability)
    print("Collision Reward: ", collision_reward)
      
    # Distance to obstacles reward
          
    DTO = round(float(
            (requests.get("http://localhost:8933/LGSVL/Status/DistanceToObstacles")).content.decode(
                encoding='utf-8')), 6)
    
    DTO_reward = 0
    
    if 0 <= DTO <= DTO_threshold:
        DTO_reward = -math.log(DTO / DTO_threshold)
    else:
        DTO_reward = -1
        
    print("Distance to obstacles: ", DTO)
    print("DTO Reward: ", DTO_reward)
        
    # Estimated time to collision reward
    
    ETTC = round(float(
            (requests.get("http://localhost:8933/LGSVL/Status/EstimatedTimeToCollision")).content.decode(
                encoding='utf-8')), 6)
    
    ETTC_reward = 0
    
    if 0 < ETTC <= ETTC_threshold:
        ETTC_reward = -math.log(ETTC / ETTC_threshold)
    else:
        ETTC_reward = -1
        
    print("Estimated time to collision: ", ETTC)
    print("ETTC Reward: ", ETTC_reward)
        
    # JERK reward
    
    JERK = round(float(
            (requests.get("http://localhost:8933/LGSVL/Status/Jerk")).content.decode(
                encoding='utf-8')), 6)
    
    JERK_reward = 0
    
    print("JERK: ", JERK)
    
    if JERK > JERK_threshold:
        JERK_reward = math.exp((JERK - 0) / (10 - 0)) - 1
    else:
        JERK_reward = -1
        
    print("JERK Reward: ", JERK_reward)

    action_reward = collision_reward + DTO_reward / 3 + JERK_reward / 3 + ETTC_reward / 3
            
    return observation, action_reward, collision_probability, DTO, ETTC, JERK, episode_done, proC_list, DTO_list, ETTC_list, JERK_list, obstacle_uid



def get_environment_state():
    global per_confi
    global pred_confi
    
    r = requests.get("http://localhost:8933/LGSVL/Status/Environment/State")
    a = r.json()
    state = np.zeros(12)
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
    
    return state

requests.post("http://localhost:8933/LGSVL/LoadScene?scene=aae03d2a-b7ca-4a88-9e41-9035287a12cc&road_num=" + '1')
requests.post("http://localhost:8933/LGSVL/SetObTime?observation_time=6")

action_space = get_action_space()['command']
action_space_size = action_space['num']

for loop in range(0, 5):
    title = ["Episode", "Step", "State", "Action", "Collision_Probability", "DTO", "ETTC", "JERK", 
            "Collision_uid", "Collision_Probability_Per_Step", "DTO_list", "ETTC_list", "JERK_list", "Done"]
    df_title = pd.DataFrame([title])
    file_name = str(int(time.time()))
    
    file_path = '../ExperimentData/Random-or-Non-random Analysis/Data_Random_BorregasAve-short_paper/'
    
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
                _, _, probability, DTO, ETTC, JERK, done, collision_probability_per_step, DTO_list, ETTC_list, JERK_list, collision_uid, = calculate_reward(api_id)
            except json.JSONDecodeError as e:
                print(e)
                retry = True

        print('api_id, probability, DTO, ETTC, JERK, done: ', api_id, probability, DTO, ETTC, JERK,  done)
        pd.DataFrame([[iteration, step, s, api_id, probability, DTO, ETTC, JERK, collision_uid, collision_probability_per_step, DTO_list, ETTC_list, JERK_list, done]]).to_csv(
            file_path + 'random_6s_road1_' + file_name + '_dis_1' + '.csv',
            mode='a',
            header=False, index=None)

        step += 1
        if done:
            # break
            requests.post("http://localhost:8933/LGSVL/LoadScene?scene=aae03d2a-b7ca-4a88-9e41-9035287a12cc&road_num=" + '1')
            iteration += 1
            step = 0
            print("Start episode ", iteration)
            if iteration == 16:
                break
