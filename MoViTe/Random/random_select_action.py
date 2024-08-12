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
        proC_list = response.json()['probability']
        # DTO_list = response.json()['distance']
        # ETTC_list = response.json()['ETTC']
        # JERK_list = response.json()['JERK']
        obstacle_uid = response.json()['collision_uid']
    except Exception as e:
        print(e)
        proC_list = [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0]
        # DTO_list = [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0]
        # ETTC_list = [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0]
        # JERK_list = [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0]
        
    return proC_list, obstacle_uid

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
    
    global DTO_threshold
    global ETTC_threshold
    global JERK_threshold
    
    # proC_list, DTO_list, ETTC_list, JERK_list, obstacle_uid = execute_action(action_id)
    proC_list, obstacle_uid = execute_action(action_id)
    observation = None
    action_reward = 0
    
    # Collision Probability Reward
    
    collision_probability = 0
    collision_reward = 0
    
    collision_info = (requests.get("http://localhost:8933/LGSVL/Status/CollisionInfo")).content.decode(
        encoding='utf-8')

    print("Collision Info: ", collision_info)

    col_uid = (requests.get("http://localhost:8933/LGSVL/Status/CollisionUid")).content.decode(
        encoding='utf-8')

    print("Collision Uid: ", col_uid)

    episode_done = judge_done()

    if collision_info != 'None':
        collision_reward = 7.5
        collision_probability = 1
    elif collision_info == "None":
        collision_probability = round(float(
            (requests.get("http://localhost:8933/LGSVL/Status/CollisionProbability")).content.decode(
                encoding='utf-8')), 6)
        if collision_probability < 0.2:
            collision_reward = -1
        elif 0.2 <= collision_probability < 1.0:
            collision_reward = collision_probability
        else:
            collision_reward = 7.5
        
    print("Collision Probability: ", collision_probability)
    print("Collision Reward: ", collision_reward)
      
    # Distance to obstacles reward
          
    # DTO = round(float(
    #         (requests.get("http://localhost:8933/LGSVL/Status/DistanceToObstacles")).content.decode(
    #             encoding='utf-8')), 6)
    
    # DTO_reward = 0
    
    # if 0 <= DTO <= DTO_threshold:
    #     DTO_reward = 1 - DTO / DTO_threshold
    # else:
    #     if collision_probability < 0.2:
    #         DTO_reward = -1
        
    # print("Distance to obstacles: ", DTO)
    # print("DTO Reward: ", DTO_reward)
        
    # Estimated time to collision reward
    
    # ETTC = round(float(
    #         (requests.get("http://localhost:8933/LGSVL/Status/EstimatedTimeToCollision")).content.decode(
    #             encoding='utf-8')), 6)
    
    # ETTC_reward = 0
    
    # if 0 < ETTC <= ETTC_threshold:
    #     ETTC_reward = 1 - ETTC / ETTC_threshold
    # else:
    #     if collision_probability < 0.2:
    #         ETTC_reward = -1
        
    # print("Estimated time to collision: ", ETTC)
    # print("ETTC Reward: ", ETTC_reward)
        
    # JERK reward
    
    # JERK = round(float(
    #         (requests.get("http://localhost:8933/LGSVL/Status/Jerk")).content.decode(
    #             encoding='utf-8')), 6)
    
    # JERK_reward = 0
    
    # print("JERK: ", JERK)
    
    # if JERK > JERK_threshold:
    #     JERK_reward = 2 / (1 + math.exp(-(JERK - JERK_threshold))) - 1
    # else:
    #     if collision_probability < 0.2:
    #         JERK_reward = -1
        
    # print("JERK Reward: ", JERK_reward)

    # action_reward = collision_reward + DTO_reward * 0.4 + JERK_reward * 0.2 + ETTC_reward * 0.4
    action_reward = collision_reward
            
    return observation, action_reward, collision_probability, episode_done, proC_list, obstacle_uid, collision_info, col_uid
    # return observation, action_reward, collision_probability, DTO, ETTC, JERK, episode_done, proC_list, DTO_list, ETTC_list, JERK_list, obstacle_uid


def get_environment_state():
    
    r = requests.get("http://localhost:8933/LGSVL/Status/Environment/State")
    a = r.json()
    state = np.zeros(24)
    state[0] = a['x']
    state[1] = a['y']
    state[2] = a['z']
    state[3] = a['weather']
    state[4] = a['timeofday']
    state[5] = a['signal']
    state[6] = a['rx']
    state[7] = a['ry']
    state[8] = a['rz']
    state[9] = a['speed']
    
    # add advanced external states 
    state[10] = a['num_obs']
    state[11] = a['min_obs_dist']
    state[12] = a['speed_min_obs_dist']
    
    # add localization option
    
    state[13] = a['local_diff']
    state[14] = a['local_angle']
    
    # add perception option
    state[15] = a['dis_diff']
    state[16] = a['theta_diff']
    state[17] = a['vel_diff']
    state[18] = a['size_diff']
    
    # add control option
    state[19] = a['throttle']
    state[20] = a['brake']
    state[21] = a['steering_rate']
    state[22] = a['steering_target']
    state[23] = a['acceleration']

    return state

requests.post("http://localhost:8933/LGSVL/LoadScene?scene=12da60a7-2fc9-474d-a62a-5cc08cb97fe8&road_num=" + '3')
requests.post("http://localhost:8933/LGSVL/SetObTime?observation_time=6")

action_space = get_action_space()['command']
action_space_size = action_space['num']

for loop in range(0, 5):
    title = ["Episode", "Step", "State", "Action", "Collision_Probability",
            "Collision_uid", "Collision_Probability_Per_Step", "Done"]
    df_title = pd.DataFrame([title])
    file_name = str(int(time.time()))
    
    file_path = '../ExperimentData/Random-or-Non-random Analysis/Data_Random_SanFrancisco_road3/'
    
    if not os.path.isdir(file_path):
        print("Create dir", file_path)
        os.makedirs(file_path)
    
    df_title.to_csv(file_path + 'random_6s_road3_' + file_name + '_dis_1' + '.csv', mode='w', header=False, index=None)

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
                _, _, probability, done, collision_probability_per_step, collision_uid, _, _ = calculate_reward(api_id)
            except json.JSONDecodeError as e:
                print(e)
                retry = True

        print('api_id, probability, proC_list, done: ', api_id, probability, collision_probability_per_step, done)

        pd.DataFrame([[iteration, step, s, api_id, probability, collision_uid, collision_probability_per_step, done]]).to_csv(
            file_path + 'random_6s_road3_' + file_name + '_dis_1' + '.csv',
            mode='a',
            header=False, index=None)

        step += 1
        if done:
            # break
            requests.post("http://localhost:8933/LGSVL/LoadScene?scene=12da60a7-2fc9-474d-a62a-5cc08cb97fe8&road_num=" + '3')
            iteration += 1
            step = 0
            print("Start episode ", iteration)
            if iteration == 16:
                break
