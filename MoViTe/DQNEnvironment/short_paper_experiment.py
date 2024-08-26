from utils import get_action_space
import random
import requests
import pandas as pd
import time
import numpy as np
import torch 
from short_paper_training_model import DQN
import json
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
    sudden_appearance = False
    overlapping = False
    position_list = []
    isCollisionAhead = False
    collision_speed = 0
    pedes_mov_fw_to = False
    
    try:
        proC_list = response.json()['probability']
        obstacle_uid = response.json()['collision_uid']
        sudden_appearance = response.json()['sudden_appearance']
        overlapping = response.json()['overlapping']
        position_list = response.json()['position_list']
        isCollisionAhead = response.json()['isCollisionAhead']
        pedes_mov_fw_to = response.json()['pedes_mov_fw_to']
    except Exception as e:
        print(e)
        proC_list = [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0]
        
    return proC_list, obstacle_uid, sudden_appearance, overlapping, position_list, isCollisionAhead, pedes_mov_fw_to


latitude_space = []
check_num = 5  # here depend on observation time: 2s->10, 4s->6, 6s->
latitude_position = 0

position_space = []
position_space_size = 0

previous_weather_and_time_step = -5

per_confi = None
pred_confi = None

npc_interaction_info = {}

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
        
        if dis < 3:
            judge = True
            
        if abs(dis2) > 25:
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
    
    # proC_list, DTO_list, ETTC_list, JERK_list, obstacle_uid = execute_action(action_id)
    proC_list, obstacle_uid, sudden_appearance, overlapping, position_list, isCollisionAhead, pedes_mov_fw_to = execute_action(action_id)
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
  
    action_reward = collision_reward
            
    return observation, action_reward, collision_probability, episode_done, proC_list, obstacle_uid, collision_info, col_uid, sudden_appearance, overlapping, position_list, isCollisionAhead, pedes_mov_fw_to
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



for loop in range(0, 1):
    requests.post("http://localhost:8933/LGSVL/LoadScene?scene=12da60a7-2fc9-474d-a62a-5cc08cb97fe8&road_num=" + '3')
    requests.post("http://localhost:8933/LGSVL/SetObTime?observation_time=6")

    action_space = get_action_space()['command']
    action_space_size = action_space['num']

    print("Number of actions: ", action_space_size)

    current_eps = str(355)
    road_num = str(3)

    file_name = str(int(time.time()))

    dqn = DQN()
    
    folder_name = 'short_paper_sanfrancisco_road3_standard_ver_2'
    
    log_path = '../ExperimentData/Random-or-Non-random Analysis/{}/'.format(folder_name)
    model_path = './model/{}/'.format(folder_name)
    
    if not os.path.isdir(log_path):
        print("Create dir", log_path)
        os.makedirs(log_path)


    dqn.eval_net.load_state_dict(torch.load(model_path + 'eval_net_' + current_eps + '_road' + road_num + '.pt'))
    dqn.target_net.load_state_dict(torch.load(model_path + 'target_net_' + current_eps + '_road' + road_num + '.pt'))

    title = ["Episode", "State", "Action", "Choosing_Type", "Collision_Probability",
            "Collision_uid", "Collision_Probability_Per_Step", "Sudden Appearance", "Overlapping", "Repeated Collision", "Unreal Pedes Col", "Done"]
    df_title = pd.DataFrame([title])
    df_title.to_csv(log_path + 'dqn_6s_road3_' + current_eps + 'eps_' + file_name + '_0.2eps.csv', mode='w', header=False, index=None)

    iteration = 0
    step = 0
    print("Start episode 0")
    state_ = []
    action_ = []
    type_ = []
    probability_ = []
    collision_uid_ = []
    sudden_appearance_ = []
    overlapping_ = []
    collision_probability_per_step_ = []
    done_ = []
    isCollision = False
    position_list_ = []
    repeated_collision_ = []
    isCollision_list_ = []
    unreal_pedes_col_ = []
    
    while True:
        # Random select action to execute

        s = get_environment_state()
        current_step = step
            
        retry = True
        
        while(retry):
        
            retry = False
        
            try:
                action, type = dqn.choose_action(s, current_step, previous_weather_and_time_step)
                _, _, probability, done, collision_probability_per_step, collision_uid, info, _, sudden_appearance, overlapping, position_list, isCollisionAhead, pedes_mov_fw_to = calculate_reward(action)
            except json.JSONDecodeError as e:
                print(e)    
                retry = True 

        if 0 <= action and action <= 12:
            previous_weather_and_time_step = step

        if sudden_appearance:
            print(20*'*', "Scenario is removed due to sudden appearance", 20*'*')
            
        if overlapping:
            print(20*'*', "Scenario is removed due to overlapping", 20*'*')
            done = True
        
        if probability == 1.0:
            isCollision = True
        
        isCollision_list = []
        
        state_.append(s)
        action_.append(action)
        type_.append(type)
        probability_.append(probability)
        collision_uid_.append(collision_uid)
        sudden_appearance_.append(sudden_appearance)
        overlapping_.append(overlapping)
        collision_probability_per_step_.append(collision_probability_per_step)
        done_.append(done)
        position_list_.append(position_list)
        
        repeated_collision = False
        unreal_pedes_col = False
        
        # 0. No Collision
        # 1. Collision
        # 2. Repeated Collissison
        # 3. Sudden Collision
        # 4. Overlap Collision
        
        no_check_repeated = False
        
        if position_list:
        
            for time_step in position_list.values():
                for npc_uid in time_step:
                    if npc_uid == collision_uid:
                        if npc_uid in npc_interaction_info:
                            
                            if info == 'pedestrian' and not no_check_repeated:
                                done = True
                                repeated_collision = True
                            
                            if npc_interaction_info[npc_uid]['collision'] and not no_check_repeated:
                                repeated_collision = True
                            else:
                                npc_interaction_info[npc_uid]['collision_distance'] = time_step[npc_uid]['dis_to_ego']
                                npc_interaction_info[npc_uid]['collision'] = True 
                                no_check_repeated = True
                        else:
                            npc_interaction_info[npc_uid] = {}
                            npc_interaction_info[npc_uid]['collision'] = True
                            npc_interaction_info[npc_uid]['collision_distance'] = time_step[npc_uid]['dis_to_ego']
                            no_check_repeated = True
                    else:
                        if npc_uid in npc_interaction_info:
                            if abs(time_step[npc_uid]['dis_to_ego'] - npc_interaction_info[npc_uid]['collision_distance']) > 2:
                                npc_interaction_info[npc_uid]['collision'] = False
                
        repeated_collision_.append(repeated_collision)
        
        if float(probability) == 1.0 and info == 'pedestrian':
            if not isCollisionAhead:
                unreal_pedes_col = True
                print(20*'*', "Not collision ahead", 20*'*')
            else:
                if pedes_mov_fw_to:
                    unreal_pedes_col = True
                    print(20*'*', "Pedes move forward to ego", 20*'*')
                    
        unreal_pedes_col_.append(unreal_pedes_col)
        
        print('api_id, probability, sudden_appearance, overlapping, repeated_collision, unreal_pedes_col, done: ', action, probability, sudden_appearance, overlapping, repeated_collision, unreal_pedes_col, done)

        step += 1
        if done:
            # break
            # requests.post("http://localhost:8933/LGSVL/LoadScene?scene=aae03d2a-b7ca-4a88-9e41-9035287a12cc&road_num=" + '1')
            requests.post("http://localhost:8933/LGSVL/LoadScene?scene=12da60a7-2fc9-474d-a62a-5cc08cb97fe8&road_num=" + '3')

            print("Length of episode: ", len(state_))
            if len(state_) <= 5:
                if not isCollision:
                    print("Restart this episode")
                else:
                    print("Episode complete")
                    for iter in range(0, len(state_)):
                        pd.DataFrame([[iteration, state_[iter], action_[iter], type_[iter], probability_[iter], collision_uid_[iter], 
                                        collision_probability_per_step_[iter], sudden_appearance_[iter], overlapping_[iter], repeated_collision_[iter], unreal_pedes_col_[iter], done_[iter]]]).to_csv(
                            log_path + 'dqn_6s_road3_' + current_eps + 'eps_' + file_name + '_0.2eps.csv',
                            mode='a',
                            header=False, index=None)
                    iteration += 1
            else:
                print("Episode complete")
                for iter in range(0, len(state_)):
                    pd.DataFrame([[iteration, state_[iter], action_[iter], type_[iter], probability_[iter], collision_uid_[iter], 
                                    collision_probability_per_step_[iter], sudden_appearance_[iter], overlapping_[iter], repeated_collision_[iter], unreal_pedes_col_[iter], done_[iter]]]).to_csv(
                        log_path + 'dqn_6s_road3_' + current_eps + 'eps_' + file_name + '_0.2eps.csv',
                        mode='a',
                        header=False, index=None)
                iteration += 1
                
            state_ = []
            action_ = []
            type_ = []
            probability_ = []
            collision_uid_ = []
            collision_probability_per_step_ = []
            sudden_appearance_ = []
            overlapping_ = []
            done_ = []
            isCollision = False
            current_step = 0
            previous_weather_and_time_step = -5
            prev_collision_object = ""
            npc_interaction_info = {}  

            position_list_ = []
            repeated_collision_ = []
            isCollision_list_ = []
            unreal_pedes_col_ = []
        
            print("Start episode ", iteration)
            if iteration == 16:
                break