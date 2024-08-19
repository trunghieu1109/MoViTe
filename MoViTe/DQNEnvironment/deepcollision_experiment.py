from utils import get_action_space
import random
import requests
import pandas as pd
import time
import numpy as np
import torch 
from deepcollision import DQN
import json
import os

# Execute action
def execute_action(action_id):
    api = action_space[str(action_id)]
    print("Start action: ", api)
    response = requests.post(api)
    obstacle_uid = None
    sudden_appearance = False
    overlapping = False
    position_list = []
    
    try:
        probability_list = response.json()['probability']
        obstacle_uid = response.json()['collision_uid']
        sudden_appearance = response.json()['sudden_appearance']
        overlapping = response.json()['overlapping']
        position_list = response.json()['position_list']
    except Exception as e:
        print(e)
        probability_list = [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0]
        
    return probability_list, obstacle_uid, sudden_appearance, overlapping, position_list


latitude_space = []
check_num = 5  # here depend on observation time: 2s->10, 4s->6, 6s->
latitude_position = 0

position_space = []
position_space_size = 0

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
def calculate_reward(api_id):
    """
    Function for reward calculating.
    First, interpret action id to real RESTful API and call function -execute_action- to execute current RESTful API;
    then after the execution of RESTful API, the collision information are collected and based on it, we can calculate the reward.
    :param api_id:
    :return:
    """
    global latitude_position
    collision_probability_per_step, collision_uid, sudden_appearance, overlapping, position_list = execute_action(api_id)
    observation = None
    action_reward = 0
    # episode_done = False
    collision_probability = 0
    # Reward is calculated based on collision probability.
    collision_info = (requests.get("http://localhost:8933/LGSVL/Status/CollisionInfo")).content.decode(
        encoding='utf-8')
    episode_done = judge_done()

    if collision_info != 'None':
        collision_probability = 1
        # episode_done = True
    elif collision_info == "None":
        collision_probability = round(float(
            (requests.get("http://localhost:8933/LGSVL/Status/CollisionProbability")).content.decode(
                encoding='utf-8')), 3)  
    
        # action_reward = collision_probability.__float__()
    return observation, collision_probability, episode_done, collision_info, collision_probability_per_step, collision_uid, sudden_appearance, overlapping, position_list


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


for loop in range(0, 5):
    requests.post("http://localhost:8933/LGSVL/LoadScene?scene=12da60a7-2fc9-474d-a62a-5cc08cb97fe8&road_num=" + '3')
    requests.post("http://localhost:8933/LGSVL/SetObTime?observation_time=6")

    action_space = get_action_space()['command']
    action_space_size = action_space['num']

    print("Number of actions: ", action_space_size)

    current_eps = str(400)
    road_num = str(3)

    file_name = str(int(time.time()))

    dqn = DQN()
    
    folder_name = 'DeepCollision_6s_SanFrancisco_mod_proc'
    
    log_path = '../ExperimentData/Random-or-Non-random Analysis/{}/'.format(folder_name)
    model_path = './model/{}/'.format(folder_name)
    
    if not os.path.isdir(log_path):
        print("Create dir", log_path)
        os.makedirs(log_path)


    dqn.eval_net.load_state_dict(torch.load(model_path + 'eval_net_' + current_eps + '_road' + road_num + '.pt'))
    dqn.target_net.load_state_dict(torch.load(model_path + 'target_net_' + current_eps + '_road' + road_num + '.pt'))

    title = ["Episode", "State", "Action", "Choosing_Type", "Collision_Probability", "Collision_uid", "Sudden Appearance", "Overlapping", "Repeated Collision", "Collision_Probability_Per_Step", 'Collision List', "Done"]
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
    
    prev_collision_object = ""
    
    while True:
        # Random select action to execute

        s = get_environment_state()
            
        retry = True
        
        while(retry):
        
            retry = False
        
            try:
                action, type = dqn.choose_action(s)
                _, probability, done, info, collision_probability_per_step, collision_uid, sudden_appearance, overlapping, position_list = calculate_reward(action)
            except json.JSONDecodeError as e:
                print(e)    
                retry = True 
                
        if sudden_appearance:
            print(20*'*', "Scenario is removed due to sudden appearance", 20*'*')
            
        if overlapping:
            print(20*'*', "Scenario is removed due to overlapping", 20*'*')
            done = True
        
        if probability == 1.0:
            isCollision = True
            
        if info != 'None':
            if info == 'pedestrian':
                if prev_collision_object == 'pedestrian':
                    done = True
                    probability = 0
            prev_collision_object = info
        
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
        
        print('api_id, probability, sudden_appearance, overlapping, repeated_collision, done: ', action, probability, sudden_appearance, overlapping, repeated_collision, done)

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
                        pd.DataFrame([[iteration, state_[iter], action_[iter], type_[iter], probability_[iter], collision_uid_[iter], sudden_appearance_[iter], overlapping_[iter], repeated_collision_[iter], collision_probability_per_step_[iter], done_[iter]]]).to_csv(
                            log_path + 'dqn_6s_road3_' + current_eps + 'eps_' + file_name + '_0.2eps.csv',
                            mode='a',
                            header=False, index=None)
                    iteration += 1
            else:
                print("Episode complete")
                for iter in range(0, len(state_)):
                    pd.DataFrame([[iteration, state_[iter], action_[iter], type_[iter], probability_[iter], collision_uid_[iter], sudden_appearance_[iter], overlapping_[iter], repeated_collision_[iter], collision_probability_per_step_[iter], done_[iter]]]).to_csv(
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
            done_ = []
            isCollision = False
            prev_collision_object = ""
            npc_interaction_info = {}  
            print("Start episode ", iteration)
            if iteration == 16:
                break
