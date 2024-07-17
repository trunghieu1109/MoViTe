from utils import get_action_space
import random
import requests
import pandas as pd
import time
import numpy as np
import torch 
from movite_training_model import DQN
import json

mode = 'basic' # basic, flexible, diversity, full

requests.post("http://localhost:8933/LGSVL/SetMode?mode=" + mode)

if mode == 'diversity':
    w_col_prob = 0.5
    w_vio_prob = 0.3
    w_div_level = 0.2
else:
    w_col_prob = 0.6
    w_vio_prob = 0.4
    w_div_level = 0.0

# Execute action
def execute_action(action_id):
    api = action_space[str(action_id)]
    print("Start action: ", api)
    response = requests.post(api)
    obstacle_uid = None
    try:
        vioRate_list = response.json()['vioRate']
        obstacle_uid = response.json()['collision_uid']
    except Exception as e:
        print(e)
        vioRate_list = [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0]
        
    return vioRate_list, obstacle_uid

check_num = 5  # here depend on observation time: 2s->10, 4s->6, 6s->

position_space = []
position_space_size = 0

previous_weather_and_time_step = -5

def judge_done():
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
    
    global w_div_level
    global w_vio_prob
    global w_col_prob
    
    vioRate_list, obstacle_uid = execute_action(action_id)
    observation = get_environment_state()
    action_reward = 0
    violation_rate = 0
    violation_reward = 0
    diversity_level = 0
    collision_probability = 0
    collision_reward = 0
    episode_done = judge_done()
    
    collision_probability = round(float(
        (requests.get("http://localhost:8933/LGSVL/Status/CollisionProbability")).content.decode(
            encoding='utf-8')), 6)
    
    if collision_probability < 0.2:
        collision_reward = -1
    else:
        collision_reward = collision_probability
    
    violation_rate_reward = round(float(
        (requests.get("http://localhost:8933/LGSVL/Status/ViolationRateReward")).content.decode(
            encoding='utf-8')), 6)
    
    if violation_rate_reward < 0.2:
        violation_reward = -1
    else:
        violation_reward = violation_rate_reward
        
    isViolation = False
    
    for i in range(0, 7):
        if float(vioRate_list[i]) == 1.0:
            isViolation = True
    
    isCollision = False
    if float(collision_probability) == 1.0:
        isCollision = True
    
    addition_collision_reward = 0.0
    if isViolation and isCollision:
        addition_collision_reward = 1.0
        
    collision_reward += addition_collision_reward
    
    diversity_level = round(float(
        (requests.get("http://localhost:8933/LGSVL/Status/DiversityLevel")).content.decode(
            encoding='utf-8')), 6)
        
    action_reward = w_col_prob * collision_reward + w_vio_prob * violation_reward + w_div_level * diversity_level
            
    return observation, action_reward, violation_rate, episode_done, vioRate_list, collision_probability, obstacle_uid


def get_environment_state():
    
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
    
    state[12] = a['local_diff']
    state[13] = a['local_angle']
    
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


for hieu in range(0, 5):
    # initialize the environment
    requests.post("http://localhost:8933/LGSVL/LoadScene?scene=bd77ac3b-fbc3-41c3-a806-25915c777022&road_num=" + '1')
    requests.post("http://localhost:8933/LGSVL/SetObTime?observation_time=6")

    action_space = get_action_space()['command']
    action_space_size = action_space['num']

    print("Number of actions: ", action_space_size)

    current_eps = str(200)
    road_num = str(1)

    file_name = str(int(time.time()))

    dqn = DQN()
    
    model_path = './model/movite_tartu_basic_2_1/'
    log_path = '../ExperimentData/Random-or-Non-random Analysis/movite_tartu_basic_2_1/'
    
    if not os.path.isdir(log_path):
        print("Create dir", log_path)
        os.makedirs(log_path)

    dqn.eval_net.load_state_dict(torch.load(model_path + 'eval_net_' + current_eps + '_road' + road_num + '.pt'))
    dqn.target_net.load_state_dict(torch.load(model_path + 'target_net_' + current_eps + '_road' + road_num + '.pt'))

    title = ["Episode", "State", "Action", "Choosing_Type", "Violation Rate", "Violation Rate List", "Collision_uid", "Done"]
    df_title = pd.DataFrame([title])
    
    df_title.to_csv(log_path + 'dqn_6s_road1_' + current_eps + 'eps_' + file_name + '.csv', mode='w', header=False, index=None)

    iteration = 0
    step = 0
    print("Start episode 0")
    state_ = []
    action_ = []
    type_ = []
    vioRate_ = []
    vioArr_ = []
    proC_ = []
    collision_uid_ = []
    done_ = []
    isCollision = False
    
    while True:
        # Random select action to execute
    
        s = get_environment_state()
        current_step = step
            
        retry = True
        
        while(retry):
        
            retry = False
        
            try:
                action, type = dqn.choose_action(s, current_step, previous_weather_and_time_step)
                _, vioRate, done, vioRate_list, proC, collision_uid, = calculate_reward(action)
            except json.JSONDecodeError as e:
                print(e)    
                retry = True 
                
        if 0 <= action and action <= 12:
            previous_weather_and_time_step = step

        print('api_id, vioRate, vioRate_list, collision_probability, done: ', action, vioRate, vioRate_list, proC, done)
        
        for beh_vio in vioRate_list:
            if beh_vio == 1:
                isCollision = True
                
        if float(proC) == 1.0:
            isCollision = True
        
        state_.append(s)
        action_.append(action)
        type_.append(type)
        vioRate_.append(vioRate)
        collision_uid_.append(collision_uid)
        vioArr_.append(vioRate_list)
        proC_.append(proC)
        done_.append(done)

        step += 1
        if done:
            requests.post("http://localhost:8933/LGSVL/LoadScene?scene=bd77ac3b-fbc3-41c3-a806-25915c777022&road_num=" + '1')

            print("Length of episode: ", len(state_))
            if len(state_) <= 5:
                if not isCollision:
                    print("Restart this episode")
                else:
                    print("Episode complete")
                    for iter in range(0, len(state_)):
                        pd.DataFrame([[iteration, state_[iter], action_[iter], type_[iter], vioRate_[iter], vioArr_[iter], proC_[iter], collision_uid_[iter], done_[iter]]]).to_csv(
                            log_path + 'dqn_6s_road1_' + current_eps + 'eps_' + file_name + '.csv',
                            mode='a',
                            header=False, index=None)
                    iteration += 1
            else:
                print("Episode complete")
                for iter in range(0, len(state_)):
                    pd.DataFrame([[iteration, state_[iter], action_[iter], type_[iter], vioRate_[iter], vioArr_[iter], proC_[iter], collision_uid_[iter], done_[iter]]]).to_csv(
                        log_path + 'dqn_6s_road1_' + current_eps + 'eps_' + file_name + '.csv',
                        mode='a',
                        header=False, index=None)
                iteration += 1
                
            state_ = []
            action_ = []
            type_ = []
            vioRate_ = []
            collision_uid_ = []
            vioArr_ = []
            proC_ = []
            done_ = []
            current_step = 0
            previous_weather_and_time_step = -5
            dqn.previous_weather_and_time = None
            isCollision = False
            print("Start episode ", iteration)
            if iteration == 16:
                break
