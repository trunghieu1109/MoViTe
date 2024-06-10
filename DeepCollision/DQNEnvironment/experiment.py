from utils import get_action_space
import random
import requests
import pandas as pd
import time
import numpy as np
import torch 
from deepcollision_per import DQN
import json

# Execute action
def execute_action(action_id):
    api = action_space[str(action_id)]
    print("Start action: ", api)
    response = requests.post(api)
    obstacle_uid = None
    try:
        probability_list = response.json()['probability']
        obstacle_uid = response.json()['collision_uid']
    except Exception as e:
        print(e)
        probability_list = [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0]
        
    return probability_list, obstacle_uid


latitude_space = []
check_num = 5  # here depend on observation time: 2s->10, 4s->6, 6s->
latitude_position = 0

position_space = []
position_space_size = 0

per_confi = None
pred_confi = None


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
    collision_probability_per_step, collision_uid = execute_action(api_id)
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
    return observation, collision_probability, episode_done, collision_info, collision_probability_per_step, collision_uid


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
    
    
    # pd.DataFrame([[a['per'], a['pred']]]).to_csv("./log/max_distance_obs_" + file_name + ".csv", mode='a', header=False, index=None)

    return state


for hieu in range(0, 5):
    # initialize the environment
    # requests.post("http://localhost:8933/LGSVL/LoadScene?scene=SanFrancisco")
    # requests.post("http://localhost:8933/LGSVL/LoadScene?scene=aae03d2a-b7ca-4a88-9e41-9035287a12cc&road_num=" + '1')
    requests.post("http://localhost:8933/LGSVL/LoadScene?scene=bd77ac3b-fbc3-41c3-a806-25915c777022&road_num=" + '1')
    requests.post("http://localhost:8933/LGSVL/SetObTime?observation_time=6")

    action_space = get_action_space()['command']
    action_space_size = action_space['num']

    print("Number of actions: ", action_space_size)

    current_eps = str(400)
    road_num = str(1)

    file_name = str(int(time.time()))

    dqn = DQN()

    # dqn.eval_net.load_state_dict(torch.load('./model/DQN_local_ver_3_experiment_6' + 's/eval_net_' + current_eps + '_road' + road_num + '.pt'))
    # dqn.target_net.load_state_dict(torch.load('./model/DQN_local_ver_3_experiment_6' +'s/target_net_' + current_eps + '_road' + road_num + '.pt'))

    # dqn.eval_net.load_state_dict(torch.load('./model/DQN_per_experiment_6' + 's/eval_net_' + current_eps + '_road' + road_num + '.pt'))
    # dqn.target_net.load_state_dict(torch.load('./model/DQN_per_experiment_6' +'s/target_net_' + current_eps + '_road' + road_num + '.pt'))

    # dqn.eval_net.load_state_dict(torch.load('./model/DQN_experiment_customized_6' + 's_161/eval_net_' + current_eps + '_road' + road_num + '.pt'))
    # dqn.target_net.load_state_dict(torch.load('./model/DQN_experiment_customized_6' + 's_161/target_net_' + current_eps + '_road' + road_num + '.pt'))

    dqn.eval_net.load_state_dict(torch.load('./model/innercollision_tartu_new_sd/eval_net_' + current_eps + '_road' + road_num + '.pt'))
    dqn.target_net.load_state_dict(torch.load('./model/innercollision_tartu_new_sd/target_net_' + current_eps + '_road' + road_num + '.pt'))

    # dqn.eval_net.load_state_dict(torch.load('./model/DQN_per_pred_ver_2_experiment_6' + 's/eval_net_' + current_eps + '_road' + road_num + '.pt'))
    # dqn.target_net.load_state_dict(torch.load('./model/DQN_per_pred_ver_2_experiment_6' +'s/target_net_' + current_eps + '_road' + road_num + '.pt'))

    title = ["Episode", "State", "Action", "Choosing_Type", "Collision_Probability", "Collision_uid", "Collision_Probability_Per_Step", "Done"]
    df_title = pd.DataFrame([title])
    # file_name = str(int(time.time()))
    # df_title.to_csv('../ExperimentData/Analysis/Data_DQN/dqn_6s_road1_' + current_eps + 'eps_' + file_name + '.csv', mode='w', header=False, index=None)
    # df_title.to_csv('../ExperimentData/Analysis/Data_DQN_local_ver_3/dqn_6s_road1_' + file_name + '.csv', mode='w', header=False, index=None)
    # df_title.to_csv('../ExperimentData/Analysis/Data_DQN_per/dqn_6s_road1_' + file_name + '.csv', mode='w', header=False, index=None)
    df_title.to_csv('../ExperimentData/Random-or-Non-random Analysis/innercollision_tartu_new_sd/dqn_6s_road1_' + current_eps + 'eps_' + file_name + '_0.2eps.csv', mode='w', header=False, index=None)


    # title3 = ["is_detected", "max_distance"]
    # pd.DataFrame([title3]).to_csv("./log/max_distance_obs_" + file_name + ".csv", mode='w', header=False, index=None)

    # title2 = ['per_confi', 'pred_confi', 'reward']
    # pd.DataFrame([title2]).to_csv("./log/per_pred_reward_" + file_name + '.csv', mode='w', header=False, index=None)

    iteration = 0
    step = 0
    print("Start episode 0")
    state_ = []
    action_ = []
    type_ = []
    probability_ = []
    collision_uid_ = []
    collision_probability_per_step_ = []
    done_ = []
    isCollision = False
    
    while True:
        # Random select action to execute

        s = get_environment_state()
            
        retry = True
        
        while(retry):
        
            retry = False
        
            try:
                action, type = dqn.choose_action(s)
                _, probability, done, _, collision_probability_per_step, collision_uid, = calculate_reward(action)
            except json.JSONDecodeError as e:
                print(e)    
                retry = True 

        print('api_id, probability, done: ', action, probability, done)
        pd.DataFrame([[action, probability, collision_probability_per_step, done]]).to_csv('experiment_data/dqn_record_' + file_name + '.csv', mode='a', header=False,
                                                        index=None)

        # pd.DataFrame([[iteration, s, action, probability, collision_probability_per_step, done]]).to_csv(
        #    '../ExperimentData/Analysis/Data_DQN_local_ver_3/dqn_6s_road1_' + file_name + '.csv',
        #    mode='a',
        #    header=False, index=None)
        
        # pd.DataFrame([[iteration, s, action, probability, collision_probability_per_step, done]]).to_csv(
        #    '../ExperimentData/Analysis/Data_DQN_per/dqn_6s_road1_' + file_name + '.csv',
        #    mode='a',
        #    header=False, index=None)
        
        # pd.DataFrame([[iteration, s, action, probability, collision_uid, collision_probability_per_step, done]]).to_csv(
        #     '../ExperimentData/Analysis/Data_DQN/dqn_6s_road1_' + current_eps + 'eps_' + file_name + '.csv',
        #     mode='a',
        #     header=False, index=None)
        
        if probability == 1.0:
            isCollision = True
        
        state_.append(s)
        action_.append(action)
        type_.append(type)
        probability_.append(probability)
        collision_uid_.append(collision_uid)
        collision_probability_per_step_.append(collision_probability_per_step)
        done_.append(done)
        
        # pd.DataFrame([[iteration, s, action, type, probability, collision_uid, collision_probability_per_step, done]]).to_csv(
        #     '../ExperimentData/Random-or-Non-random Analysis/Data_DQN_default_1000MS/dqn_6s_road1_' + current_eps + 'eps_' + file_name + '_0.2eps.csv',
        #     mode='a',
        #     header=False, index=None)
        
        # pd.DataFrame([[per_confi, pred_confi, probability]]).to_csv("./log/per_pred_reward_" + file_name + '.csv', mode='a', header=False, index=None)

        step += 1
        if done:
            # break
            # requests.post("http://localhost:8933/LGSVL/LoadScene?scene=aae03d2a-b7ca-4a88-9e41-9035287a12cc&road_num=" + '1')
            requests.post("http://localhost:8933/LGSVL/LoadScene?scene=bd77ac3b-fbc3-41c3-a806-25915c777022&road_num=" + '1')

            print("Length of episode: ", len(state_))
            if len(state_) <= 5:
                if not isCollision:
                    print("Restart this episode")
                else:
                    print("Episode complete")
                    for iter in range(0, len(state_)):
                        pd.DataFrame([[iteration, state_[iter], action_[iter], type_[iter], probability_[iter], collision_uid_[iter], collision_probability_per_step_[iter], done_[iter]]]).to_csv(
                            '../ExperimentData/Random-or-Non-random Analysis/innercollision_tartu_new_sd/dqn_6s_road1_' + current_eps + 'eps_' + file_name + '_0.2eps.csv',
                            mode='a',
                            header=False, index=None)
                    iteration += 1
            else:
                print("Episode complete")
                for iter in range(0, len(state_)):
                    pd.DataFrame([[iteration, state_[iter], action_[iter], type_[iter], probability_[iter], collision_uid_[iter], collision_probability_per_step_[iter], done_[iter]]]).to_csv(
                        '../ExperimentData/Random-or-Non-random Analysis/innercollision_tartu_new_sd/dqn_6s_road1_' + current_eps + 'eps_' + file_name + '_0.2eps.csv',
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
            print("Start episode ", iteration)
            if iteration == 16:
                break
