import requests
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import time
import socket
import math
import pickle
import os

from memory.utils import device, set_seed
from memory.buffer import ReplayBuffer, PrioritizedReplayBuffer

from utils import *

ETTC_threshold = 7 # (s)
DTO_threshold = 10 # (m)
JERK_threshold = 5 # (m/s^2)

current_eps = ''
start_eps = '0'
end_eps = '200'

road_num = '1'  # the Road Number
second = '6'  # the experiment second
scene = 'aae03d2a-b7ca-4a88-9e41-9035287a12cc'
requests.post(f"http://localhost:8933/LGSVL/LoadScene?scene={scene}&road_num=" + road_num)
file_name = str(int(time.time()))

goal = [
    348.2, 
    -7.5, 
    -64.4
]

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

action_space = get_action_space()['command']
scenario_space = get_action_space()['scenario_description']
N_ACTIONS = action_space['num']
N_STATES = get_environment_state().shape[0]
ENV_A_SHAPE = 0

print("Number of action: ", N_ACTIONS)
print("Number of state: ", N_STATES)

HyperParameter = dict(BATCH_SIZE=32, GAMMA=0.9, EPS_START=1, EPS_END=0.1, EPS_DECAY=6000, TARGET_UPDATE=100,
                      lr=3*1e-3, INITIAL_MEMORY=2000, MEMORY_SIZE=2000, SCHEDULER_UPDATE=100, WEIGHT_DECAY=1e-5,
                      LEARNING_RATE_DECAY=0.8)

print("MEMORY SIZE: ", HyperParameter["MEMORY_SIZE"])
print("BATCH SIZE: ", HyperParameter["BATCH_SIZE"])
print("EPS DECAY: ", HyperParameter["EPS_DECAY"])

class Net(nn.Module):
    def __init__(self, ):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(N_STATES, 512)
        self.fc1.weight.data.normal_(0, 0.1)  # initialization
        self.bn1 = nn.BatchNorm1d(512, track_running_stats = True)
        self.fc2 = nn.Linear(512, 512)
        self.fc2.weight.data.normal_(0, 0.1)  # initialization
        self.bn2 = nn.BatchNorm1d(512, track_running_stats = True)
        self.out = nn.Linear(512, N_ACTIONS)
        self.out.weight.data.normal_(0, 0.1)  # initialization

    def forward(self, x, is_training=True):
        x = self.fc1(x)
        x = F.relu(x)
        if is_training:
            x = self.bn1(x)
        x = self.fc2(x)
        x = F.relu(x)
        if is_training:
            x = self.bn2(x)
        actions_value = self.out(x)
        return actions_value

class DQN(object):
    def __init__(self):
        self.eval_net, self.target_net = Net(), Net()
        # learning rate decay
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=HyperParameter['lr'], weight_decay=HyperParameter['WEIGHT_DECAY'])
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=self.lr_lambda_)
        
        self.steps_done = 0
        self.memory_counter = 0
        self.learn_step_counter = 0
        self.buffer_memory = PrioritizedReplayBuffer(N_STATES, N_ACTIONS, HyperParameter["MEMORY_SIZE"], 0.01, 0.7, 0.4)
        self.previous_weather_and_time = None

    def lr_lambda_(self, epoch):
        return HyperParameter['LEARNING_RATE_DECAY'] ** epoch

    def choose_action(self, x, current_step, prev_step):
        print("Current Step: ", current_step)
        x = torch.unsqueeze(torch.FloatTensor(x), 0)
        eps_threshold = HyperParameter['EPS_END'] + (
                HyperParameter['EPS_START'] - HyperParameter['EPS_END']) * math.exp(
            -1. * self.steps_done / HyperParameter['EPS_DECAY'])
        
        print("eps threshold:", eps_threshold)
        
        choose = ""
        
        action = None
            
        self.steps_done += 1
        if np.random.uniform() > eps_threshold:  # greedy
            choose = "by model"
            print("Choose by model")
            # print("Let choose action by model")
            actions_value = self.eval_net.forward(x, False)
            # print(actions_value)
            # print('action value: ', actions_value, actions_value.shape)
            action = torch.max(actions_value, 1)[1].data.numpy()
            # print(actions_value.data)
            action = action[0] if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)  # return the argmax index
            
            is_weather_time_action = (0 <= action and action <= 12)
            
            if is_weather_time_action:
                if current_step - prev_step >= 5:
                    if action == self.previous_weather_and_time:
                        topk_values, topk_indices = torch.topk(actions_value, k=2)
                        action = topk_indices[0][1].data.numpy()
                else:
                    topk_values, topk_indices = torch.topk(actions_value, k=6)
                    cnt = 1
                    while cnt <= 5:
                        action = topk_indices[0][cnt].data.numpy()
                        # print("action: ", action)
                        if not (0 <= action and action <= 12):
                            break
                        
                        cnt += 1
            
            # print(action)
        else:  # random
            choose = "randomly"
            print("Choose randomly")
            action = np.random.randint(0, N_ACTIONS)
            
            is_weather_time_action = (0 <= action and action <= 12)
            
            if is_weather_time_action:
                if current_step - prev_step >= 5:
                    if action == self.previous_weather_and_time:
                        action = np.random.randint(0, N_ACTIONS)
                else:
                    action = np.random.randint(0, N_ACTIONS)
            
            action = action if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)
        
        if 0 <= action and action <= 12:
            self.previous_weather_and_time = action    
        
        return action, choose

    def store_transition(self, s, a, r, s_, done):
        self.memory_counter += 1
        self.buffer_memory.add((s, a, r, s_, int(done)))

    def learn(self):
        
        print(80*'*')
        
        print("Start Learning Deep Q Network")
        
        print(80*'*')
        
        # target parameter update
        if self.learn_step_counter % HyperParameter['TARGET_UPDATE'] == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        # sample batch transitions
        batch, weights, tree_idxs = self.buffer_memory.sample(HyperParameter["BATCH_SIZE"])
        state, action, reward, next_state, done = batch        
        action = action.max(dim=1).values.unsqueeze(1).to(torch.int64)
        reward = reward.unsqueeze(1)
        q_eval = self.eval_net.forward(state).gather(1, action)  # shape (batch, 1)
        q_action = self.eval_net.forward(next_state).max(dim=1).indices
        q_value = self.target_net.forward(next_state).detach()  # detach from graph, don't backpropagate       
        q_next = q_value.max(dim=1).values
        for i in range(0, HyperParameter['BATCH_SIZE']):
            q_next[i] = q_value[i][q_action[i]]
            
        q_next = q_next.view(-1, 1)
        
        q_target = reward + HyperParameter['GAMMA'] * q_next  # shape (batch, 1)
        
        td_error = torch.abs(q_eval - q_target).detach()
        loss = torch.mean(torch.abs(q_eval - q_target) ** 2 * weights)
        
        pd.DataFrame([[self.learn_step_counter, self.optimizer.param_groups[0]['lr'], loss.item()]]).to_csv('./loss_log/loss_log_' + file_name + '.csv', 
                                                         mode='a', header=False, index=None)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # update learning rate
        if self.learn_step_counter % HyperParameter['SCHEDULER_UPDATE'] == 0:
            print("Updating learning rate")
            self.scheduler.step()
        
        self.buffer_memory.update_priorities(tree_idxs, td_error.numpy())
    
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
    # return proC_list, DTO_list, ETTC_list, JERK_list, obstacle_uid

check_num = 5  # here depend on observation time: 2s->10, 4s->6, 6s->

position_space = []
position_space_size = 0

def judge_done():
    global position_space_size
    global position_space
    judge = False
    position = requests.get("http://localhost:8933/LGSVL/Status/EGOVehicle/Position").json()
    print("Ego's position: ", position)
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
    elif collision_info == "None":
        collision_probability = round(float(
            (requests.get("http://localhost:8933/LGSVL/Status/CollisionProbability")).content.decode(
                encoding='utf-8')), 6)
        if collision_probability < 0.2:
            collision_reward = -1
        else:
            collision_reward = collision_probability
        
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
            
    return observation, action_reward, collision_probability, episode_done, proC_list, obstacle_uid
    # return observation, action_reward, collision_probability, DTO, ETTC, JERK, episode_done, proC_list, DTO_list, ETTC_list, JERK_list, obstacle_uid

# title = ["Episode", "Step", "State", "Action", "Reward", "Collision Probability", "Collision Probability List", 
#          "Distance To Obstacles", "Estimated Time To Collision", "JERK", "DTO_list", "ETTC_list", "JERK_list", "Action_Description", "Done"]
title = ["Episode", "Step", "State", "Action", "Reward", "Collision Probability", "Collision Probability List", "Action_Description", "Done"]
df_title = pd.DataFrame([title])

if __name__ == '__main__':
    '''
    Establish client to connect to Apollo
    '''

    dqn = DQN()
        
    folder_name = './model/short_paper_borregasave_internal_feature/'
    
    print("Folder name: ", folder_name)
    
    if not os.path.isdir(folder_name):
        print("Create dir", folder_name)
        os.makedirs(folder_name)
        
    if current_eps != '':
        print("Continue at episode: " + current_eps)
        
        with open(folder_name + 'rl_network_' + current_eps + '_road' + road_num + '.pkl', "rb") as file:
            dqn = pickle.load(file)
        dqn.eval_net.load_state_dict(torch.load(folder_name + 'eval_net_' + current_eps + '_road' + road_num + '.pt'))
        dqn.target_net.load_state_dict(torch.load(folder_name + 'target_net_' + current_eps + '_road' + road_num + '.pt'))
        with open(folder_name + 'memory_buffer_' + current_eps + '_road' + road_num + '.pkl', "rb") as file:
            dqn.buffer_memory = pickle.load(file)       
            
        print(dqn.buffer_memory.real_size, dqn.learn_step_counter, dqn.steps_done)
        
    print('\nCollecting experience...')
    road_num_int = int(road_num)

    print("Road num: ", road_num_int)
    road_num = str(road_num_int)

    df_title = pd.DataFrame([title])
    file_name = str(int(time.time()))
    log_name = '../ExperimentData/short_paper_borregasave_internal_feature_road' + road_num + '_' + file_name + '.csv'
    
    df_title.to_csv(log_name, mode='w', header=False, index=None)

    requests.post("http://localhost:8933/LGSVL/SetObTime?observation_time=" + '6')
    
    collide_with_obstacle = False
    pre_pos_obstacle_collision = None
    previous_weather_and_time_step = -5
    prev_position = None
    is_stopped = False
        
    for i_episode in range(int(start_eps), int(end_eps)):
        print('------------------------------------------------------')
        print('+                 Road, Episode: ', road_num_int, i_episode, '                +')
        print('------------------------------------------------------')
        requests.post(f"http://localhost:8933/LGSVL/LoadScene?scene={scene}&road_num=" + road_num)

        s = get_environment_state()
        # print("Environment state: ", s)
        ep_r = 0
        step = 0
        while True:
            current_step = step
            action, _ = dqn.choose_action(s, current_step, previous_weather_and_time_step)
            
            if 0 <= action and action <= 12:
                previous_weather_and_time_step = step
                
            action_description = scenario_space[str(action)]
            
            # print("Action chosen: ", action, action_description)
            # take action
            s_, reward, proC, done, proC_list, obstacle_uid = calculate_reward(action)
            # s_, reward, proC, DTO, ETTC, JERK, done, proC_list, DTO_list, ETTC_list, JERK_list, obstacle_uid = calculate_reward(action)
            
            print("Reward: ", reward)
            
            dis__ = 100
            
            if prev_position:
                dis__x = prev_position['x'] - s_[0]
                dis__y = prev_position['y'] - s_[1]
                dis__z = prev_position['z'] - s_[2]
                dis__ = math.sqrt(dis__x ** 2 + dis__y ** 2 + dis__z ** 2) 
                
            finished = False
            dis_to_goal = 100
            dis_to_goal_x = s_[0] - goal[0]
            dis_to_goal_y = s_[1] - goal[1]
            dis_to_goal_z = s_[2] - goal[2]
            dis_to_goal = math.sqrt(dis_to_goal_x ** 2 + dis_to_goal_y ** 2 + dis_to_goal_z ** 2) 
            
            if dis__ <= 2 and dis_to_goal <= 5:
                if not is_stopped:
                    dqn.store_transition(s, action, reward, s_, done)
                else:
                    print("Don't save this transition into replay buffer")
                    dqn.steps_done -= 1
                is_stopped = True
                done = True
            else:
                is_stopped = False
                dqn.store_transition(s, action, reward, s_, done)
            
            prev_position = {
                'x': s_[0],
                'y': s_[1],
                'z': s_[2]
            }
            
            # Consider whether colliding to obstacle (signal, static obstacle, v.v) or not
            if collide_with_obstacle == True:
                dis_x = pre_pos_obstacle_collision['x'] - s_[0]
                dis_y = pre_pos_obstacle_collision['y'] - s_[1]
                dis_z = pre_pos_obstacle_collision['z'] - s_[2]
                dis_ = math.sqrt(dis_x ** 2 + dis_y ** 2 + dis_z ** 2)
                
                if dis_ <= 2:
                    print(80*'*')
                    print("Stop because of colliding to obstacles")
                    print(80*'*')
                    done = True
            
            if obstacle_uid == 'OBSTACLE':
                print(80*'*')
                print("Colliding with obstalces")
                print(80*'*')
                collide_with_obstacle = True
            else:
                collide_with_obstacle = False
            
            pre_pos_obstacle_collision = {
                'x': s_[0],
                'y': s_[1],
                'z': s_[2],
            }

            # print('>>>>>step, action, reward, collision_probability, DTO, ETTC, JERK, action_description, done: ', step, action,
            #         reward, round(proC, 6), round(DTO, 6), round(ETTC, 6), round(JERK, 6),
            #         "<" + action_description + ">",
            #         done)
            
            print('>>>>>step, action, reward, collision_probability, action_description, done: ', step, action,
                    reward, round(proC, 6), "<" + action_description + ">", done)
            
            # pd.DataFrame(
            #     [[i_episode, step, s, action, reward, proC, proC_list, DTO, ETTC, JERK, DTO_list, ETTC_list, JERK_list, action_description, done]]).to_csv(
            #     log_name,
            #     mode='a',
            #     header=False, index=None)
            
            pd.DataFrame(
                [[i_episode, step, s, action, reward, proC, proC_list, action_description, done]]).to_csv(
                log_name,
                mode='a',
                header=False, index=None)

            ep_r += reward
            if dqn.memory_counter > HyperParameter['MEMORY_SIZE']:
                dqn.learn()
                if done:
                    print('Ep: ', i_episode,
                            '| Ep_r: ', round(ep_r, 2))

            if (i_episode + 1) % 5 == 0:
                torch.save(dqn.eval_net.state_dict(),
                            folder_name + 'eval_net_' + str(
                                i_episode + 1) + '_road' + road_num + '.pt')
                torch.save(dqn.target_net.state_dict(),
                            folder_name + 'target_net_' + str(
                                i_episode + 1) + '_road' + road_num + '.pt')
                
                with open(folder_name + 'memory_buffer_' + str(
                                i_episode + 1) + '_road' + road_num + '.pkl', "wb") as file:
                    pickle.dump(dqn.buffer_memory, file)
                    
                with open(folder_name + 'rl_network_' + str(
                                i_episode + 1) + '_road' + road_num + '.pkl', "wb") as file:
                    pickle.dump(dqn, file)
                
            if done:
                print("Restart episode")
                collide_with_obstacle = False
                pre_pos_obstacle_collision = None
                prev_position = None
                is_stopped = False
                dqn.previous_weather_and_time = None
                position_space = []
                current_step = 0
                previous_weather_and_time_step = -5
                break
            step += 1
            s = s_
