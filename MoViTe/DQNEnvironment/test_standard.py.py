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

# metrics threshold
ETTC_threshold = 7 # (s)
DTO_threshold = 10 # (m)
JERK_threshold = 5 # (m/s^2)

# continue training episode
current_eps = ''
reuse_mem_eps = ''
start_eps = '0'
end_eps = '200'

# map variable (road num, otp, map id, v.v)
road_num = '3'  # the Road Number
second = '6'  # the experiment second
scene = '12da60a7-2fc9-474d-a62a-5cc08cb97fe8'
requests.post(f"http://localhost:8933/LGSVL/LoadScene?scene={scene}&road_num=" + road_num)
file_name = str(int(time.time()))

# destination location
goal = [
    -208.2, 
    10.2, 
    -181.6
]

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

action_space = get_action_space()['command']
scenario_space = get_action_space()['scenario_description']
N_ACTIONS = action_space['num']
N_STATES = get_environment_state().shape[0]
ENV_A_SHAPE = 0

print("Number of action: ", N_ACTIONS)
print("Number of state: ", N_STATES)

HyperParameter = dict(BATCH_SIZE=64, GAMMA=0.9, EPS_START=1, EPS_END=0.1, EPS_DECAY=10000, TARGET_UPDATE=100,
                      lr=1e-2, INITIAL_MEMORY=3500, MEMORY_SIZE=3500, SCHEDULER_UPDATE=100, WEIGHT_DECAY=1e-5,
                      LEARNING_RATE_DECAY=0.8)

print("MEMORY SIZE: ", HyperParameter["MEMORY_SIZE"])
print("BATCH SIZE: ", HyperParameter["BATCH_SIZE"])
print("EPS DECAY: ", HyperParameter["EPS_DECAY"])

class Net(nn.Module):
    def __init__(self, ):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(N_STATES, 1024)
        self.fc1.weight.data.normal_(0, 0.1)  # initialization
        self.bn1 = nn.BatchNorm1d(1024, track_running_stats = True)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc2.weight.data.normal_(0, 0.1)  # initialization
        self.bn2 = nn.BatchNorm1d(1024, track_running_stats = True)
        self.out = nn.Linear(1024, N_ACTIONS)
        self.out.weight.data.normal_(0, 0.1)  # initialization

    def forward(self, x, is_training=True):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
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
        
        self.num_of_npc_action = 32
        self.num_of_action = 45
        self.action_chosen_prob = [1 / self.num_of_action] * self.num_of_action
        self.npc_action_chosen_prob = [1 / self.num_of_npc_action] * self.num_of_npc_action

    def lr_lambda_(self, epoch):
        return HyperParameter['LEARNING_RATE_DECAY'] ** epoch

    def update_action_prob(self, action):
        # update the random choosing probability of normal action
        
        prob = self.action_chosen_prob[action]
        
        reduced_amount = prob * 0.05
        
        for i in range(0, self.num_of_action):
            if i == action:
                self.action_chosen_prob[i] -= reduced_amount
            else:
                self.action_chosen_prob[i] += reduced_amount / (self.num_of_action - 1)    
                
    def update_npc_action_prob(self, action):
        # update the random choosing probability of npc, pedes action
        
        prob = self.npc_action_chosen_prob[action]
        
        reduced_amount = prob * 0.05
        
        for i in range(0, self.num_of_npc_action):
            if i == action:
                self.npc_action_chosen_prob[i] -= reduced_amount
            else:
                self.npc_action_chosen_prob[i] += reduced_amount / (self.num_of_npc_action - 1)    

    def choose_action(self, x, current_step, prev_step):
        x = torch.unsqueeze(torch.FloatTensor(x), 0)
        eps_threshold = HyperParameter['EPS_END'] + (
                HyperParameter['EPS_START'] - HyperParameter['EPS_END']) * math.exp(
            -1. * self.steps_done / HyperParameter['EPS_DECAY'])

        print("eps threshold:", eps_threshold)
        
        choose = ""
        
        action = None
            
        self.steps_done += 1
        
        if self.steps_done > HyperParameter["EPS_DECAY"]:
            eps_threshold = 0.1
        
        isGreedy = (np.random.uniform() > eps_threshold)
        
        if self.steps_done <= HyperParameter["MEMORY_SIZE"]:
            isGreedy = False
        
        if isGreedy:  # greedy
            choose = "by model"
            print("Choose by model")
            # print("Let choose action by model")
            actions_value = self.eval_net.forward(x, False)

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
                    topk_values, topk_indices = torch.topk(actions_value, k=15)
                    cnt = 1
                    while cnt <= 14: # choose other npc action
                        
                        action = topk_indices[0][cnt].data.numpy()
                        
                        if not (0 <= action and action <= 12):
                            break
                        
                        cnt += 1
            
        else:  # random
            choose = "randomly"
            print("Choose randomly")
            
            action = np.random.choice(self.num_of_action, size=1, replace = False, p=self.action_chosen_prob)
            action = action[0]
            
            is_weather_time_action = (0 <= action and action <= 12)
            
            if is_weather_time_action:
                if current_step - prev_step >= 5:
                    if action == self.previous_weather_and_time:
                        action = np.random.choice(self.num_of_action, size=1, replace = False, p=self.action_chosen_prob)
                        action = action[0]
                else:
                    action = np.random.choice(self.num_of_npc_action, size=1, replace = False, p=self.npc_action_chosen_prob) + (self.num_of_action - self.num_of_npc_action)
                    action = action[0]

            self.update_action_prob(action)
            
            if 12 < action:
                self.update_npc_action_prob(action - (self.num_of_action - self.num_of_npc_action))
            
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
    generated_uid = None
    print(response)
    try:
        proC_list = response.json()['probability']
        obstacle_uid = response.json()['collision_uid']
        generated_uid = response.json()['generated_uid']
    except Exception as e:
        print(e)
        proC_list = [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0]
        
    return proC_list, obstacle_uid, generated_uid

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
    
    proC_list, obstacle_uid, generated_uid = execute_action(action_id)
    observation = get_environment_state()
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
            
    return observation, action_reward, collision_probability, episode_done, proC_list, obstacle_uid, collision_info, col_uid, generated_uid

def continue_training(folder_name, current_eps, road_num):
    
    dqn = DQN()
    
    print("Continue at episode: " + current_eps)
        
    with open(folder_name + 'rl_network_' + current_eps + '_road' + road_num + '.pkl', "rb") as file:
        dqn = pickle.load(file)
    dqn.eval_net.load_state_dict(torch.load(folder_name + 'eval_net_' + current_eps + '_road' + road_num + '.pt'))
    dqn.target_net.load_state_dict(torch.load(folder_name + 'target_net_' + current_eps + '_road' + road_num + '.pt'))
    with open(folder_name + 'memory_buffer_' + current_eps + '_road' + road_num + '.pkl', "rb") as file:
        dqn.buffer_memory = pickle.load(file)       
            
    print(dqn.buffer_memory.real_size, dqn.learn_step_counter, dqn.steps_done)
    
    return dqn

def reuse_buffer_memory(reuse_folder, reuse_mem_eps, road_num):
    
    dqn = DQN()
    
    print("Reuse memory buffer from episode: " + reuse_mem_eps)
        
    with open(reuse_folder + 'rl_network_' + reuse_mem_eps + '_road' + road_num + '.pkl', "rb") as file:
        dqn = pickle.load(file)
    with open(reuse_folder + 'memory_buffer_' + reuse_mem_eps + '_road' + road_num + '.pkl', "rb") as file:
        dqn.buffer_memory = pickle.load(file)       
            
    print(dqn.buffer_memory.real_size, dqn.learn_step_counter, dqn.steps_done)
    print(dqn.action_chosen_prob)
    print(dqn.npc_action_chosen_prob)
    
    return dqn

title = ["Episode", "Step", "State", "Action", "Reward", "Collision Probability", "Collision Probability List", "Action_Description", "Done"]
df_title = pd.DataFrame([title])

if __name__ == '__main__':
    '''
    Establish client to connect to Apollo
    '''

    dqn = DQN()
        
    folder_name = './model/crisis_sanfrancisco_road3/'
    reuse_folder = './model/short_paper_/'
    
    print("Folder name: ", folder_name)
    
    if not os.path.isdir(folder_name):
        print("Create dir", folder_name)
        os.makedirs(folder_name)
        
    if current_eps != '':
        dqn = continue_training(folder_name, current_eps, road_num)
        
    if reuse_mem_eps != '':
        dqn = reuse_buffer_memory(reuse_folder, reuse_mem_eps, road_num)
        
    print('\nCollecting experience...')
    road_num_int = int(road_num)

    print("Road num: ", road_num_int)
    road_num = str(road_num_int)

    df_title = pd.DataFrame([title])
    file_name = str(int(time.time()))
    log_name = '../ExperimentData/crisis_' + road_num + '_' + file_name + '.csv'
    
    df_title.to_csv(log_name, mode='w', header=False, index=None)

    requests.post("http://localhost:8933/LGSVL/SetObTime?observation_time=" + '6')
    
    collide_with_obstacle = False
    pre_pos_obstacle_collision = None
    previous_weather_and_time_step = -5
    prev_position = None
    is_stopped = False
    prev_collision_info = ""
    prev_collision_uid = ""
    collision_position = {
        'x': 0, 
        'y': 0, 
        'z': 0
    }
    
    step_after_collision = -1
    
    uid_list = {}
        
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
            s_, reward, proC, done, proC_list, obstacle_uid, collision_info, col_uid, generated_uid = calculate_reward(action)            
            
            # check if there is repeated collision or ego is stopped
            dis_to_prev_col = math.sqrt((s_[0] - collision_position['x']) ** 2 + (s_[1] - collision_position['y']) ** 2 + (s_[2] - collision_position['z']) ** 2)
            
            if collision_info != 'None':
                isCollision_ = False
            
                if prev_collision_uid != col_uid:
                    isCollision_ = True
                else:
                    if dis_to_prev_col >= 3:
                        isCollision_ = True
                    else:
                        reward = 0
                        
                if isCollision_:
                    collision_position = {
                        'x': s_[0],
                        'y': s_[1],
                        'z': s_[2]
                    }
                    prev_collision_info = collision_info
                    prev_collision_uid = col_uid
                    step_after_collision = 0
                else:
                    if step_after_collision >= 0:
                        step_after_collision += 1
                        if step_after_collision >= 2:
                            if dis_to_prev_col < 3:
                                done = True
                            else:
                                step_after_collision = -1    
                                            
            else:
                if step_after_collision >= 0:
                    step_after_collision += 1
                    if step_after_collision >= 2:
                        if dis_to_prev_col < 3:
                            done = True
                        else:
                            step_after_collision = -1    

            print("Reward: ", reward)     
            
            
            # Delayed Reward
            if generated_uid:
                uid_list[generated_uid] = dqn.buffer_memory.count
                
            if (collision_info == 'pedestrian' or collision_info == 'npc_vehicle') and col_uid != generated_uid:
                dqn.buffer_memory.reward[uid_list[col_uid]] = torch.as_tensor(reward)
                reward = reward * 2/3
            
            # Calculate distance to previous step and goal. If this distance is small enough, this episode should be interrupted
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
            
            print('>>>>>step, action, reward, collision_probability, action_description, done: ', step, action,
                    reward, round(proC, 6), "<" + action_description + ">", done)
            
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
                prev_collision_info = ""
                prev_collision_uid = ""
                collision_position = {
                    'x': 0,
                    'y': 0,
                    'z': 0
                }
                step_after_collision = -1
                break
            step += 1
            s = s_