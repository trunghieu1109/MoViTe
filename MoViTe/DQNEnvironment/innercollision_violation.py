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

from enum import Enum
from memory.utils import device, set_seed
from memory.buffer import ReplayBuffer, PrioritizedReplayBuffer, PrioritizedViolationReplayBuffer

from utils import *

current_eps = '220'

road_num = '1'  # the Road Number
second = '6'  # the experiment second
requests.post("http://localhost:8933/LGSVL/LoadScene?scene=aae03d2a-b7ca-4a88-9e41-9035287a12cc&road_num=" + road_num)
file_name = str(int(time.time()))

collide_with_obstacle = False
position_pre_obstacle_collision = None

prev_position = None
is_stopped = False

# get environment state
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
N_VIOLATIONS = 6

class Violation(Enum):
    NOT_STOP_FOR_PEDESTRIAN = 0
    PEDES_VEHICLE_COLLISION = 1
    SUDDEN_BRAKING = 2
    IMPROPER_PASSING = 3
    IMPROPER_LANE_CHANGING = 4
    RUNNING_ON_RED_LIGHT = 5

# 1. Not stop for pedestrian
# 2. Pedestrian / Vehicle collision
# 3. Improper turn
# 4. Sudden Braking
# 5. Improper Passing
# 6. Improper Lane changing
# 7. Running on signal

ENV_A_SHAPE = 0

print("Number of action: ", N_ACTIONS)
print("Number of state: ", N_STATES)

HyperParameter = dict(BATCH_SIZE=32, GAMMA=0.9, EPS_START=1, EPS_END=0.1, EPS_DECAY=6000, TARGET_UPDATE=100,
                      lr=3*1e-3, INITIAL_MEMORY=2000, MEMORY_SIZE=2000, SCHEDULER_UPDATE=100, WEIGHT_DECAY=1e-5,
                      LEARNING_RATE_DECAY=0.8)

print("MEMORY SIZE: ", HyperParameter["MEMORY_SIZE"])
print("BATCH SIZE: ", HyperParameter["BATCH_SIZE"])

class Net(nn.Module):
    def __init__(self, ):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(N_STATES, 1024)
        self.fc1.weight.data.normal_(0, 0.1)  # initialization
        self.bn1 = nn.BatchNorm1d(1024, track_running_stats = True)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc2.weight.data.normal_(0, 0.1)  # initialization
        self.bn2 = nn.BatchNorm1d(1024, track_running_stats = True)
        self.out = nn.Linear(1024, N_ACTIONS * N_VIOLATIONS)
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
        self.buffer_memory = PrioritizedViolationReplayBuffer(N_STATES, N_ACTIONS, N_VIOLATIONS, HyperParameter["MEMORY_SIZE"], 0.01, 0.7, 0.4)
        self.previous_weather_and_time = None

    def lr_lambda_(self, epoch):
        return HyperParameter['LEARNING_RATE_DECAY'] ** epoch

    def choose_action(self, x):
        x = torch.unsqueeze(torch.FloatTensor(x), 0)

        # update epsilon threshold
        eps_threshold = HyperParameter['EPS_END'] + (
                HyperParameter['EPS_START'] - HyperParameter['EPS_END']) * math.exp(
            -1. * self.steps_done / HyperParameter['EPS_DECAY'])

        eps_threshold = 0.2

        print("eps threshold:", eps_threshold)

        choose = ""

        action = None

        self.steps_done += 1

        # choose action greedy or by model
        if np.random.uniform() > eps_threshold:  # greedy
            choose = "by model"
            print("Choose by model from deepcollision_per")
            # get q value
            actions_value = self.eval_net.forward(x, False)
            # calculate return from violation's prob
            values = [actions_value[0][i * N_VIOLATIONS:i * N_VIOLATIONS + N_VIOLATIONS] for i in range(0, N_ACTIONS)]
            sqrt_values = [np.sqrt(np.sum(np.square(value.detach().numpy()))) for value in values]
            action_q_value = torch.as_tensor(sqrt_values) # new return
            action_q_value = np.reshape(action_q_value, (-1, N_ACTIONS))
            action = torch.max(action_q_value, 1)[1].data.numpy()
            action = action[0] if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)  # return the argmax index
            if action == self.previous_weather_and_time:
                print("Choose another action")
                topk_values, topk_indices = torch.topk(action_q_value, k=2)
                action = topk_indices[0][1].data.numpy()
        else:  # random
            choose = "randomly"
            print("Choose randomly")
            action = np.random.randint(0, N_ACTIONS)

            while action == self.previous_weather_and_time:
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
        # reward = reward.unsqueeze(1)

        action_segment = np.array([np.arange(a[0] * N_VIOLATIONS, (a[0] + 1) * N_VIOLATIONS) for a in action])
        action_segment = torch.as_tensor(action_segment)

        q_eval = self.eval_net.forward(state).gather(1, action_segment)  # shape (batch, 1)

        q_eval_next = self.eval_net.forward(next_state)

        q_action = []
        for batch in q_eval_next:
            sqrt_values_batch = [np.linalg.norm(batch[i * N_VIOLATIONS:(i + 1) * N_VIOLATIONS].detach().numpy()) for i in range(0, N_ACTIONS)]
            action_q_value_batch = np.reshape(torch.as_tensor(sqrt_values_batch), (-1, N_ACTIONS))
            action_batch = torch.max(action_q_value_batch, 1)[1].data.numpy()
            action = action_batch[0] if ENV_A_SHAPE == 0 else action_batch.reshape(ENV_A_SHAPE)  # return the
            q_action.append(action)

        q_value = self.target_net.forward(next_state).detach()  # detach from graph, don't backpropagate
        q_next = torch.as_tensor(np.zeros((HyperParameter['BATCH_SIZE'], N_VIOLATIONS)))
        for i in range(0, HyperParameter['BATCH_SIZE']):
            q_next[i] = q_value[i][q_action[i]*N_VIOLATIONS:(q_action[i] + 1) * N_VIOLATIONS]

        q_target = reward + HyperParameter['GAMMA'] * q_next  # shape (batch, 1)

        td_error_v = torch.abs(q_eval - q_target).detach()

        td_error = [np.sqrt(np.sum(np.square(td.detach().numpy()))) for td in td_error_v]

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

        self.buffer_memory.update_priorities(tree_idxs, td_error)

def execute_action(action_id):
    api = action_space[str(action_id)]
    print("Start action: ", api)
    response = requests.post(api)
    obstacle_uid = None
    violation_reward = None
    try:
        probability_list = response.json()['probability']
        obstacle_uid = response.json()['collision_uid']
        not_stop_pedes = response.json()['not_stop_pedes']
        sudden_braking = response.json()['sudden_braking']
        improper_passing = response.json()['improper_passing']
        improper_lane_changing = response.json()['improper_lane_changing']
        violation_reward = [not_stop_pedes, 0, sudden_braking, improper_passing, improper_lane_changing, 0]
    except Exception as e:
        print(e)
        probability_list = np.zeros((12, N_VIOLATIONS))
        violation_reward = [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0]

    return probability_list, obstacle_uid, violation_reward


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
            #position_space = []

        if abs(dis2) > 25:
            judge = True
            #position_space = []

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
    # global latitude_position
    # global position_space_size
    # global position_space
    # global collision_probability
    probability_list, obstacle_uid, violation_reward = execute_action(action_id)
    observation = get_environment_state()
    action_reward = np.zeros((N_VIOLATIONS))
    collision_probability = 0
    # episode_done = False
    # Reward is calculated based on collision probability.
    collision_info = (requests.get("http://localhost:8933/LGSVL/Status/CollisionInfo")).content.decode(
        encoding='utf-8')

    episode_done = judge_done()


    if collision_info != 'None':
        action_reward[Violation.PEDES_VEHICLE_COLLISION] = 1
        collision_probability = 1
    elif collision_info == "None":
        collision_probability = round(float(
            (requests.get("http://localhost:8933/LGSVL/Status/CollisionProbability")).content.decode(
                encoding='utf-8')), 6)

        if collision_probability < 0.2:
            # action_reward[Violation.PEDES_VEHICLE_COLLISION] = -1
            violation_reward[1] = -1    
        else:
            # action_reward[Violation.PEDES_VEHICLE_COLLISION] = collision_probability
            violation_reward[1] = collision_probability
    return observation, violation_reward, collision_probability, episode_done, collision_info, probability_list, obstacle_uid


title = ["Episode", "Step", "State", "Action", "Reward", "Collision_Probability", "Collision_Probability_List", "Action_Description", "Done"]
df_title = pd.DataFrame([title])

df_title.to_csv('../ExperimentData/InnerCollision_new_action_space_2000MS_' + second + 's_abs_compute_' + file_name + '_road' + road_num + '.csv', mode='w',
                header=False,
                index=None)

if __name__ == '__main__':
    '''
    Establish client to connect to Apollo
    '''

    dqn = DQN()

    if int(road_num) >= 2:
        dqn.eval_net.load_state_dict(torch.load('./model/InnerCollision_new_action_space_2000MS_'+second+'s/eval_net_600_road'+str(int(road_num)-1)+'.pt'))


    if current_eps != '':
        print("Continue at episode: " + current_eps)

        with open('./model/innercollision_tune_violation/rl_network_' + current_eps + '_road' + road_num + '.pkl', "rb") as file:
            dqn = pickle.load(file)
        # print(dqn.buffer_memory.real_size, dqn.memory_counter, dqn.steps_done, dqn.learn_step_counter)
        dqn.eval_net.load_state_dict(torch.load('./model/innercollision_tune_violation/eval_net_' + current_eps + '_road' + road_num + '.pt'))
        dqn.target_net.load_state_dict(torch.load('./model/innercollision_tune_violation/target_net_' + current_eps + '_road' + road_num + '.pt'))
        # restore memory buffer
        with open('./model/innercollision_tune_violation/memory_buffer_' + current_eps + '_road' + road_num + '.pkl', "rb") as file:
            dqn.buffer_memory = pickle.load(file)

        print(dqn.buffer_memory.real_size, dqn.learn_step_counter, dqn.steps_done)

    print('\nCollecting experience...')
    road_num_int = int(road_num)

    while road_num_int <= 1:
        road_num = str(road_num_int)

        df_title = pd.DataFrame([title])
        file_name = str(int(time.time()))
        pd.DataFrame([["Learning Step", "Learning Rate", "Loss"]]).to_csv('./loss_log/loss_log_' + file_name + '.csv',
                                                         mode='w', header=False, index=None)

        requests.post("http://localhost:8933/LGSVL/SetObTime?observation_time=" + '6')


        for i_episode in range(220, 400):
            print('------------------------------------------------------')
            print('+                 Road, Episode: ', road_num_int, i_episode, '                +')
            print('------------------------------------------------------')
            # if i_episode == 1:
            #    requests.post("http://localhost:8933/LGSVL/SaveTransform")
            requests.post("http://localhost:8933/LGSVL/LoadScene?scene=aae03d2a-b7ca-4a88-9e41-9035287a12cc&road_num=" + road_num)

            s = get_environment_state()
            # s = format_state(get_environment_state())
            ep_r = np.zeros(6)
            step = 0
            while True:
                # env.render()
                action, _ = dqn.choose_action(s)
                # print("Action to get description: ", action)
                action_description = scenario_space[str(action)]
                # take action
                s_, reward, collision_probability, done, info, list_prob, obstacle_uid = calculate_reward(action)

                dis__ = 100

                if prev_position:
                    dis__x = prev_position['x'] - s_[0]
                    dis__y = prev_position['y'] - s_[1]
                    dis__z = prev_position['z'] - s_[2]
                    dis__ = math.sqrt(dis__x ** 2 + dis__y ** 2 + dis__z ** 2)

                if dis__ <= 2:
                    if not is_stopped:
                        dqn.store_transition(s, action, reward, s_, done)
                    else:
                        dqn.steps_done -= 1
                    is_stopped = True
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
                    dis_x = position_pre_obstacle_collision['x'] - s_[0]
                    dis_y = position_pre_obstacle_collision['y'] - s_[1]
                    dis_z = position_pre_obstacle_collision['z'] - s_[2]
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

                position_pre_obstacle_collision = {
                    'x': s_[0],
                    'y': s_[1],
                    'z': s_[2],
                }

                # print(s, action, reward, s_, done)
                print('>>>>>step, action, reward, collision_probability, action_description, done: ', step, action,
                      reward, round(collision_probability, 6),
                      "<" + action_description + ">",
                      done)
                pd.DataFrame(
                    [[i_episode, step, s, action, reward, collision_probability, list_prob, action_description, done]]).to_csv(
                    '../ExperimentData/InnerCollision_new_action_space_2000MS_' + second + 's_abs_compute_' + file_name + '_road' + road_num + '.csv',
                    mode='a',
                    header=False, index=None)

                # pd.DataFrame([[per_confi, pred_confi, reward]]).to_csv(
                #     './log/per_pred_reward_' + file_name + '.csv', mode='a', header=False, index=None
                # )

                ep_r += reward
                if dqn.memory_counter > HyperParameter['MEMORY_SIZE']:
                    dqn.learn()
                    # if done:
                    #     print('Ep: ', i_episode,
                    #           '| Ep_r: ', round(ep_r, 2))

                if (i_episode + 1) % 10 == 0:
                    # print('save')
                    # print(dqn.eval_net.state_dict())
                    # print(dqn.target_net.state_dict())
                    torch.save(dqn.eval_net.state_dict(),
                               './model/innercollision_tune_violation/eval_net_' + str(
                                   i_episode + 1) + '_road' + road_num + '.pt')
                    torch.save(dqn.target_net.state_dict(),
                               './model/innercollision_tune_violation/target_net_' + str(
                                   i_episode + 1) + '_road' + road_num + '.pt')

                    with open('./model/innercollision_tune_violation/memory_buffer_' + str(
                                   i_episode + 1) + '_road' + road_num + '.pkl', "wb") as file:
                        pickle.dump(dqn.buffer_memory, file)

                    with open('./model/innercollision_tune_violation/rl_network_' + str(
                                   i_episode + 1) + '_road' + road_num + '.pkl', "wb") as file:
                        pickle.dump(dqn, file)

                if done:
                    # comm_apollo.send(repr('1').encode())
                    collide_with_obstacle = False
                    position_pre_obstacle_collision = None
                    prev_position = None
                    is_stopped = False
                    dqn.previous_weather_and_time = None
                    position_space = []
                    break
                step += 1
                s = s_
        road_num_int = road_num_int + 1