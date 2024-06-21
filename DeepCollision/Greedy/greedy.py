import socket
import time

import pandas as pd

from utils import *

requests.post(
    "http://localhost:8933/LGSVL/LoadScene?scene=bd77ac3b-fbc3-41c3-a806-25915c777022&road_num=1")


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


action_space = get_action_space()['command']
scenario_space = get_action_space()['scenario_description']
N_ACTIONS = action_space['num']
N_STATES = get_environment_state().shape[0]
ENV_A_SHAPE = 0


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
    position = requests.get(
        "http://localhost:8933/LGSVL/Status/EGOVehicle/Position").json()
    position_space.append((position['x'], position['y'], position['z']))
    position_space_size = (position_space_size + 1) % check_num
    if len(position_space) == 5:
        start_pos = position_space[0]
        end_pos = position_space[4]
        position_space = []
        dis = pow(
            pow(start_pos[0] - end_pos[0], 2) + pow(start_pos[1] -
                                                    end_pos[1], 2) + pow(start_pos[2] - end_pos[2], 2),
            0.5)

        dis2 = start_pos[1] - end_pos[1]

        if dis < 0.15:
            judge = True

        if dis2 > 25:
            judge = True
    return judge


def calculate_reward(action_id):
    
    vioRate_list, obstacle_uid = execute_action(action_id)
    observation = get_environment_state()
    violation_rate = 0
    episode_done = judge_done()

    violation_rate = round(float(
        (requests.get("http://localhost:8933/LGSVL/Status/ViolationRate")).content.decode(
            encoding='utf-8')), 6)

    return observation, violation_rate, episode_done, vioRate_list, obstacle_uid


title = ["Episode", "Step", "State", "Action", "Violation Rate",
         "Collision_uid", "Violation Rate List" "Done"]
df_title = pd.DataFrame([title])
file_tag = str(int(time.time()))

def greedy(road_num, opt):
    i_episode = 0
    step = 0
    file_name = '../ExperimentData/Random-or-Non-random Analysis/Data Greedy/greedy_road{}_{}s.csv'.format(road_num, opt)
    df_title.to_csv(file_name, mode='w', header=False,
                    index=None)

    requests.post(
        "http://localhost:8933/LGSVL/LoadScene?scene=bd77ac3b-fbc3-41c3-a806-25915c777022&road_num=" + str(road_num))
    requests.post("http://localhost:8933/LGSVL/SaveState?ID={}".format(step))
    print("Start")
    while True:
        vioRate_max = 0
        max_api = 0
        for api in range(0, N_ACTIONS):
            try:
                _, violation_rate, done, vioRate_list, obstacle_uid = calculate_reward(api)
            except Exception as e:
                print(e)
            finally:
                print("Handled Exception")
            print("Action ", api)
            if violation_rate > vioRate_max:
                vioRate_max = violation_rate
                max_api = api
                
            print("Action", api, " Rollback")
            requests.post(
                "http://localhost:8933/LGSVL/RollBack?ID={}".format(step))
            print("After rollback")

        s = get_environment_state()
        print("Before implementing ", str(max_api))
        _, violation_rate, done, vioRate_list, obstacle_uid = calculate_reward(max_api)
        print("After implementing ", str(max_api))

        print('episode, step, api_id, violation rate, violation rate list, done: ',
              i_episode, step, max_api, violation_rate, vioRate_list, done)
        
        pd.DataFrame([[i_episode, step, s, max_api, violation_rate, obstacle_uid, vioRate_list, done]]).to_csv(
            file_name, mode='a', header=False, index=None)
        step += 1
        requests.post(
            "http://localhost:8933/LGSVL/SaveState?ID={}".format(step))

        if done:
            requests.post(
                "http://localhost:8933/LGSVL/LoadScene?scene=bd77ac3b-fbc3-41c3-a806-25915c777022&road_num=" + str(road_num))
            i_episode += 1
            step = 0
            requests.post(
                "http://localhost:8933/LGSVL/SaveState?ID={}".format(step))
            if i_episode == 16:
                break

if __name__ == '__main__':
    requests.post("http://localhost:8933/LGSVL/SetObTime?observation_time=6")
    greedy("1", "6")