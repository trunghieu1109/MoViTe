import math
import numpy as np
import random
from lgsvl.agent import NpcVehicle
from numba import jit
from enum import Enum

N_VIOLATIONS = 6

NOT_STOP_FOR_PEDESTRIAN = 0
PEDES_VEHICLE_COLLISION = 1
SUDDEN_BRAKING = 2
IMPROPER_PASSING = 3
IMPROPER_LANE_CHANGING = 4
RUNNING_ON_RED_LIGHT = 5

pedestrian = [
    "Bob",
    "EntrepreneurFemale",
    "Howard",
    "Johny",
    "Pamela",
    "Presley",
    "Robin",
    "Stephen",
    "Zoe"
]

print("Bob" in pedestrian)

# while True:
#     print(random.randint(0, 8))

npc_vehicle = {
    "Sedan",
    "SUV",
    "Jeep",
    "Hatchback",
    "SchoolBus",
    "BoxTruck"
}


#
# for v in pedestrian:
#     print(v, pedestrian[v])

@jit(nopython=True, fastmath=True)
def calculate_angle_tan(k1, k2):
    if k1 == k2:
        k2 = k2 - 0.0001
    tan_theta = abs((k1 - k2) / (1 + k1 * k2))
    theta = np.arctan(tan_theta)
    return theta

@jit(nopython=True, fastmath=True)
def calculate_angle(vector1, vector2):
    cos_theta = (vector1[0] * vector2[0] + vector1[1] * vector2[1] + vector1[2] * vector2[2]) / \
                ((math.sqrt(vector1[0] * vector1[0] + vector1[1] * vector1[1] + vector1[2] * vector1[2]) *
                  math.sqrt(vector2[0] * vector2[0] + vector2[1] * vector2[1] + vector2[2] * vector2[2])))
    theta = np.arccos(cos_theta)
    # print(cos_theta)
    return theta

@jit(nopython=True, fastmath=True)
def calculate_distance(vector1, vector2):
    distance = math.sqrt(pow(vector1[0] - vector2[0], 2) + pow(vector1[1] - vector2[1], 2) + pow(vector1[2] - vector2[2], 2))
    return distance


# Calculate safe distance
@jit(nopython=True, fastmath=True)
def calculate_safe_distance(speed, u):
    safe_distance = (speed * speed) / (2 * 9.8 * u)
    return safe_distance


# Calculate collision probability
@jit(nopython=True, fastmath=True)
def calculate_collision_probability(safe_distance, current_distance):
    collision_probability = None
    if current_distance >= safe_distance:
        collision_probability = 0
    elif current_distance < safe_distance:
        collision_probability = (safe_distance - current_distance) / safe_distance
    return collision_probability

@jit(nopython=True, fastmath=True)
def get_collision_probability_2(current_distance, ego_rotation, agent_rotation, ego_position, 
                                agent_position, break_distance, z_axis):
    probability1 = probability2 = probability3 = 0
    if current_distance > 40:
        return 0
    if ego_rotation[1] - 10 < agent_rotation[1] < ego_rotation[1] + 10:
        # In this case, we can believe the ego vehicle and obstacle are on the same direction.
        vector = agent_position - ego_position
        if ego_rotation[1] - 10 < calculate_angle(vector, z_axis) < ego_rotation[1] + 10:
            # In this case, we can believe the ego vehicle and obstacle are driving on the same lane.
            safe_distance = break_distance
            probability1 = calculate_collision_probability(safe_distance, current_distance)
        else:
            # In this case, the ego vehicle and obstacle are not on the same lane. They are on two parallel lanes.
            safe_distance = 1
            probability2 = calculate_collision_probability(safe_distance, current_distance)
    else:
        # In this case, the ego vehicle and obstacle are either on the same direction or the same lane.
        safe_distance = 5
        probability3 = calculate_collision_probability(safe_distance, current_distance)
    new_probability = probability1 + (1 - probability1) * 0.2 * probability2 + \
                        (1 - probability1) * 0.8 * probability3
                        
    return new_probability
    

def get_collision_probability(agents, ego, agents_len, u, z_axis):
   
    ego_speed = ego.state.speed
    ego_transform = ego.transform
    ego_position = np.array([ego_transform.position.x, ego_transform.position.y, ego_transform.position.z])
    ego_rotation = np.array([ego_transform.rotation.x, ego_transform.rotation.y, ego_transform.rotation.z])
    
    # global
    probability = 0
    break_distance = calculate_safe_distance(ego_speed, u)
    for i in range(1, agents_len):
        transform = agents[i].state.transform
        a_position = np.array([transform.position.x, transform.position.y, transform.position.z])
        a_rotation = np.array([transform.rotation.x, transform.rotation.y, transform.rotation.z])
        current_distance = calculate_distance(a_position, ego_position)
        # print('current distance: ', current_distance)
        new_probability = get_collision_probability_2(current_distance, ego_rotation, a_rotation, 
                                                      ego_position, a_position, break_distance, z_axis)
        if new_probability > probability:
            probability = new_probability
    # print(probability)
    return str(probability)
    
    
@jit(nopython=True, fastmath=True)
def get_line(agent_position, agent_velocity):
    agent_position_x = agent_position[0]
    agent_position_z = agent_position[2]

    agent_velocity_x = agent_velocity[0] if agent_velocity[0] != 0 else 0.0001
    agent_velocity_z = agent_velocity[2]

    # print('x, z, vx, vz: ', agent_position_x, agent_position_z, agent_velocity_x, agent_velocity_z)
    # if agent_velocity_x == 0:
    #     agent_velocity_x = 0.0001
    # print('k, b: ', agent_velocity_z / agent_velocity_x, -(agent_velocity_z / agent_velocity_x) * agent_position_x + agent_position_z)

    return agent_velocity_z / agent_velocity_x, -(
                agent_velocity_z / agent_velocity_x) * agent_position_x + agent_position_z

@jit(nopython=True, fastmath=True)
def get_distance(agent_position, x, z):
    return math.sqrt(pow(agent_position[0] - x, 2) + pow(agent_position[2] - z, 2))

@jit(nopython=True, fastmath=True)
def judge_same_line(a1_position, a1_speed, a1_velocity, a2_position, a2_speed, k1, k2):
    judge = False
    ego_ahead = False
    direction_vector = (a1_position[0] - a2_position[0], a1_position[2] - a2_position[2])
    distance = get_distance(a1_position, a2_position[0], a2_position[2])

    if abs(k1 - k2) < 0.6:
        if abs((a1_position[2] - a2_position[2]) /
               ((a1_position[0] - a2_position[0]) if (a1_position[0] - a2_position[0]) != 0 else 0.01) - (k1 + k2) / 2) < 0.6:
            judge = True

    if not judge:
        return judge, ego_ahead, -1

    if direction_vector[0] * a1_velocity[0] >= 0 and direction_vector[1] * a1_velocity[2] >= 0:
        ego_ahead = True  # Ego ahead of NPC.
        TTC = distance / (a1_speed - a2_speed)
    else:
        TTC = distance / (a2_speed - a1_speed)
    if TTC < 0:
        TTC = 100000

    return judge, ego_ahead, TTC

@jit(nopython=True, fastmath=True)
def judge_intersect_behind(a1_position, a1_velocity, a2_position, a2_velocity):
    judge = False
    k1, b1 = get_line(a1_position, a1_velocity)
    k2, b2 = get_line(a2_position, a2_velocity)
    if (k1 != k2):
        inter_point = ((b2-b1)/(k1-k2), (b2-b1)/(k1-k2) * k1 + b1)
        a1_pos_predict = a1_position + a1_velocity
        a2_pos_predict = a2_position + a2_velocity
        if ((a1_position[0] >= inter_point[0] and a1_position[0] <= a1_pos_predict[0]) or
            (a1_position[0] <= inter_point[0] and a1_position[0] >= a1_pos_predict[0]) or
            (a2_position[0] >= inter_point[0] and a2_position[0] <= a2_pos_predict[0]) or
            (a2_position[0] <= inter_point[0] and a2_position[0] >= a2_pos_predict[0])):
            judge = True
    return judge

@jit(nopython=True, fastmath=True)
def calculate_violation_probability(ego_position, ego_velocity, a_position, a_velocity, SD, distance):
    basic_vio_prob = SD / distance
    
    if SD > distance:
        basic_vio_prob = 1.0
    
    is_intersect_behind = judge_intersect_behind(ego_position, ego_velocity, a_position, a_velocity)
    if is_intersect_behind:
        return 0.5 * basic_vio_prob, is_intersect_behind
    
    return basic_vio_prob, is_intersect_behind

@jit(nopython=True, fastmath=True)
def not_stop_pedestrian(ego_acceleration, is_intersect_behind):
    if is_intersect_behind:
        return False
    
    if ego_acceleration < 0:
        return False
    
    return True


@jit(nopython=True, fastmath=True)
def sudden_braking(ego_brake, ego_pre_brake):
    if ego_brake - ego_pre_brake > 30:
        return True
    
    return False

@jit(nopython=True, fastmath=True)
def improper_passing(ego_position, ego_velocity, ego_acceleration, a_position, a_velocity, distance):
    is_npc_lane_change = not judge_intersect_behind(a_position, a_velocity, ego_position, ego_velocity)
    if is_npc_lane_change and ego_acceleration > 0 and distance < 30:
        return True

    return False

@jit(nopython=True, fastmath=True)        
def improper_lane_change(is_intersect_behind, ego_rotation, pre_ego_rotation):
    if is_intersect_behind:
        return False
    
    if abs(ego_rotation - pre_ego_rotation) > 20:
        return True
    
    return False 

@jit(nopython=True, fastmath=True)
def calculate_violation_reward(ego_position, ego_velocity, ego_brake, pre_ego_brake, ego_acceleration, 
                               ego_rotation, pre_ego_rotation, a_position, a_velocity, SD, distance, is_pedes):
    vio_prob, is_intersect_behind = calculate_violation_probability(ego_position, ego_velocity, a_position, a_velocity, SD, distance)
    
    violation_reward = np.zeros(N_VIOLATIONS)
    violation_reward[0] = -1.0
    violation_reward[1] = -1.0
    violation_reward[2] = -1.0
    violation_reward[3] = -1.0
    violation_reward[4] = -1.0
    violation_reward[5] = -1.0
        
    if not_stop_pedestrian(ego_acceleration, is_intersect_behind) and is_pedes:
        violation_reward[NOT_STOP_FOR_PEDESTRIAN] = vio_prob
    
    if improper_lane_change(is_intersect_behind, ego_rotation, pre_ego_rotation):
        violation_reward[IMPROPER_LANE_CHANGING] = vio_prob
        
    if sudden_braking(ego_brake, pre_ego_brake):
        violation_reward[SUDDEN_BRAKING] = vio_prob
    
    if improper_passing(ego_position, ego_velocity, ego_acceleration, a_position, a_velocity, distance):
        violation_reward[IMPROPER_PASSING] = vio_prob
        
    return violation_reward
        


def calculate_TTC(agents, ego, dis_tag):
    ego_transform = ego.transform
    ego_position = np.array([ego_transform.position.x, ego_transform.position.y, ego_transform.position.z])
    ego_velocity = np.array([ego.state.velocity.x, ego.state.velocity.y, ego.state.velocity.z])
    trajectory_ego_k, trajectory_ego_b = get_line(ego_position, ego_velocity)
    ego_speed = ego.state.speed if ego.state.speed > 0 else 0.0001

    # time_ego_list = []
    # time_agent_list = []
    TTC = 100000
    distance = 100000
    loProC_list, laProC_list = [0], [0]
    
    for i in range(1, len(agents)):
        transform = agents[i].transform
        state = agents[i].state
        a_position = np.array([transform.position.x, transform.position.y, transform.position.z])
        a_velocity = np.array([state.velocity.x, state.velocity.y, state.velocity.z])
        
        if dis_tag:
            dis = get_distance(ego_position, a_position[0], a_position[2])
            distance = dis if dis <= distance else distance
        # print('distance:', get_distance(ego, agents[i].transform.position.x, agents[i].transform.position.z))
        trajectory_agent_k, trajectory_agent_b = get_line(a_position, a_velocity)
        agent_speed = agents[i].state.speed if agents[i].state.speed > 0 else 0.0001
    
        same_lane, _, ttc = judge_same_line(ego_position, ego.state.speed, ego_velocity,
                                            a_position, agents[i].state.speed, trajectory_ego_k, trajectory_agent_k)
        # print('same_lane, TTC: ', same_lane, ttc)
        if same_lane:
            # print('Driving on Same Lane, TTC: {}'.format(ttc))
            time_ego = ttc
            time_agent = ttc
        else:
            trajectory_agent_k = trajectory_agent_k if trajectory_ego_k - trajectory_agent_k != 0 else trajectory_agent_k + 0.0001

            collision_point_x, collision_point_z = (trajectory_agent_b - trajectory_ego_b) / (trajectory_ego_k - trajectory_agent_k), \
                                                   (trajectory_ego_k * trajectory_agent_b - trajectory_agent_k * trajectory_ego_b) / (trajectory_ego_k - trajectory_agent_k)

            ego_distance = get_distance(ego_position, collision_point_x, collision_point_z)
            agent_distance = get_distance(a_position, collision_point_x, collision_point_z)
            time_ego = ego_distance / ego_speed
            time_agent = agent_distance / agent_speed
            # print('Driving on Different Lane, TTC: {}'.format(time_ego))

        if abs(time_ego - time_agent) < 1:
            TTC = min(TTC, (time_ego + time_agent) / 2)

    return TTC, distance

@jit(nopython=True, fastmath=True)
def cal_dis(a, b):
    dis_x = a[0] - b[0]
    dis_z = a[2] - b[2]
    
    return math.sqrt(dis_x ** 2 + dis_z ** 2)

# @jit(nopython=True, fastmath=True)
def calculate_measures(state_list, ego_state, ego_brake, pre_ego_brake, ego_acceleration,
                        ego_rotation, pre_ego_rotation, isNpcVehicle, mid_point = None, dis_tag = True):
    
    # print(len(state_list), len(isPedestrian))
    
    ego_transform = ego_state.transform
    ego_speed = ego_state.speed if ego_state.speed > 0 else 0.0001
    
    ego_position = np.array([ego_transform.position.x, ego_transform.position.y, ego_transform.position.z])
    ego_velocity = np.array([ego_state.velocity.x, ego_state.velocity.y, ego_state.velocity.z])

    trajectory_ego_k, trajectory_ego_b = get_line(ego_position, ego_velocity)

    # time_ego_list = []
    # time_agent_list = []
    TTC = 100000
    distance = 100
    loProC_list, laProC_list = [0], [0]  # probability
    not_stop_pedes_list = []
    default_vio = np.zeros(6)
    default_vio[0] = -1.0
    default_vio[1] = -1.0
    default_vio[2] = -1.0
    default_vio[3] = -1.0
    default_vio[4] = -1.0
    default_vio[5] = -1.0
    reward_list = [default_vio]
    
    #for i in range(1, len(agents)):
    for i in range(0, len(state_list)):
        transform = state_list[i].transform
        state = state_list[i]
        a_position = np.array([transform.position.x, transform.position.y, transform.position.z])
        a_velocity = np.array([state.velocity.x, state.velocity.y, state.velocity.z])
        dis = get_distance(ego_position, a_position[0], a_position[2])
        # dis = cal_dis(a_position, mid_point)
        # print("Distance from ego to obstacle: ", dis)
        distance = dis if dis <= distance else distance
        trajectory_agent_k, trajectory_agent_b = get_line(a_position, a_velocity)
        agent_speed = state.speed if state.speed > 0 else 0.0001

        same_lane, ego_ahead, ttc = judge_same_line(ego_position, ego_speed, ego_velocity,
                                                    a_position, state.speed, trajectory_ego_k, trajectory_agent_k)
        ego_deceleration = 6  # probability
        if same_lane:
            # print('Driving on Same Lane, TTC: {}'.format(ttc))
            time_ego = ttc
            time_agent = ttc

            loSD = 100000
            if isNpcVehicle[i]:  # type value, 1-EGO, 2-NPC, 3-Pedestrian
                agent_deceleration = 6
                loSD = 1 / 2 * (
                    abs(pow(ego_speed, 2) / ego_deceleration - pow(agent_speed, 2) / agent_deceleration)) + 5
            else:
                agent_deceleration = 1.5
                loSD = 1 / 2 * (pow(ego_speed, 2) / ego_deceleration - pow(agent_speed, 2) / agent_deceleration) + 5
            
            loProC = calculate_collision_probability(loSD, dis)
            loProC_list.append(loProC)
            # calculate violation probability
            violation_reward = calculate_violation_reward(ego_position, ego_velocity, ego_brake, pre_ego_brake, ego_acceleration, 
                                                          ego_rotation, pre_ego_rotation, a_position, a_velocity, loSD, dis, not isNpcVehicle[i])
            reward_list.append(violation_reward)
        else:
            trajectory_agent_k = trajectory_agent_k if trajectory_ego_k - trajectory_agent_k != 0 else trajectory_agent_k + 0.0001

            collision_point_x, collision_point_z = (trajectory_agent_b - trajectory_ego_b) / (
                        trajectory_ego_k - trajectory_agent_k), \
                                                   (
                                                               trajectory_ego_k * trajectory_agent_b - trajectory_agent_k * trajectory_ego_b) / (
                                                               trajectory_ego_k - trajectory_agent_k)

            ego_distance = get_distance(ego_position, collision_point_x, collision_point_z)
            agent_distance = get_distance(a_position, collision_point_x, collision_point_z)
            time_ego = ego_distance / ego_speed
            time_agent = agent_distance / agent_speed
            # print('Driving on Different Lane, TTC: {}'.format(time_ego))

            theta = calculate_angle_tan(trajectory_ego_k, trajectory_agent_k)
            # print(trajectory_ego_k, trajectory_agent_k, theta)
            laSD = pow(ego_speed * math.sin(theta), 2) / (ego_deceleration * math.sin(theta)) + 5
            laProC = calculate_collision_probability(laSD, dis)
            laProC_list.append(laProC)
            # calculate violation probability
            violation_reward = calculate_violation_reward(ego_position, ego_velocity, ego_brake, pre_ego_brake, ego_acceleration, 
                                                          ego_rotation, pre_ego_rotation, a_position, a_velocity, laSD, dis, not isNpcVehicle[i])
            reward_list.append(violation_reward)

        if abs(time_ego - time_agent) < 1:
            TTC = min(TTC, (time_ego + time_agent) / 2)

    loProC_dt, laProC_dt = max(loProC_list), max(laProC_list)
    proC_dt = max(loProC_dt, laProC_dt) + (1 - max(loProC_dt, laProC_dt)) * min(loProC_dt, laProC_dt)
    reward = np.max(reward_list, axis=0)
    return TTC, distance, proC_dt, reward


if __name__ == "__main__":
    # v1 = lgsvl.Vector(0, 0, 1)
    # v2 = lgsvl.Vector(0, 1, 1)
    #
    # print(calculate_angle(v1, v2))
    # print(calculate_distance(v1, v2))
    print(calculate_collision_probability(10, 2))

    a = (1, 99, 3)
    print(a[2])
# print(npc_vehicle)
