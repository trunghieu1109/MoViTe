import math
import numpy as np
import random
from lgsvl.agent import NpcVehicle
from numba import jit

print('*' * 80)

print("Using original collision utils")

print('*' * 80)

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

def calculate_violation_rate(safe_distance, current_distance):
    return safe_distance / current_distance

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

def judge_condition(state_list, ego_state, brake_percentage, ego_world_vel, agent_world_vel, 
                       ego_world_acc, ego_world_pos, road, next_road, current_signals):
    ego_transform = ego_state.transform
    ego_position = np.array([ego_transform.position.x, ego_transform.position.y, ego_transform.position.z])
    ego_velocity = np.array([ego_state.velocity.x, ego_state.velocity.y, ego_state.velocity.z])
    ego_world_speed =  math.sqrt(ego_world_vel['vx']**2 + ego_world_vel['vy']**2)

    trajectory_ego_k, trajectory_ego_b = get_line(ego_position, ego_velocity)
    #Passing 0, Lane Changing 1, Turning 2, Braking 3, Speeding 4, Cruising 5 
    condition = [0, 0, 0, 0 , 0, 0]
    
    #Check braking
    if brake_percentage >= 40:
        condition[3] = 1
        
    #Check Speeding
    if ego_world_acc > 0: 
        condition[4] = 1
        
    #Check Passing
    for i in range(0, len(state_list)):
        transform = state_list[i].transform
        state = state_list[i]
        a_position = np.array([transform.position.x, transform.position.y, transform.position.z])
        a_velocity = np.array([state.velocity.x, state.velocity.y, state.velocity.z])
        trajectory_agent_k, trajectory_agent_b = get_line(a_position, a_velocity)
        agent_world_speed = math.sqrt(agent_world_vel[i]['vx'] ** 2 + agent_world_vel[i]['vy'] ** 2)
        ego_diff = ego_velocity - ego_position
        agent_diff = a_velocity - a_position
        if (abs(trajectory_ego_k - trajectory_agent_k) < 0.01) and np.dot(ego_diff, agent_diff) > 0  and  (ego_world_speed > agent_world_speed) and (ego_world_acc > 0):
            condition[0] = 1
    
    #Check Turn & Lane change
        angle_radians = math.atan2(road['y'], road['x']) - math.atan2(next_road['y'], next_road['x'])
        if (abs(angle_radians) > math.pi/4): 
            condition[2] = 1
        elif (road['lane_id'] != next_road['lane_id']):
            condition[1] = 1
    
    #Check signal
    for signal in current_signals:
        if signal['color'] == 1:
            signal_dist = abs(ego_world_pos['x'] * signal['a'] + ego_world_pos['y'] * signal['b'] + signal['c']) / math.sqrt(signal['a']**2 + signal['b']**2)
            if (signal_dist <= 5):
                condition[5] = 1
        
    return condition


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

def get_world_acc(world_speed, sim_speed, sim_acc, coord, world_vel):
    
    if sim_speed == 0:
        return sim_acc
    
    world_acc = world_speed / sim_speed * sim_acc
    
    
    if coord == 0:
        return world_vel['vx'] / world_speed * world_acc
    
    return world_vel['vy'] / world_speed * world_acc

# @jit(nopython=True, fastmath=True)
def calculate_measures(state_list, ego_state, isNpcVehicle, ego_world_vel, agent_world_vel, 
                       ego_world_pos, agent_world_pos, current_signals,
                       road, next_road, mid_point = None, dis_tag = True):
    
    ego_transform = ego_state.transform
    ego_world_speed = math.sqrt(ego_world_vel['vx'] ** 2 + ego_world_vel['vy'] ** 2)
    ego_world_acc = {
        'x': get_world_acc(ego_world_speed, ego_state.speed, 6, 0, ego_world_vel),
        'y': get_world_acc(ego_world_speed, ego_state.speed, 6, 1, ego_world_vel)
    }
    
    ego_position = np.array([ego_transform.position.x, ego_transform.position.y, ego_transform.position.z])
    ego_velocity = np.array([ego_state.velocity.x, ego_state.velocity.y, ego_state.velocity.z])

    trajectory_ego_k, trajectory_ego_b = get_line(ego_position, ego_velocity)

    TTC = 100000
    distance = 100
    loProC_list, laProC_list = [0], [0]  # probability
    loVioRate_list, laVioRate_list = [0], [0]
    proC_list = [0]
    vioRate_list = [0]
    
    
    reaction_time = 1.0
    
    for i in range(0, len(state_list)):
        transform = state_list[i].transform
        state = state_list[i]
        a_position = np.array([transform.position.x, transform.position.y, transform.position.z])
        a_velocity = np.array([state.velocity.x, state.velocity.y, state.velocity.z])
        
        if dis_tag:
            dis = get_distance(ego_position, a_position[0], a_position[2])
            distance = dis if dis <= distance else distance
        trajectory_agent_k, trajectory_agent_b = get_line(a_position, a_velocity)
        agent_world_speed = math.sqrt(agent_world_vel[i]['vx'] ** 2 + agent_world_vel[i]['vy'] ** 2)
        if isNpcVehicle:
            agent_world_acc = {
                'x': get_world_acc(agent_world_speed, state.speed, 6, 0, agent_world_vel[i]),
                'y': get_world_acc(agent_world_speed, state.speed, 6, 1, agent_world_vel[i])
            }
        else:
            agent_world_acc = {
                'x': get_world_acc(agent_world_speed, state.speed, 1.5, 0, agent_world_vel[i]),
                'y': get_world_acc(agent_world_speed, state.speed, 1.5, 1, agent_world_vel[i])
            }

        ttc = 5

        time_ego = ttc
        time_agent = ttc

        # Calculate LoSD
        loSD = 100000

        if ego_world_vel['vx'] * agent_world_vel[i]['vx'] > 0:
            if ego_world_vel['vx'] * (ego_world_pos['x'] - agent_world_pos[i]['x']) < 0:
                loSD = 1 / 2 * (
                    abs(pow(ego_world_vel['vx'], 2) / ego_world_acc['x'] - pow(agent_world_vel[i]['vx'], 2) / agent_world_acc['x'])) + ego_world_vel['vx'] * reaction_time + 5
            else: 
                loSD = 1 / 2 * (
                    abs(pow(ego_world_vel['vx'], 2) / ego_world_acc['x'] - pow(agent_world_vel[i]['vx'], 2) / agent_world_acc['x'])) + agent_world_vel[i]['vx'] * reaction_time + 5
        else:
            loSD = 1 / 2 * (
                abs(pow(ego_world_vel['vx'], 2) / ego_world_acc['x'] + pow(agent_world_vel[i]['vx'], 2) / agent_world_acc['x'])) + ego_world_vel['vx'] + agent_world_vel[i]['vx'] + 5
        
        loProC = calculate_collision_probability(loSD, abs(ego_world_pos['x'] - agent_world_pos[i]['x']))
        loVioRate = calculate_violation_rate(loSD, abs(ego_world_pos['x'] - agent_world_pos[i]['x']))
        loProC_list.append(loProC)
        loVioRate_list.append(loVioRate)

        # Calculate LaSD
        trajectory_agent_k = trajectory_agent_k if trajectory_ego_k - trajectory_agent_k != 0 else trajectory_agent_k + 0.0001

        collision_point_x, collision_point_z = (trajectory_agent_b - trajectory_ego_b) / (
                    trajectory_ego_k - trajectory_agent_k), \
                                                (
                                                            trajectory_ego_k * trajectory_agent_b - trajectory_agent_k * trajectory_ego_b) / (
                                                            trajectory_ego_k - trajectory_agent_k)

        ego_distance = get_distance(ego_position, collision_point_x, collision_point_z)
        agent_distance = get_distance(a_position, collision_point_x, collision_point_z)
        time_ego = ego_distance / ego_world_speed
        time_agent = agent_distance / agent_world_speed
        
        laSD = 1000000
        
        if ego_world_vel['vy'] * agent_world_vel[i]['vy'] > 0:
            if ego_world_vel['vy'] * (ego_world_pos['y'] - agent_world_pos[i]['y']) < 0:
                laSD = 1 / 2 * (
                    abs(pow(ego_world_vel['vy'], 2) / ego_world_acc['y'] - pow(agent_world_vel[i]['vy'], 2) / agent_world_acc['y'])) + ego_world_vel['vy'] * reaction_time + 5
            else:
                laSD = 1 / 2 * (
                    abs(pow(ego_world_vel['vy'], 2) / ego_world_acc['y'] - pow(agent_world_vel[i]['vy'], 2) / agent_world_acc['y'])) + agent_world_vel[i]['vy'] * reaction_time + 5
        else:
            laSD = 1 / 2 * (
                abs(pow(ego_world_vel['vy'], 2) / ego_world_acc['y'] + pow(agent_world_vel[i]['vy'], 2) / agent_world_acc['y'])) + ego_world_vel['vy'] + agent_world_vel[i]['vy'] + 5

        laProC = calculate_collision_probability(laSD, abs(ego_world_pos['y'] - agent_world_pos[i]['y']))
        laVioRate = calculate_violation_rate(laSD, abs(ego_world_pos['y'] - agent_world_pos[i]['y']))
        laProC_list.append(laProC)
        laVioRate_list.append(laVioRate)
        
        proC = laProC * loProC
        vioRate = laVioRate * loVioRate
        
        proC_list.append(proC)
        vioRate_list.append(vioRate)
        
        if abs(time_ego - time_agent) < 1:
            TTC = min(TTC, (time_ego + time_agent) / 2)

    proC_dt = max(proC_list)
    vioRate_dt = max(vioRate_list)

    return TTC, distance, proC_dt


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
