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

# threshold
theta_brake = 40 # brake percentage deviation in 2s
theta_speed = 30 # speed level considered as speeding
theta_lane_change = math.pi/6 # angle of ego and current lane direction when lane changing
theta_turn = math.pi/4 # angle of ego and current lane direction when turnings
theta_lane_direction = math.pi/18 # angle of current and next lane


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
    if current_distance >= safe_distance:
        return safe_distance / current_distance
    else:
        return 1.0

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

def judge_condition(state_list, ego_state, prev_brake_percentage, brake_percentage, ego_acc, 
                    road, next_road, p_lane_id, current_signals, p_tlight_sign, orientation):
    
    global theta_brake
    global theta_speed
    global theta_lane_change
    global theta_turn
    global theta_lane_direction
    
    ego_transform = ego_state.transform
    ego_position = np.array([ego_transform.position.x, ego_transform.position.y, ego_transform.position.z])

    yaw_deg = ego_transform.rotation.z
    yaw_rad = np.deg2rad(yaw_deg)
    front_position = np.array([
       ego_position[0] + 2.3 * np.cos(yaw_rad),
       ego_position[1],
       ego_position[2] + 2.3 * np.sin(yaw_rad)
    ])
    # print(f"Ego_position: {ego_position}\n")
    # print(f"Front_position: {front_position}\n")
    
    ego_velocity = np.array([ego_state.velocity.x, ego_state.velocity.y, ego_state.velocity.z])
    ego_speed =  ego_state.speed
    
    trajectory_ego_k, trajectory_ego_b = get_line(ego_position, ego_velocity)
    #Passing 0, Lane Changing 1, Turning 2, Braking 3, Speeding 4, Cruising 5 , Other 6
    condition = [0, 0, 0, 0, 0, 0, 0]
    
    #Check braking (Verified)
    if brake_percentage - prev_brake_percentage >= theta_brake:
        condition[3] = 1
        
    #Check Speeding (Verified)
    if ego_speed*3.6 > theta_speed and ego_acc > 0: 
        condition[4] = 1
        
    #Check Passing (Verified)
    for i in range(0, len(state_list)):
        transform = state_list[i].transform
        state = state_list[i]
        a_position = np.array([transform.position.x, transform.position.y, transform.position.z])
        a_velocity = np.array([state.velocity.x, state.velocity.y, state.velocity.z])
        if judge_intersect_behind(ego_position, ego_velocity, a_position, a_velocity):
            continue
        
        if ego_velocity[0] * (ego_position[0] - a_position[0]) + ego_velocity[2] * (ego_position[2] - a_position[2]) > 0:
            continue
        
        trajectory_agent_k, trajectory_agent_b = get_line(a_position, a_velocity)
        agent_speed = state.speed
        ego_diff = ego_velocity
        agent_diff = a_velocity
        if np.dot(ego_diff, agent_diff) > 0 and (ego_speed > agent_speed) and (ego_acc > 0):
            condition[0] = 1
    
    #Check Turn & Lane change (Verified)
    angle_radians = math.atan2(road['z'], road['x']) - math.atan2(next_road['z'], next_road['x'])
    angle_radians_ego = math.atan2(road['z'], road['x']) - math.atan2(ego_velocity[2], ego_velocity[0])
    # if (abs(angle_radians) > theta_lane_direction): 
    #     condition[2] = 1
    # elif (abs(angle_radians_ego) > theta_lane_change and road['lane_id'] != next_road['lane_id']):
    #     condition[1] = 1
    
    # Lane changing (Verified)
    if abs(angle_radians_ego) > theta_lane_change:
        if abs(angle_radians) < theta_lane_direction and road['lane_id'] != next_road['lane_id']:
            condition[1] = 1
    
    # Turning (Verified)
    if abs(angle_radians_ego) > theta_turn:
        if abs(angle_radians) > theta_lane_direction:
            condition[2] = 1
        
    #Check Run on red light (Verified)

    ego_position = front_position
    for signal in current_signals:
        tlight = current_signals[signal]
        if tlight['color'] == 1:
            a = tlight['stop_line']['a']
            b = tlight['stop_line']['b']
            c = tlight['stop_line']['c']
            
            signal_dist = abs(ego_position[0] * a + ego_position[2] * b + c) / math.sqrt(a**2 + b**2)
            if (signal_dist <= 5):
                # print(f"Signal dist: signal_dist {signal_dist}\n")
                condition[5] = max(condition[5], 2 * (1 - 1/(1 + math.exp(-signal_dist))))
                if tlight['id'] in p_tlight_sign:
                    # tlight_sign_i = False
                    tlight_sign_temp = (ego_position[0] * a + ego_position[2] * b + c  > 0)
                    if (p_tlight_sign[tlight['id']] != tlight_sign_temp ):
                        print(f"Run on a red light\n") 
                        condition[5] = 1
                        # p_tlight_sign[tlight['id']] = tlight_sign_temp   
                    # if p_tlight_sign[tlight['id']] == False:
                    #     tlight_sign_i = (ego_position[0] * a + ego_position[2] * b + c - addition > 0)
                    # else:
                    #     tlight_sign_i = (ego_position[0] * a + ego_position[2] * b + c + addition > 0)
                        
                    # if p_tlight_sign[tlight['id']] != tlight_sign_i:
                    #     condition[5] = 1
                    #     p_tlight_sign[tlight['id']] = tlight_sign_i
                else:
                    tlight_sign_i = (ego_position[0] * a + ego_position[2] * b + c > 0)
                    p_tlight_sign[tlight['id']] = tlight_sign_i
                    # if tlight_sign_i == False:
                    #     p_tlight_sign[tlight['id']] = (ego_position[0] * a + ego_position[2] * b + c - addition > 0)
                    # else:
                    #     p_tlight_sign[tlight['id']] = (ego_position[0] * a + ego_position[2] * b + c + addition > 0)
                        
    # print("Curr: ", p_tlight_sign) 
#    print(f"Speed, ego: {ego_speed}, {ego_acc}\n")
    # print(f"Condition: Passing {condition[0]}, Lane Changing {condition[1]}, Turning {condition[2]}, Braking {condition[3]}, Speeding {condition[4]}, Cruising {condition[5]} \n ")
    return condition, p_tlight_sign


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

@jit(nopython=True, fastmath=True)
def get_world_acc(world_speed, sim_speed, sim_acc, coord, world_vel):
    
    if sim_speed == 0:
        return sim_acc
    
    world_acc = world_speed / sim_speed * sim_acc
    
    
    if coord == 0:
        return world_vel['vx'] / world_speed * world_acc
    
    return world_vel['vy'] / world_speed * world_acc

@jit(nopython=True, fastmath=True)
def get_cos(vec1, vec2):
    numerator = vec1[0] * vec2[0] + vec1[1] * vec2[1] + vec1[2] * vec2[2]
    denominator = math.sqrt(vec1[0] ** 2 + vec1[1] ** 2 + vec1[2] ** 2) * math.sqrt(vec2[0] ** 2 + vec2[1] ** 2 + vec2[2] ** 2) + 0.0001
    
    return numerator / denominator

# @jit(nopython=True, fastmath=True)
def calculate_measures(state_list, ego_state, isNpcVehicle, current_signals, ego_curr_acc, prev_brake_percentage, brake_percentage, 
                       agent_uid, road, next_road, p_lane_id, p_tlight_sign, 
                       orientation, mid_point = None, dis_tag = True):
    
    ego_transform = ego_state.transform
    ego_speed = ego_state.speed
    
    ego_position = np.array([ego_transform.position.x, ego_transform.position.y, ego_transform.position.z])
    ego_velocity = np.array([ego_state.velocity.x, ego_state.velocity.y, ego_state.velocity.z])
    
    ego_acc = None
    
    if ego_speed == 0:
        ego_acc = np.array([6, 6, 6])
    else:
        ego_acc = np.array([abs(ego_velocity[0]) / ego_speed * 6, abs(ego_velocity[1]) / ego_speed * 6, abs(ego_velocity[2]) / ego_speed * 6])
        if ego_acc[0] == 0: 
            ego_acc[0] = 0.001
        if ego_acc[1] == 0: 
            ego_acc[1] = 0.001
        if ego_acc[2] == 0: 
            ego_acc[2] = 0.001

    trajectory_ego_k, trajectory_ego_b = get_line(ego_position, ego_velocity)

    TTC = 100000
    distance = 10000
    loProC_list, laProC_list = [0], [0]  # probability
    frontVioRate_list = [0]
    behindVioRate_list = [0]
    proC_list = [0]
    vioRate_list = [0]
    
    reaction_time = 0.5
    
    for i in range(0, len(state_list)):
        transform = state_list[i].transform
        state = state_list[i]
        a_position = np.array([transform.position.x, transform.position.y, transform.position.z])
        a_velocity = np.array([state.velocity.x, state.velocity.y, state.velocity.z])
        
        if judge_intersect_behind(ego_position, ego_velocity, a_position, a_velocity):
            continue
        
        dis_vec = np.array([ego_position[0] - a_position[0], ego_position[1] - a_position[1], ego_position[2] - a_position[2]])
        
        agent_acc = None
        
        if dis_tag:
            dis = get_distance(ego_position, a_position[0], a_position[2])
            distance = dis if dis <= distance else distance
        trajectory_agent_k, trajectory_agent_b = get_line(a_position, a_velocity)
        agent_speed = state.speed
        if isNpcVehicle[i]:
            if agent_speed == 0:
                agent_acc = np.array([6, 6, 6])
            else:
                agent_acc = np.array([abs(a_velocity[0]) / state.speed * 6, abs(a_velocity[1]) / state.speed * 6, abs(a_velocity[2]) / state.speed * 6])
        else:
            if agent_speed == 0:
                agent_acc = np.array([1.5, 1.5, 1.5])
            else:
                agent_acc = np.array([abs(a_velocity[0]) / state.speed * 1.5, abs(a_velocity[1]) / state.speed * 1.5, abs(a_velocity[2]) / state.speed * 1.5])

        if agent_acc[0] == 0: 
            agent_acc[0] = 0.001
        if agent_acc[1] == 0: 
            agent_acc[1] = 0.001
        if agent_acc[2] == 0: 
            agent_acc[2] = 0.001

        ttc = 5

        time_ego = ttc
        time_agent = ttc

        # Calculate LoSD
        loSD = 100000

        if ego_velocity[0] * a_velocity[0] > 0:
            if ego_velocity[0] * (ego_position[0] - a_position[0]) < 0:
                loSD = 1 / 2 * (
                    abs(pow(ego_velocity[0], 2) / ego_acc[0] - pow(a_velocity[0], 2) / agent_acc[0])) + ego_velocity[0] * reaction_time
            else: 
                loSD = 1 / 2 * (
                    abs(pow(ego_velocity[0], 2) / ego_acc[0] - pow(a_velocity[0], 2) / agent_acc[0])) + a_velocity[0] * reaction_time
        else:
            loSD = 1 / 2 * (
                abs(pow(ego_velocity[0], 2) / ego_acc[0] + pow(a_velocity[0], 2) / agent_acc[0]))
        
        loProC = calculate_collision_probability(loSD, abs(ego_position[0] - a_position[0]))
        loVioRate = calculate_violation_rate(loSD, abs(ego_position[0] - a_position[0]))
        loProC_list.append(loProC)

        # Calculate LaSD
        trajectory_agent_k = trajectory_agent_k if trajectory_ego_k - trajectory_agent_k != 0 else trajectory_agent_k + 0.0001

        collision_point_x, collision_point_z = (trajectory_agent_b - trajectory_ego_b) / (
                    trajectory_ego_k - trajectory_agent_k), \
                                                (
                                                            trajectory_ego_k * trajectory_agent_b - trajectory_agent_k * trajectory_ego_b) / (
                                                            trajectory_ego_k - trajectory_agent_k)

        ego_distance = get_distance(ego_position, collision_point_x, collision_point_z)
        agent_distance = get_distance(a_position, collision_point_x, collision_point_z)
        if ego_speed:
            time_ego = ego_distance / ego_speed
        else:
            time_ego = ego_distance / 0.001
        if agent_speed:
            time_agent = agent_distance / agent_speed
        else:
            time_agent = agent_distance / 0.001
        
        laSD = 1000000
        
        if ego_velocity[2] * a_velocity[2] > 0:
            if ego_velocity[2] * (ego_position[2] - a_position[2]) < 0:
                laSD = 1 / 2 * (
                    abs(pow(ego_velocity[2], 2) / ego_acc[2] - pow(a_velocity[2], 2) / agent_acc[2])) + ego_velocity[2] * reaction_time
            else: 
                laSD = 1 / 2 * (
                    abs(pow(ego_velocity[2], 2) / ego_acc[2] - pow(a_velocity[2], 2) / agent_acc[2])) + a_velocity[2] * reaction_time
        else:
            laSD = 1 / 2 * (
                abs(pow(ego_velocity[2], 2) / ego_acc[2] + pow(a_velocity[2], 2) / agent_acc[2]))

        laProC = calculate_collision_probability(laSD, abs(ego_position[2] - a_position[2]))
        laVioRate = calculate_violation_rate(laSD, abs(ego_position[2] - a_position[2]))
        laProC_list.append(laProC)
        
        proC = (laProC + loProC) / 2
        
        vioRate = (laVioRate + loVioRate) / 2
        
        if get_cos(dis_vec, ego_velocity) < 0:
            frontVioRate_list.append(vioRate)
        else:
            if get_cos(ego_velocity, a_velocity) > 0:
                behindVioRate_list.append(vioRate)
        
        proC_list.append(proC)
        vioRate_list.append(vioRate)
        
        if abs(time_ego - time_agent) < 1:
            TTC = min(TTC, (time_ego + time_agent) / 2)

    proC_dt = max(proC_list) + (1 - max(proC_list)) * min(proC_list)
    vioRate_dt = max(vioRate_list) + (1 - max(vioRate_list)) * min(vioRate_list)
    frontVioRate_dt = max(frontVioRate_list) + (1 - max(frontVioRate_list)) * min(frontVioRate_list)
    behindVioRate_dt = max(behindVioRate_list) + (1 - max(behindVioRate_list)) * min(behindVioRate_list)
    
    #Passing 0, Lane Changing 1, Turning 2, Braking 3, Speeding 4, Cruising 5, Other 6
    condition, curr_tlight_sign = judge_condition(state_list, ego_state, prev_brake_percentage, brake_percentage, ego_curr_acc, road, next_road, p_lane_id, current_signals, p_tlight_sign, orientation)
    
    total_rate = 0
    
    np_condition = []
    
    for i in range(0, 7):
        violation_rate_ = 0
        
        if i == 3:
            violation_rate_ = behindVioRate_dt
        elif i == 4 or i == 0:
            violation_rate_ = frontVioRate_dt
        else:
            violation_rate_ = vioRate_dt
            
        if i == 5:
            np_condition.append(condition[i])
        else:    
            np_condition.append(condition[i] * violation_rate_)
    
    # np_condition.append(proC_dt)
    
    if condition[5] == 1.0:
        np_condition[5] = 1.0
        
    return TTC, distance, proC_dt, curr_tlight_sign, np_condition


if __name__ == "__main__":
    print(calculate_collision_probability(10, 2))

    a = (1, 99, 3)
    print(a[2])