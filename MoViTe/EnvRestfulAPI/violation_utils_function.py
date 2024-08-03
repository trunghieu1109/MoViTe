import math
import numpy as np
import random
from lgsvl.agent import NpcVehicle
from numba import jit

free_of_traffic_range = 20

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

# def judge_condition(state_list, ego_state, prev_brake_percentage, brake_percentage, ego_acc, 
#                     road, next_road, p_lane_id, current_signals, p_tlight_sign, orientation):
    
#     global theta_brake
#     global theta_speed
#     global theta_lane_change
#     global theta_turn
#     global theta_lane_direction
    
#     ego_transform = ego_state.transform
#     ego_position = np.array([ego_transform.position.x, ego_transform.position.y, ego_transform.position.z])

#     yaw_deg = ego_transform.rotation.z
#     yaw_rad = np.deg2rad(yaw_deg)
#     front_position = np.array([
#        ego_position[0] + 2.3 * np.cos(yaw_rad),
#        ego_position[1],
#        ego_position[2] + 2.3 * np.sin(yaw_rad)
#     ])
#     # print(f"Ego_position: {ego_position}\n")
#     # print(f"Front_position: {front_position}\n")
    
#     ego_velocity = np.array([ego_state.velocity.x, ego_state.velocity.y, ego_state.velocity.z])
#     ego_speed =  ego_state.speed
    
#     trajectory_ego_k, trajectory_ego_b = get_line(ego_position, ego_velocity)
#     #Passing 0, Lane Changing 1, Turning 2, Braking 3, Speeding 4, Cruising 5 , Other 6
#     condition = [0, 0, 0, 0, 0, 0, 0]
    
#     #Check braking (Verified)
#     if brake_percentage - prev_brake_percentage >= theta_brake:
#         condition[3] = 1
        
#     #Check Speeding (Verified)
#     if ego_speed*3.6 > theta_speed and ego_acc > 0: 
#         condition[4] = 1
        
#     #Check Passing (Verified)
#     for i in range(0, len(state_list)):
#         transform = state_list[i].transform
#         state = state_list[i]
#         a_position = np.array([transform.position.x, transform.position.y, transform.position.z])
#         a_velocity = np.array([state.velocity.x, state.velocity.y, state.velocity.z])
#         if judge_intersect_behind(ego_position, ego_velocity, a_position, a_velocity):
#             continue
        
#         if ego_velocity[0] * (ego_position[0] - a_position[0]) + ego_velocity[2] * (ego_position[2] - a_position[2]) > 0:
#             continue
        
#         trajectory_agent_k, trajectory_agent_b = get_line(a_position, a_velocity)
#         agent_speed = state.speed
#         ego_diff = ego_velocity
#         agent_diff = a_velocity
#         if np.dot(ego_diff, agent_diff) > 0 and (ego_speed > agent_speed) and (ego_acc > 0):
#             condition[0] = 1
    
#     #Check Turn & Lane change (Verified)
#     angle_radians = math.atan2(road['z'], road['x']) - math.atan2(next_road['z'], next_road['x'])
#     angle_radians_ego = math.atan2(road['z'], road['x']) - math.atan2(ego_velocity[2], ego_velocity[0])
#     # if (abs(angle_radians) > theta_lane_direction): 
#     #     condition[2] = 1
#     # elif (abs(angle_radians_ego) > theta_lane_change and road['lane_id'] != next_road['lane_id']):
#     #     condition[1] = 1
    
#     # Lane changing (Verified)
#     if abs(angle_radians_ego) > theta_lane_change:
#         if abs(angle_radians) < theta_lane_direction and road['lane_id'] != next_road['lane_id']:
#             condition[1] = 1
    
#     # Turning (Verified)
#     if abs(angle_radians_ego) > theta_turn:
#         if abs(angle_radians) > theta_lane_direction:
#             condition[2] = 1
        
#     #Check Run on red light (Verified)

#     ego_position = front_position
#     for signal in current_signals:
#         tlight = current_signals[signal]
#         if tlight['color'] == 1:
#             a = tlight['stop_line']['a']
#             b = tlight['stop_line']['b']
#             c = tlight['stop_line']['c']
            
#             signal_dist = abs(ego_position[0] * a + ego_position[2] * b + c) / math.sqrt(a**2 + b**2)
#             if (signal_dist <= 5):
#                 # print(f"Signal dist: signal_dist {signal_dist}\n")
#                 condition[5] = max(condition[5], 2 * (1 - 1/(1 + math.exp(-signal_dist))))
#                 if tlight['id'] in p_tlight_sign:
#                     # tlight_sign_i = False
#                     tlight_sign_temp = (ego_position[0] * a + ego_position[2] * b + c  > 0)
#                     if (p_tlight_sign[tlight['id']] != tlight_sign_temp ):
#                         print(f"Run on a red light\n") 
#                         condition[5] = 1
#                         # p_tlight_sign[tlight['id']] = tlight_sign_temp   
#                     # if p_tlight_sign[tlight['id']] == False:
#                     #     tlight_sign_i = (ego_position[0] * a + ego_position[2] * b + c - addition > 0)
#                     # else:
#                     #     tlight_sign_i = (ego_position[0] * a + ego_position[2] * b + c + addition > 0)
                        
#                     # if p_tlight_sign[tlight['id']] != tlight_sign_i:
#                     #     condition[5] = 1
#                     #     p_tlight_sign[tlight['id']] = tlight_sign_i
#                 else:
#                     tlight_sign_i = (ego_position[0] * a + ego_position[2] * b + c > 0)
#                     p_tlight_sign[tlight['id']] = tlight_sign_i
#                     # if tlight_sign_i == False:
#                     #     p_tlight_sign[tlight['id']] = (ego_position[0] * a + ego_position[2] * b + c - addition > 0)
#                     # else:
#                     #     p_tlight_sign[tlight['id']] = (ego_position[0] * a + ego_position[2] * b + c + addition > 0)
                        
#     # print("Curr: ", p_tlight_sign) 
# #    print(f"Speed, ego: {ego_speed}, {ego_acc}\n")
#     # print(f"Condition: Passing {condition[0]}, Lane Changing {condition[1]}, Turning {condition[2]}, Braking {condition[3]}, Speeding {condition[4]}, Cruising {condition[5]} \n ")
#     return condition, p_tlight_sign

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
def calculate_cos(vec1, vec2):
    numerator = vec1[0] * vec2[0] + vec1[1] * vec2[1] + vec1[2] * vec2[2]
    denominator = math.sqrt(vec1[0] ** 2 + vec1[1] ** 2 + vec1[2] ** 2) * math.sqrt(vec2[0] ** 2 + vec2[1] ** 2 + vec2[2] ** 2) + 0.0001
    
    return numerator / denominator

def calculate_angle(vec1, vec2):
    return math.acos(calculate_cos(vec1, vec2))

def dis_to(pos1, pos2):
    return math.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2 + (pos1[2] - pos2[2]) ** 2)

def get_brake_acc(velocity, speed, isVehicle):
    
    brake_acc = None
    
    deceleration_acc = 6
    
    if not isVehicle:
        deceleration_acc = 1.5
    
    if speed == 0:
        brake_acc = np.array([deceleration_acc, deceleration_acc, deceleration_acc])
    else:
        clone = []
        
        for i in range(0, 3):
            ax = max(abs(velocity[i] / speed * deceleration_acc), 0.001)
            clone.append(ax)
            
        brake_acc = np.array(clone)
            
    return brake_acc

def get_vehicle_info(state, isVehicle):
    transform = state.transform
    speed = state.speed
    
    position = np.array([transform.position.x, transform.position.y, transform.position.z])
    rotation = np.array([transform.rotation.x, transform.rotation.y, transform.rotation.z])
    velocity = np.array([state.velocity.x, state.velocity.y, state.velocity.z])
    
    moving_direction = velocity / max(speed, 0.001)
    
    brake_acc = get_brake_acc(velocity, speed, isVehicle)
            
    return position, velocity, speed, brake_acc, moving_direction, rotation
    return {
        'state': {
            'pos': position,
            'vel': velocity,
            'speed': speed,
            'brk_acc': brake_acc,
            'mov_dir': moving_direction,
            'dir': rotation
        }
    }

def onLeft(veh_pos, obj_pos, obj_mov_dir):
    perpen_dir = np.array([obj_mov_dir[2], obj_mov_dir[1], - obj_mov_dir[0]])
    dis_vec = np.array([veh_pos[0] - obj_pos[0], veh_pos[1] - obj_pos[1], veh_pos[2] - obj_pos[2]])
    angle = calculate_angle(dis_vec, perpen_dir)
    if abs(angle) < math.pi/2:
        return False
    
    return True

def isFreeOfOncommingTraffic(ego, npc_ahd):
    
    for npc in npc_ahd:
        if np.dot(ego['state']['mov_dir'], npc['state']['mov_dir']) < 0:
            if ego['lane']['target'] == npc['lane']['cur'] and dis_to(ego['state']['position'], npc['state']['position']) <= free_of_traffic_range:
                return False
            
    return True

def adaptive_speed_limit(traffic, weather):
    return 40
