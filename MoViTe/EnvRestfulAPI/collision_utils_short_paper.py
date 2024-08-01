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
    velocity = np.array([state.velocity.x, state.velocity.y, state.velocity.z])
    
    moving_direction = velocity / max(speed, 0.001)
    
    brake_acc = get_brake_acc(velocity, speed, isVehicle)
            
    return position, velocity, speed, brake_acc

# @jit(nopython=True, fastmath=True)
def calculate_measures(npc_state, ego_state, isNpcVehicle):
    
    ego_position, ego_velocity, ego_speed, ego_brake_acc = get_vehicle_info(ego_state, True)

    print("EGO position: ", ego_position)
    print("EGO velocity: ", ego_velocity)
    print("EGO speed: ", ego_speed)
    print("EGO brake acc: ", ego_brake_acc)

    trajectory_ego_k, trajectory_ego_b = get_line(ego_position, ego_velocity)

    ETTC = 100
    distance = 100000
    loProC_list, laProC_list = [0], [0]
    proC_list = [0]
    reaction_time = 0.5
    
    for i in range(0, len(npc_state)):
        
        a_position, a_velocity, agent_speed, agent_brake_acc = get_vehicle_info(npc_state[i], isNpcVehicle[i])
        
        dis_vec = np.array([ego_position[0] - a_position[0], ego_position[1] - a_position[1], ego_position[2] - a_position[2]])
        
        trajectory_agent_k, trajectory_agent_b = get_line(a_position, a_velocity)
        
        # if judge_intersect_behind(ego_position, ego_velocity, a_position, a_velocity):
        #     continue

        # calculate DTO
        dis = get_distance(ego_position, a_position[0], a_position[2])
        distance = dis if dis <= distance else distance
        
        print("Sub distance: ", distance)

        # calculate ETTC
        same_lane, _, ttc = judge_same_line(ego_position, ego_speed, ego_velocity, a_position, agent_speed, trajectory_ego_k, trajectory_agent_k)

        time_ego = 100
        time_agent = 100

        if same_lane:
            time_ego = ttc
            time_agent = ttc
        else:
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
                
        if abs(time_ego - time_agent) < 1:
            ETTC = min(ETTC, (time_ego + time_agent) / 2)
            
        print("Sub ETTC: ", ETTC)

        # Calculate LoSD
        loSD = 100000

        if ego_velocity[0] * a_velocity[0] > 0:
            if ego_velocity[0] * (ego_position[0] - a_position[0]) < 0:
                loSD = 1 / 2 * (
                    abs(pow(ego_velocity[0], 2) / ego_brake_acc[0] - pow(a_velocity[0], 2) / agent_brake_acc[0])) + abs(ego_velocity[0]) * reaction_time
            else: 
                loSD = 1 / 2 * (
                    abs(pow(ego_velocity[0], 2) / ego_brake_acc[0] - pow(a_velocity[0], 2) / agent_brake_acc[0])) + abs(a_velocity[0]) * reaction_time
        else:
            loSD = 1 / 2 * (
                abs(pow(ego_velocity[0], 2) / ego_brake_acc[0] + pow(a_velocity[0], 2) / agent_brake_acc[0]))
        
        loProC = calculate_collision_probability(loSD, abs(ego_position[0] - a_position[0]))
        loProC_list.append(loProC)

        # Calculate LaSD
        laSD = 1000000
        
        if ego_velocity[2] * a_velocity[2] > 0:
            if ego_velocity[2] * (ego_position[2] - a_position[2]) < 0:
                laSD = 1 / 2 * (
                    abs(pow(ego_velocity[2], 2) / ego_brake_acc[2] - pow(a_velocity[2], 2) / agent_brake_acc[2])) + abs(ego_velocity[2]) * reaction_time
            else: 
                laSD = 1 / 2 * (
                    abs(pow(ego_velocity[2], 2) / ego_brake_acc[2] - pow(a_velocity[2], 2) / agent_brake_acc[2])) + abs(a_velocity[2]) * reaction_time
        else:
            laSD = 1 / 2 * (
                abs(pow(ego_velocity[2], 2) / ego_brake_acc[2] + pow(a_velocity[2], 2) / agent_brake_acc[2]))

        laProC = calculate_collision_probability(laSD, abs(ego_position[2] - a_position[2]))
        laProC_list.append(laProC)
        
        proC = (laProC + loProC) / 2
        
        proC_list.append(proC)
        
        print("Sub ProC: ", proC)

    # calculate collision probability
    proC_dt = max(proC_list) + (1 - max(proC_list)) * min(proC_list)
        
    return ETTC, distance, proC_dt


if __name__ == "__main__":
    print(calculate_collision_probability(10, 2))

    a = (1, 99, 3)
    print(a[2])