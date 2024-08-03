import math
import numpy as np
import random
from lgsvl.agent import NpcVehicle
from numba import jit
from violation_utils import Violation
from violation_utils_function import *

print('*' * 80)
print("Using violation utilities")
print('*' * 80)

NUM_VIOLATIONS = 60

# threshold
theta_acc = 3 # acceleration threshold distinguishes Acceleration, Deceleration and Steady Driving
theta_1 = math.pi / 36
theta_2 = math.pi / 6 
theta_backing = 5 # minimum total of npc vehicles's position deviation within 0.5s

def judge_basic_behavior(ego_acc, ego_mov_dir, cur_lane):
    
    global theta_acc
    global theta_1
    
    # EGO's Basic Behavior: (0) Acceleration, (1) Steady Driving, (2) Deceleration, (3) Direction Changing
    basic_behavior = [False, False, False, False]
    
    if ego_acc >= theta_acc:
        basic_behavior[0] = True
    elif abs(ego_acc) <= abs(theta_acc):
        basic_behavior[1] = True
    else:
        basic_behavior[2] = True
        
    lane_dir = np.array([lane['x'], lane['y'], lane['z']])
        
    dev_angle = calculate_angle(ego_mov_dir, lane_dir)
    
    if dev_angle > theta_1:
        basic_behavior[3] = True
        
    return basic_behavior   

# def judge_acceleration_violation(violation_prob, ego_speed, lane, traffic, weather):
    
#     # Violating a speed limit
#     if ego_speed > lane['speed_limit']:
#         violation_prob[Violation.VIO_SPD_LIM] = 1.0
#     else:
#         violation_prob[Violation.VIO_SPD_LIM] = ego_speed / lane['speed_limit']
        
#     # Violation of basic speed rule
#     apt_spd_lim = adaptive_speed_limit(traffic, weather)
#     if ego_speed > apt_spd_lim:
#         violation_prob[Violation.VIO_BSC_SPD] = 1.0
#     else:
#         violation_prob[Violation.VIO_BSC_SPD] = ego_speed / apt_spd_lim

#     return 'Judge successfully'

def judge_lane_change_violation():
    pass

def judge_turning_violation():
    pass

def judge_direction_changing_violation(violation_prob, lane, next_lane):
    
    global theta_2
    
    lane_dir = np.array([lane['x'], lane['y'], lane['z']])
    next_lane_dir = np.array([next_lane['x'], next_lane['y'], next_lane['z']])
    
    dev_angle = calculate_angle(lane_dir, next_lane_dir)
    
    if dev_angle < theta_2:
        judge_lane_change_violation()
    else:
        judge_turning_violation()
        
    judge_backing_violation()
    
    return "Judge successfully"

# passing behavior and violations

def judge_passing(ego, npc):
    return ((np.dot(ego['state']['mov_dir'], npc['state']['mov_dir']) > 0) and (ego['state']['speed'] > npc['state']['speed']))

def judge_passing_violation(violation_prob, ego, npc_ahd):
    
    isPassing = False
    
    isFreeOfTraffic = isFreeOfOncommingTraffic(ego, npc_ahd)
    
    for npc in range(0, len(npc_ahd)):
        
        if judge_passing(ego, npc):
            isPassing = True
            if onLeft(ego['state']['pos'], npc['state']['pos'], npc['state']['mov_dir']):
                violation_prob[DANGER_LFT_TURN] = max(violation_prob[DANGER_LFT_TURN], npc['distance']['vioRate'])
                
                if not ego['state']['hasClrView']:
                    violation_prob[DANGER_LFT_TURN] = 1.0 # ?
                    
                if not isFreeOfOncommingTraffic:
                    violation_prob[DANGER_LFT_TURN] = 1.0 # ?
                    
            else:
                
                
        else:
            continue
    
    if not isPassing:
        return "No Passing"
        
    return "Passing"

# backing behavior and violations

def judge_backing(ego_mov_dir, ego_dir):
    return (np.dot(ego_mov_dir, ego_dir) < 0)

def judge_backing_violation(violation_prob, ego, npc_back):
    
    if not judge_backing(ego['state']['mov_dir'], ego['state']['dir']):
        return "No Backing"
    
    sum_dev_pos = 0
    
    for npc in npc_back:
        
        violation_prob[ILL_BACK] = max(violation_prob[ILL_BACK], npc['distance']['vioRate'])
        
        sum_dev_pos += npc['dev_pos']
        
    if sum_dev_pos < theta_backing:
        violation_prob[ILL_BACK] = 1
        
    return "Backing"

def judge_overlap_violation():
    return "Overlap"

def judge_deceleration_violation():
    return "Deceleration"

def judge_cruising_violation():
    return "Cruising"

def judge_other_violation():
    return "Other"

# @jit(nopython=True, fastmath=True)
def calculate_violations(npc_state, ego_state, isNpcVehicle, prev_npc_pos, hasAClearView, current_signals, ego_curr_acc, prev_brake_percentage, brake_percentage, 
                       agent_uid, cur_lane, next_lane, p_lane_id, p_tlight_sign, 
                       orientation, mid_point = None, dis_tag = True):
    
    # get ego information from state
    ego = get_vehicle_info(ego_state)
    
    ego['state']['acc'] = ego_curr_acc
    ego['state']['hasClrView'] = hasAClearView
    ego['lane'] = {
        'cur': cur_lane,
        'target': next_lane
    }
    
    basic_behavior = judge_basic_behavior(ego['acc'], ego['mov_dir'], cur_lane)
    
    print(f"Check basic behavior results: \n Acceleration: {str(basic_behavior[0])} \n Steady Driving: {str(basic_behavior[1])} 
          \n Deceleration: {str(basic_behavior[2])} \n Direction Changing: {str(basic_behavior[3])}")
    
    violation_prob = np.zeros(NUM_VIOLATIONS)
    
    npc_list = []
    npc_back_list = []
    npc_ahd_list = []
    
    for i in range(0, len(npc_state)):
        
        # get npc's pos, vel, brk acc, v.v
        npc = get_vehicle_info(npc_state[i], isNpcVehicle[i])
        
        # Calculate LoSD
        loSD = 100000

        if ego['state']['vel'][0] * npc['state']['vel'][0] > 0:
            if ego['state']['vel'][0] * (ego['state']['pos'][0] - npc['state']['pos'][0]) < 0:
                loSD = 1 / 2 * (
                    abs(pow(ego['state']['vel'][0], 2) / ego['state']['brk_acc'][0] - pow(npc['state']['vel'][0], 2) / npc['state']['brk_acc'][0])) + abs(ego['state']['vel'][0]) * reaction_time
            else: 
                loSD = 1 / 2 * (
                    abs(pow(ego['state']['vel'][0], 2) / ego['state']['brk_acc'][0] - pow(npc['state']['vel'][0], 2) / npc['state']['brk_acc'][0])) + abs(npc['state']['vel'][0]) * reaction_time
        else:
            loSD = 1 / 2 * (
                abs(pow(ego['state']['vel'][0], 2) / ego['state']['brk_acc'][0] + pow(npc['state']['vel'][0], 2) / npc['state']['brk_acc'][0]))
        
        npc['distance']['loSD'] = loSD
        npc['distance']['loCD'] = abs(ego['state']['pos'][0] - npc['state']['pos'][0])
        npc['distance']['loVioRate'] = calculate_violation_rate(npc['distance']['loSD'], npc['distance']['lo7+CD'])

        # Calculate LaSD
        
        laSD = 1000000
        
        if ego['state']['vel'][2] * npc['state']['vel'][2] > 0:
            if ego['state']['vel'][2] * (ego['state']['pos'][2] - npc['state']['pos'][2]) < 0:
                laSD = 1 / 2 * (
                    abs(pow(ego['state']['vel'][2], 2) / ego['state']['brk_acc'][2] - pow(npc['state']['vel'][2], 2) / npc['state']['brk_acc'][2])) + abs(ego['state']['vel'][2]) * reaction_time
            else: 
                laSD = 1 / 2 * (
                    abs(pow(ego['state']['vel'][2], 2) / ego['state']['brk_acc'][2] - pow(npc['state']['vel'][2], 2) / npc['state']['brk_acc'][2])) + abs(npc['state']['vel'][2]) * reaction_time
        else:
            laSD = 1 / 2 * (
                abs(pow(ego['state']['vel'][2], 2) / ego['state']['brk_acc'][2] + pow(npc['state']['vel'][2], 2) / npc['state']['brk_acc'][2]))

        npc['distance']['laSD'] = laSD
        npc['distance']['laCD'] = abs(ego['state']['pos'][2] - npc['state']['pos'][2])
        npc['distance']['laVioRate'] = calculate_violation_rate(npc['distance']['laSD'], npc['distance']['laCD'])
        
        npc['distance']['vioRate'] = (npc['distance']['loVioRate'] + npc['distance']['laVioRate']) / 2
         
        # calculate npc's position deviation
        npc['dev_pos'] = math.sqrt((npc['state']['pos'][0] - prev_npc_pos[0]) ** 2 + (npc['state']['pos'][1] - prev_npc_pos[1]) ** 2 + (npc['state']['pos'][2] - prev_npc_pos[2]) ** 2) 
         
        if ego['state']['vel'][0] * (ego['state']['pos'][0] - npc['state']['pos'][0]) + ego['state']['vel'][2] * (ego['state']['pos'][2] - npc['state']['pos'][2]) > 0:
            npc_ahd_list.append(npc)
        else:
            npc_back_list.append(npc)
            
        npc_list.append(npc)
    
    if basic_behavior[0]:
        # judge_acceleration_violation(violation_prob, ego_speed, lane, traffic, weather)
        # print("Let's check Acceleration Violations")
            
        judge_passing_violation(violation_prob, ego, npc_ahd_list)
        print("Let's check Passsing Violations")
        
        # judge_overlap_violation(violation_prob)
        print("Let's check Overlap Violations")
        
    elif basic_behavior[1]:
        if not basic_behavior[3]:
            # judge_cruising_violation(violation_prob)
            print("Let's check Cruising Violations")
        
        # judge_overlap_violation(violation_prob)
        print("Let's check Overlap Violations")
    elif basic_behavior[2]:
        # judge_deceleration_violation(violation_prob)
        print("Let's check Deceleration Violations")
        
    if basic_behavior[3]:
        # judge_direction_changing_violation(violation_prob)
        print("Let's check Direction Changing Violations")
    
    # judge_other_violation(violation_prob)
    
    np_condition = [0, 0, 0, 0, 0, 0, 0]
    TTC = 0
    distance = 0
    proC_dt = 0
    
    return TTC, distance, proC_dt, p_tlight_sign, np_condition


if __name__ == "__main__":
    print(calculate_collision_probability(10, 2))

    a = (1, 99, 3)
    print(a[2])