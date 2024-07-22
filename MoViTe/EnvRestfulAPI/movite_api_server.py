import socket
import time

import cv2
from flask import Flask, request
import os
from datetime import timedelta
import json
import lgsvl
import numpy as np
import pickle
# from ScenarioCollector.createUtils import *
from collision_utils_origin_modified import pedestrian, npc_vehicle, calculate_measures
from clustering import cluster
import math
import threading
from lgsvl.agent import NpcVehicle
import random
import queue
import pandas as pd
import pickle
import shutil

import torch
from lgsvl.dreamview import CoordType
import requests
from numba import jit
from diversity_utils import merging_frame, clustering, calculate_diversity_level


########################################
observation_time = 6  # OTP [4, 6, 8, 10]

app = Flask(__name__)

app.config['SECRET_KEY'] = os.urandom(24)
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=7)

# create simulator

sim = lgsvl.Simulator(os.environ.get(
    "SIMUSaveStateLATOR_HOST", "localhost"), 8977)

DREAMVIEW = None

# init variable

equal_prob = True
isCalculateDiversity = True
VIOLATION_WEIGHT_DECAY = 0.1
UPDATE_WEIGHT_FREQ = 10
flexible_weight = False
update_counter = 0
collision_object = None
probability = 0
diversity_level = 0
time_step_collision_object = None
sensors = None
DATETIME_UNIX = None
WEATHER_DATA = None
TIMESTAMP = None
DESTINATION = None
EGO = None
CONSTRAINS = True
ROAD = '1'
SAVE_SCENARIO = False
REALISTIC = False
collision_tag = False
EFFECT_NAME = 'Default'
EPISODE = 0
MID_POINT = {0, 0, 0}
CONTROL = False
NPC_QUEUE = queue.Queue(maxsize=10)
collision_speed = 0  # 0 indicates there is no collision occurred.
collision_uid = "No collision"
lane_waypoint = {
    'point_f': {
        'x': 0,
        'y': 0,
        'z': 0
    },
    'point_l': {
        'x': 0,
        'y': 0,
        'z': 0
    },
    'lane_id': "",
}

next_lane_waypoint = {
    'point_f': {
        'x': 0,
        'y': 0,
        'z': 0
    },
    'point_l': {
        'x': 0,
        'y': 0,
        'z': 0
    },
    'lane_id': ""
}

clustering_timestamp = str(int(time.time()))

current_signals = {}

speed_list = []

cars = ['Jeep', 'BoxTruck', 'Sedan', 'SchoolBus', 'SUV', 'Hatchback']
colors = ['pink', 'yellow', 'red', 'white', 'black', 'skyblue']

u = 0.6
z_axis = lgsvl.Vector(0, 0, 100)
prefix = '/deepqtest/lgsvl-api/'

# setup connect to apollo

APOLLO_HOST = '112.137.129.158'  # or 'localhost'
PORT = 8966
DREAMVIEW_PORT = 9988
BRIDGE_PORT = 9090

msg_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_address = (APOLLO_HOST, PORT)

msg_socket.connect(server_address)

# import lane_information

map = 'tartu' # map: tartu, sanfrancisco, borregasave

lanes_map_file = "./map/{}_lanes.pkl".format(map)
lanes_map = None

with open(lanes_map_file, "rb") as file:
    lanes_map = pickle.load(file)

file.close()
    
signals_map_file = "./map/{}_signals.pkl".format(map)
signals_map = None

with open(signals_map_file, "rb") as file:
    signals_map = pickle.load(file)
    
file.close()
    
signals_params = {}

vioRate_reward = 0

prev_tlight_sign = {}
prev_lane_id = ""
brake_percentage_queue = []
brake_count = 0

time_stamp = str(int(time.time()))

violation_weight = [1/7, 1/7, 1/7, 1/7, 1/7, 1/7, 1/7]

violation_segment = []

# on collision callback function
def on_collision(agent1, agent2, contact):
    name1 = agent1.__dict__.get('name')
    name2 = agent2.__dict__.get('name') if agent2 is not None else "OBSTACLE"
    uid = agent2.__dict__.get('uid') if agent2 is not None else "OBSTACLE"
    print("{} collided with {} at {}".format(name1, name2, contact))
    global collision_object
    global collision_tag
    global collision_speed
    global collision_uid
    collision_uid = uid
    collision_object = name2
    collision_tag = True
    try:
        collision_speed = agent1.state.speed
    except KeyError:
        collision_speed = -1
        print('KeyError')

# check whether there is collision or not


def get_no_conflict_position(position, car):
    if car == 'BoxTruck' or car == 'SchoolBus':
        sd = 10
    else:
        sd = 8
    generate = True
    if CONSTRAINS:
        agents = sim.get_agents()
        for agent in agents:
            if math.sqrt(pow(position.x - agent.transform.position.x, 2) +
                         pow(position.y - agent.transform.position.y, 2) +
                         pow(position.z - agent.transform.position.z, 2)) < sd:
                generate = False
                break

    return generate

# set vehicles's color


def set_color(color):
    colorV = lgsvl.Vector(0, 0, 0)
    if color == 'black':
        colorV = lgsvl.Vector(0, 0, 0)
    elif color == 'white':
        colorV = lgsvl.Vector(1, 1, 1)
    elif color == 'yellow':
        colorV = lgsvl.Vector(1, 1, 0)
    elif color == 'pink':
        colorV = lgsvl.Vector(1, 0, 1)
    elif color == 'skyblue':
        colorV = lgsvl.Vector(0, 1, 1)
    elif color == 'red':
        colorV = lgsvl.Vector(1, 0, 0)
    elif color == 'green':
        colorV = lgsvl.Vector(0, 1, 0)
    elif color == 'blue':
        colorV = lgsvl.Vector(0, 0, 1)
    return colorV

# control number of agents


def control_agents_density(agent):
    if CONTROL:

        if NPC_QUEUE.full():
            sim.remove_agent(NPC_QUEUE.get())
            NPC_QUEUE.put(agent)
        else:
            NPC_QUEUE.put(agent)

# get type of obstacles


def get_type(class_name):
    # print(class_name)
    object_type = None
    if str(class_name) == '<class \'lgsvl.agent.EgoVehicle\'>':
        object_type = 'Ego'
    elif str(class_name) == '<class \'lgsvl.agent.Pedestrian\'>':
        object_type = 'Pedestrian'
    elif str(class_name) == '<class \'lgsvl.agent.NpcVehicle\'>':
        object_type = 'NPC'
    return object_type

# update violation weights
def update_violation_weight(violation_list):
    
    global violation_weight
    global VIOLATION_WEIGHT_DECAY

    reduced_part = 0
    
    reduced_info = [1, 1, 1, 1, 1, 1, 1]
    
    rest = 7
    
    for i in range(0, 7):
        if violation_list[i] == float(1):
            reduced_info[i] = -1
            rest -= 1
            reduced_part += violation_weight[i] * VIOLATION_WEIGHT_DECAY
            violation_weight[i] = violation_weight[i] * (1 - VIOLATION_WEIGHT_DECAY)
            
    for i in range(0, 7):
        if reduced_info[i] == 1:
            violation_weight[i] += reduced_part / rest

# calculate measures thread, use in multi-thread
def calculate_measures_thread(state_list, ego_state, isNpcVehicle, TTC_list, vioRate_list, agent_uid, frame_list, sub_frame_list,
                              distance_list, probability_list, current_signals, ego_curr_acc, prev_brake_percentage, brake_percentage, 
                              road, next_road, p_lane_id, prev_tlight_sign_, orientation, mid_point=None, collision_tag_=False):

    p_tlight_sign = prev_tlight_sign_
    
    # print(prev_tlight_sign_)

    TTC, distance, probability2, tlight_sign, vioRate = calculate_measures(
        state_list, ego_state, isNpcVehicle, current_signals, ego_curr_acc, prev_brake_percentage, brake_percentage, agent_uid,
        road, next_road, p_lane_id, p_tlight_sign, orientation, mid_point, True)

    prev_tlight_sign_ = tlight_sign
    
    # print(prev_tlight_sign_)    

    TTC_list.append(round(TTC, 6))

    distance_list.append(round(distance, 6))
    if collision_tag_:
        probability2 = 1
        
        is_other = 1
        
        for i in range(0, 6):
            if vioRate[i] > 0:
                vioRate[i] = 1.0
                is_other = 0
                
        vioRate[6] = 1.0 * is_other
    probability_list.append(round(probability2, 6))
    
    vioRate_list.append(vioRate)
    
    if isCalculateDiversity:
        for i in range(0, len(sub_frame_list)):
            for j in range(0, 7):
                if vioRate[j] == 1.0:
                    sub_frame_list[i].append(1)
                else:
                    sub_frame_list[i].append(0)
                    
            if probability2 == 1:
                sub_frame_list[i].append(1)
            else:
                sub_frame_list[i].append(0)
                    
        frame_list += sub_frame_list

def cal_dis_position(ego, agent):
    return math.sqrt((ego.x - agent.x) ** 2 + (ego.y - agent.y) ** 2 + (ego.z - agent.z) ** 2)        
        
def collecting_data(sim):
    print(10*'*', "Collecting data to create frame", 10*'*')
    agents = sim.get_agents()
    ego = agents[0]
    
    ego_rotation = ego.state.rotation
    ego_velocity = ego.state.velocity
    ego_position = ego.state.position
    
    obs_info = []
    
    for i in range(1, len(agents)):
        a_position = agents[i].state.position
        a_velocity = agents[i].state.velocity
        a_rotation = agents[i].state.rotation
        
        dis_to_ego = cal_dis_position(ego_position, a_position)
        
        type = 1 if isinstance(agents[i], NpcVehicle) else 0 
        
        obs_info.append({
            'dis_to_ego': dis_to_ego,
            'rotation': a_rotation,
            'velocity': a_velocity,
            'type': type
        })
        
    weather = sim.weather
    time = sim.time_of_day
    signal = sim.get_controllable(ego_position, "signal").current_state
    
    control_info, chassis_info = get_apollo_ctrl_msg()
    
    ego_acceleration = control_info['acceleration']
    ego_brake = chassis_info['brake_percentage']
    
    sorted_obs = sorted(obs_info, key=lambda x: x["dis_to_ego"], reverse=True)
    
    # print("Ego Position: ", ego_position)
    # print("Ego Rotation: ", ego_rotation)
    # print("Ego Velocity: ", ego_velocity)
    # print("Ego Acceleration: ", ego_acceleration)
    # print("Ego Brake Percentage: ", ego_brake)
    # print("Obstacle info: ", sorted_obs)
    # print("Weather: ", weather)
    # print("Time: ", time)
    # print("Traffic light: ", signal)
    
    state = []
    # ego rotation
    state.append(ego_rotation.x)
    state.append(ego_rotation.y)
    state.append(ego_rotation.z)
    
    # ego velocity
    state.append(ego_velocity.x)
    state.append(ego_velocity.y)
    state.append(ego_velocity.z)
    
    # ego acceleration
    state.append(ego_acceleration)
    
    # ego brake
    state.append(ego_brake)
    
    for i in range(0, min(len(sorted_obs), 2)):
        # dis from obs to ego
        state.append(sorted_obs[i]['dis_to_ego'])
        
        # obs rotation
        state.append(sorted_obs[i]['rotation'].x)
        state.append(sorted_obs[i]['rotation'].y)
        state.append(sorted_obs[i]['rotation'].z)
        
        # obs velocity
        state.append(sorted_obs[i]['velocity'].x)
        state.append(sorted_obs[i]['velocity'].y)
        state.append(sorted_obs[i]['velocity'].z)
        
        # obs type
        state.append(sorted_obs[i]['type'])
    
    # add default value if not enough 3 obstacles
    if len(state) < 24:
        while len(state) < 24:
            state.append(0)
            
    # weather state
    state.append(weather.rain * 100 + weather.fog * 10 + weather.wetness)
    # state.append(weather.fog)
    # state.append(weather.wetness)
    
    # time state
    state.append(time)
    
    # traffic light state
    if signal == 'green':
        state.append(0)
    elif signal == 'red':
        state.append(1)
    elif signal == 'yellow':
        state.append(2)
    else:
        state.append(-1)    
    
    # print(state)
    
    return state

# calculate metrics
def calculate_metrics(agents, ego):
    global probability
    global DATETIME_UNIX
    global collision_tag

    global SAVE_SCENARIO

    global collision_object
    global collision_speed
    global TIMESTAMP
    global MID_POINT
    global collision_uid
    global lane_waypoint
    global next_lane_waypoint
    global vioRate_reward
    global prev_tlight_sign
    global prev_lane_id
    global violation_weight
    global equal_prob
    global brake_percentage_queue
    global brake_count
    global diversity_level
    global flexible_weight
    global isCalculateDiversity
    global clustering_timestamp
    
    diversity_level = 0

    collision_object = None
    collision_speed = 0  # 0 indicates there is no collision occurred.
    collision_speed_ = 0
    collision_type = "None"
    collision_uid = "No collision"

    print("Calculating metrics ....")

    TTC_list = []
    distance_list = []
    probability_list = []
    vioRate_list = []
    frame_list = []
    i = 0
    time_step = 0.5
    sliding_step = 0.125
    speed = 0

    if SAVE_SCENARIO:
        doc, root = initialization(
            '2022-11-11', get_time_stamp(), './{}.json'.format(EFFECT_NAME))
        entities, storyboard = initializeStory(agents, doc, root)
        story = doc.createElement('Story')
        story.setAttribute('name', 'Default')

    while i < observation_time / time_step:
        
        check_modules_status()
        
        num_of_frame = int(time_step / sliding_step)
        
        sub_frame_list = []
        
        # print("Number of frame", num_of_frame)
        
        for j in range(0, num_of_frame):
            sim.run(time_limit=sliding_step)
            if isCalculateDiversity:
                frame = collecting_data(sim)
                sub_frame_list.append(frame) 
        
        # sim.run(time_limit=time_step)  # , time_scale=2
        
        ego_curr_acc, brake_percentage, orientation = get_sub_state()
        
        if len(brake_percentage_queue) == 0:
            brake_percentage_queue.append(0)
            brake_percentage_queue.append(0)
            brake_percentage_queue.append(0)
            brake_percentage_queue.append(0)
        
        prev_brake_percentage = brake_percentage_queue[brake_count % 4]

        state_list = []
        isNpcVehicle = []
        agent_uid = []
        for j in range(1, len(agents)):
            agent_uid.append(agents[j].uid)
            state_ = agents[j].state
            state_list.append(state_)
            isNpc = (isinstance(agents[j], NpcVehicle))
            isNpcVehicle.append(isNpc)

        ego_state = ego.state

        road = {
            'x': lane_waypoint['point_l']['x'] - lane_waypoint['point_f']['x'],
            'y': lane_waypoint['point_l']['y'] - lane_waypoint['point_f']['y'],
            'z': lane_waypoint['point_l']['z'] - lane_waypoint['point_f']['z'],
            'lane_id': lane_waypoint['lane_id']
        }
        
        next_road = {
            'x': next_lane_waypoint['point_l']['x'] - next_lane_waypoint['point_f']['x'],
            'y': next_lane_waypoint['point_l']['y'] - next_lane_waypoint['point_f']['y'],
            'z': next_lane_waypoint['point_l']['z'] - next_lane_waypoint['point_f']['z'],
            'lane_id': next_lane_waypoint['lane_id']
        }
        
        if prev_lane_id == "":
            p_lane_id = road['lane_id']
        else:        
            p_lane_id = prev_lane_id
            
        prev_lane_id = road['lane_id']
        
        thread = threading.Thread(
            target=calculate_measures_thread,
            args=(state_list, ego_state, isNpcVehicle, TTC_list, vioRate_list, agent_uid, frame_list, sub_frame_list, distance_list, 
                  probability_list, current_signals, ego_curr_acc, prev_brake_percentage, brake_percentage, 
                  road, next_road, p_lane_id, prev_tlight_sign, orientation, MID_POINT, collision_tag,)
        )

        thread.start()

        if collision_tag:
            collision_tag = False

        i += 1
        
        brake_percentage_queue[brake_count % 4] = brake_percentage
        brake_count = (brake_count + 1) % 4
        

    # if SAVE_SCENARIO:
    collision_type, collision_speed_, collision_uid_ = get_collision_info()
    if collision_speed_ == -1:
        collision_speed_ = speed
    
    probability = round(max(probability_list), 6)
    
    transposed_vio = zip(*vioRate_list)
    max_values = [max(column) for column in transposed_vio]
    
    print("Violation rate: ", max_values)
    
    cnt_vio = 0
    total_rate = 0
    
    isViolation = False
    
    violation_reward_list = []
    
    for i in range (0, len(max_values)):
        if max_values[i] > 0:
            cnt_vio += 1
            addition = 0.0
            
            if float(max_values[i]) == 1.0:
                addition = 1.0
            
            if flexible_weight:
                total_rate += (max_values[i] + addition) * violation_weight[i]
                violation_reward_list.append((max_values[i] + addition) * violation_weight[i])
            else:
                total_rate += (max_values[i] + addition)
                violation_reward_list.append(max_values[i] + addition)
                
            if float(max_values[i]) == 1.0:
                isViolation = True
        else:
            violation_reward_list.append(0)
    
    if cnt_vio == 0:
        cnt_vio = 1
        
    max_vioReward = max(violation_reward_list)
    avg_rest_vioReward = (total_rate - max_vioReward) / (max(cnt_vio - 1, 1))
    
    vioRate_reward = max_vioReward + (2 - max_vioReward) * avg_rest_vioReward
    
    # print("Merged Frame List: ")
    
    if flexible_weight:
        update_violation_weight(max_values)
    
    if isViolation and isCalculateDiversity:
        merging_frame(frame_list, clustering_timestamp)
        cluster_result = clustering(clustering_timestamp)
        diversity_level = calculate_diversity_level(cluster_result)
    
    # print(merged_frame_list)
    
    return {'TTC': TTC_list, 'distance': distance_list, 'collision_type': collision_type, 'collision_uid': collision_uid_,
            'collision_speed': collision_speed_, 'probability': probability_list, 'vioRate': max_values}  # 'uncomfortable': uncomfortable,


@app.route('/LGSVL')
def index():
    return "RESTful APIs for LGSVL simulator control."


@app.route('/LGSVL/get-datetime', methods=['GET'])
def get_time_stamp():
    return json.dumps(int(time.time()))


@app.route('/LGSVL/Episode', methods=['POST'])
def get_effect_info():
    global EPISODE
    EPISODE = int(request.args.get('episode'))

    return 'set effect-episode'


@app.route('/LGSVL/ego/collision_info', methods=['GET'])
def get_collision_info():
    """
    three types of collision: obstacle, NPC vehicle, pedestrian, None(no collision)
    :return:
    """
    global collision_object
    global collision_speed
    global collision_uid
    global JERK

    collision_info = str(collision_object)
    collision_speed_ = collision_speed

    collision_object = None
    collision_speed = 0
    JERK = 0
    collision_type = 'None'
    if collision_info == 'OBSTACLE':
        collision_type = "obstacle"
    if collision_info in npc_vehicle:
        collision_type = "npc_vehicle"
    if collision_info in pedestrian:
        collision_type = "pedestrian"
    return collision_type, collision_speed_, collision_uid


@app.route('/LGSVL/SetObTime', methods=['POST'])
def set_time():
    global observation_time
    observation_time = int(request.args.get('observation_time'))
    print(observation_time)
    return 'get time'


"""
Command APIs
"""


@app.route('/LGSVL/LoadScene', methods=['POST'])
def load_scene():
    global sensors
    global EGO
    global ROAD
    global prev_tlight_sign
    global prev_lane_id
    global DREAMVIEW
    global brake_count
    global brake_percentage_queue
    prev_lane_id = ""
    prev_tlight_sign = {}
    brake_count = 0
    brake_percentage_queue = []

    print('obTime: ', observation_time)
    scene = str(request.args.get('scene'))
    road_num = str(request.args.get('road_num'))
    ROAD = str(road_num)
    if sim.current_scene == scene:
        sim.reset()
    else:
        sim.load(scene)

    EGO = None
    state = lgsvl.AgentState()
    roadTransform_start = open(
        'Transform/transform-road' + road_num + '-start', 'rb')
    state.transform = torch.load(
        './Transform/{}.pt'.format("road" + "4" + "_start"))
    if road_num == '1':
        if scene == 'bd77ac3b-fbc3-41c3-a806-25915c777022':
            state.transform.position.x = 213.8
            state.transform.position.y = 35.7
            state.transform.position.z = 122.8
            state.transform.rotation.y = 49
            state.transform.rotation.x = 0
        elif scene == '12da60a7-2fc9-474d-a62a-5cc08cb97fe8':
            state.transform.position.x = -768.9
            state.transform.position.y = 10.2
            state.transform.position.z = 224.1
            state.transform.rotation.y = 81
            state.transform.rotation.x = 0
        else:
            state.transform.position.x = -40.3
            state.transform.position.y = -1.4
            state.transform.position.z = -11.8
            state.transform.rotation.y = 105
            state.transform.rotation.x = 1
    elif road_num == '2':
        state.transform.position.x = -442.1
        state.transform.position.y = 10.2
        state.transform.position.z = -65.1
        state.transform.rotation.y = 170
        state.transform.rotation.x = 0
    elif road_num == '3':
        state.transform.position.x = -62.7
        state.transform.position.y = 10.2
        state.transform.position.z = -110.2
        state.transform.rotation.y = 224
        state.transform.rotation.x = 0

    forward = lgsvl.utils.transform_to_forward(state.transform)

    state.velocity = 3 * forward

    EGO = sim.add_agent("8e776f67-63d6-4fa3-8587-ad00a0b41034",
                        lgsvl.AgentType.EGO, state)
    EGO.connect_bridge(os.environ.get("BRIDGE_HOST", APOLLO_HOST), BRIDGE_PORT)
    
    DREAMVIEW = lgsvl.dreamview.Connection(sim, EGO, APOLLO_HOST, str(DREAMVIEW_PORT))

    sensors = EGO.get_sensors()
    sim.get_agents()[0].on_collision(on_collision)

    data = {'road_num': road_num}

    if road_num == '1':
        if scene == 'bd77ac3b-fbc3-41c3-a806-25915c777022':
            requests.post(
                "http://localhost:8933/LGSVL/SetDestination?des_x=338.4&des_y=35.5&des_z=286.9")

        elif scene == '12da60a7-2fc9-474d-a62a-5cc08cb97fe8':
            requests.post(
                "http://localhost:8933/LGSVL/SetDestination?des_x=-494.3&des_y=10.2&des_z=294.7")
        else:
            requests.post(
                "http://localhost:8933/LGSVL/SetDestination?des_x=348.2&des_y=-7.5&des_z=-64.4")
    elif road_num == '2':
        requests.post(
            "http://localhost:8933/LGSVL/SetDestination?des_x=-384.6&des_y=10.2&des_z=-357.8")
    elif road_num == '3':
        requests.post(
            "http://localhost:8933/LGSVL/SetDestination?des_x=-208.2&des_y=10.2&des_z=-181.6")

    print(road_num)

    roadTransform_start.close()
    sim.run(2)

    return 'load success'


@app.route('/LGSVL/SaveTransform', methods=['POST'])
def save_transform():
    transform = sim.get_agents()[0].state.transform
    f = open('Transform/transform2', 'wb')
    pickle.dump(transform, f)
    return 'saved'


"""
Sim run
"""


@app.route('/LGSVL/Run', methods=['POST'])
def run():
    sim.run(8)
    return 'sim run'


"""
Randomly Load Agents
"""


@app.route('/LGSVL/LoadNPCVehicleRandomly', methods=['POST'])
def load_npc_vehicle_randomly():
    sim.add_random_agents(lgsvl.AgentType.NPC)
    return 'NPC Loaded'


@app.route('/LGSVL/LoadPedestriansRandomly', methods=['POST'])
def load_pedestrians_randomly():
    sim.add_random_agents(lgsvl.AgentType.PEDESTRIAN)
    sim.run(6)
    return "Pedestrians Loaded"


@app.route('/LGSVL/Reset', methods=['POST'])
def reset_env():
    state = lgsvl.AgentState()
    state.transform = sim.get_agents()[0].state.transform
    sim.reset()
    ego = sim.add_agent("8e776f67-63d6-4fa3-8587-ad00a0b41034",
                        lgsvl.AgentType.EGO, state)
    ego.connect_bridge(os.environ.get("BRIDGE_HOST", APOLLO_HOST), BRIDGE_PORT)
    global sensors
    sensors = ego.get_sensors()
    sim.get_agents()[0].on_collision(on_collision)
    sim.run(6)
    return "reset"


"""
Reset Agent
"""


@app.route('/LGSVL/EGOVehicle/Reset', methods=['POST'])
def clear_env():
    agents = sim.get_agents()

    for i in range(1, len(agents)):
        sim.remove_agent(agents[i])

    sim.run(6)
    return 'reset'


@app.route('/LGSVL/SaveState', methods=['POST'])
def save_state():
    state_id = str(request.args.get('ID'))

    agents = sim.get_agents()
    count_ego = 0
    count_npc = 0
    count_pedestrian = 0

    states_dict = {}

    weather_dict = {}

    weather_dict.update(
        {'rain': sim.weather.rain, 'fog': sim.weather.fog, 'wetness': sim.weather.wetness, 'time': sim.time_of_day})

    for agent in agents:
        obj_name = "None"
        obj_uid = agent.uid
        print(obj_uid, type(agent.uid))
        obj_color_vector = "Vector(1, 1, 0)"
        obj_type = get_type(agent.__class__)
        if obj_type == 'Ego':
            obj_name = 'Ego' + str(count_ego)
            count_ego += 1
        elif obj_type == 'NPC':
            obj_name = 'NPC' + str(count_npc)
            count_npc += 1
        elif obj_type == 'Pedestrian':
            obj_name = 'Pedestrian' + str(count_pedestrian)
        model = agent.name

        agent_dict = {}
        agent_dict.update({'model': model, 'name:': obj_name,
                          'obj_color': obj_color_vector})
        agent_dict.update({'positionX': agent.state.position.x, 'positionY': agent.state.position.y,
                           'positionZ': agent.state.position.z})
        agent_dict.update({'rotationX': agent.state.rotation.x, 'rotationY': agent.state.rotation.y,
                           'rotationZ': agent.state.rotation.z})
        agent_dict.update({'velocityX': agent.state.velocity.x, 'velocityY': agent.state.velocity.y,
                           'velocityZ': agent.state.velocity.z})
        agent_dict.update(
            {'angularVelocityX': agent.state.angular_velocity.x, 'angularVelocityY': agent.state.angular_velocity.y,
             'angularVelocityZ': agent.state.angular_velocity.z})

        states_dict.update({obj_uid: agent_dict})

    states_dict.update({'worldEffect': weather_dict})

    b = json.dumps(states_dict, indent=4)
    file = open('state/current_state_{}.json'.format(state_id), 'w')
    file.write(b)
    file.close()
    return 'save successfully'

@app.route('/LGSVL/SaveViolationWeight', methods=['POST'])
def save_violation_weight():
    global violation_weight
    global map
    
    eps = str(request.args.get('eps'))
    
    violation_weight_file = './violation_weight/{}_violation_weight_{}.pkl'.format(map, eps)
    
    with open(violation_weight_file, 'wb') as file:
        pickle.dump(violation_weight, file)
        
    return 'Save successfully'
        

@app.route('/LGSVL/LoadViolationWeight', methods=['POST'])
def load_violation_weight():
    global violation_weight
    global map
    
    eps = str(request.args.get('eps'))
    
    violation_weight_file = './violation_weight/{}_violation_weight_{}.pkl'.format(map, eps)

    if os.path.exists(violation_weight_file):
        with open(violation_weight_file, "rb") as file:
            violation_weight = pickle.load(file)
            
        file.close()
        
    return 'Load successfully'
    
@app.route('/LGSVL/SetMode', methods=['POST'])
def set_mode():
    global isCalculateDiversity
    global flexible_weight
    
    mode = str(request.args.get('mode'))
    
    if mode == 'flexible':
        flexible_weight = True
        isCalculateDiversity = False
    elif mode == 'diversity':
        isCalculateDiversity = True
        flexible_weight = False
    elif mode == 'basic':
        flexible_weight = False
        isCalculateDiversity = False
    else:
        flexible_weight = True
        isCalculateDiversity = True
        
    print(10*'*', 'Set Mode:', mode, 10*'*')
        
    return 'Set mode successfully'

@app.route('/LGSVL/SaveMergedState', methods=['POST'])
def save_merged_state():
    global map
    global clustering_timestamp
    
    eps = str(request.args.get('eps'))
    
    merged_state_file = './merged_state/{}_merged_state_{}.csv'.format(map, eps)
    
    source_file = './merged_state/merged_state_{}.csv'.format(clustering_timestamp)
    
    shutil.copy(source_file, merged_state_file)
        
    return 'Save successfully'

@app.route('/LGSVL/LoadMergedState', methods=['POST'])
def load_merged_state():
    global map
    global clustering_timestamp
    
    eps = str(request.args.get('eps'))
    
    merged_state_file = './merged_state/{}_merged_state_{}.csv'.format(map, eps)
    
    current_file = './merged_state/merged_state_{}.csv'.format(clustering_timestamp)
    
    shutil.copy(merged_state_file, current_file)
        
    return 'Load successfully'

@app.route('/LGSVL/SetDestination', methods=['POST'])
def set_destination():
    global DESTINATION
    global EGO
    global DREAMVIEW
    x = float(request.args.get('des_x'))
    y = float(request.args.get('des_y'))
    z = float(request.args.get('des_z'))
    print("x y z: ", x, y, z)
    DREAMVIEW.set_destination(x, z, y, coord_type=CoordType.Unity)
    return 'set destination.'


@app.route('/LGSVL/RollBack', methods=['POST'])
def roll_back():
    state_ID = str(request.args.get('ID'))
    state = open('state/current_state_{}.json'.format(state_ID), 'r')
    content = state.read()
    state_ = json.loads(content)
    sim.weather = lgsvl.WeatherState(rain=state_['worldEffect']['rain'], fog=state_['worldEffect']['fog'],
                                     wetness=state_['worldEffect']['wetness'])
    sim.set_time_of_day(state_['worldEffect']['time'])

    for agent in sim.get_agents():
        if agent.uid not in state_.keys():
            sim.remove_agent(agent)
            continue
        agent_state = state_[agent.uid]
        position = lgsvl.Vector(float(agent_state['positionX']), float(agent_state['positionY']),
                                float(agent_state['positionZ']))
        rotation = lgsvl.Vector(float(agent_state['rotationX']), float(agent_state['rotationY']),
                                float(agent_state['rotationZ']))
        velocity = lgsvl.Vector(float(agent_state['velocityX']), float(agent_state['velocityY']),
                                float(agent_state['velocityZ']))
        angular_velocity = lgsvl.Vector(float(agent_state['angularVelocityX']), float(agent_state['angularVelocityY']),
                                        float(agent_state['angularVelocityZ']))
        state = lgsvl.AgentState()
        state.transform.position = position
        state.transform.rotation = rotation
        state.velocity = velocity
        state.angular_velocity = angular_velocity
        agent.state = state

    return 'rollback'


"""
Set weather effect
"""


@app.route('/LGSVL/Control/Weather/Nice', methods=['POST'])
def nice():
    global REALISTIC
    sim.weather = lgsvl.WeatherState(rain=0, fog=0, wetness=0)
    REALISTIC = False
    print('realistic constraints: ', REALISTIC)

    agents = sim.get_agents()
    ego = agents[0]

    return calculate_metrics(agents, ego)


@app.route('/LGSVL/Control/Weather/Rain', methods=['POST'])
def rain():
    """
    three parameters: Light, Moderate and Heavy,
    apparently, wetness will be influenced.
    :return:
    """
    global REALISTIC
    rain_level = request.args.get('rain_level')
    r_level = 0
    w_level = 0
    if rain_level == 'Light':
        r_level = 0.2
        w_level = 0.2

    elif rain_level == 'Moderate':
        r_level = 0.5
        w_level = 0.5
    elif rain_level == 'Heavy':
        r_level = 1
        w_level = 1
    sim.weather = lgsvl.WeatherState(rain=r_level, fog=0, wetness=w_level)
    REALISTIC = False
    print('realistic constraints: ', REALISTIC)

    agents = sim.get_agents()
    ego = agents[0]

    return calculate_metrics(agents, ego)


@app.route('/LGSVL/Control/Weather/Fog', methods=['POST'])
def fog():
    """
    three parameters: Light, Moderate and Heavy
    :return:
    """
    global REALISTIC
    fog_level = request.args.get('fog_level')
    f_level = 0
    if fog_level == 'Light':
        f_level = 0.2
    elif fog_level == 'Moderate':
        f_level = 0.5
    elif fog_level == 'Heavy':
        f_level = 1
    sim.weather = lgsvl.WeatherState(rain=0, fog=f_level, wetness=0)
    REALISTIC = False
    print('realistic constraints: ', REALISTIC)

    agents = sim.get_agents()
    ego = agents[0]
    return calculate_metrics(agents, ego)


@app.route('/LGSVL/Control/Weather/Wetness', methods=['POST'])
def wetness():
    """
    three parameters: Light, Moderate and Heavy
    :return:
    """
    global REALISTIC
    wetness_level = request.args.get('wetness_level')
    w_level = 0
    if wetness_level == 'Light':
        w_level = 0.2
    elif wetness_level == 'Moderate':
        w_level = 0.5
    elif wetness_level == 'Heavy':
        w_level = 1
    sim.weather = lgsvl.WeatherState(rain=0, fog=0, wetness=w_level)
    REALISTIC = False
    print('realistic constraints: ', REALISTIC)
    agents = sim.get_agents()
    ego = agents[0]

    return calculate_metrics(agents, ego)


"""
Set time of day
"""


@app.route('/LGSVL/Control/TimeOfDay', methods=['POST'])
def time_of_day():
    """
    three parameters: Morning(10), Noon(14), Evening(20)
    :return:
    """
    global REALISTIC
    time = request.args.get('time_of_day')
    day_time = 10  # initial time: 10
    if time == 'Morning':
        day_time = 10
    elif time == 'Noon':
        day_time = 14
    elif time == 'Evening':
        day_time = 20
    sim.set_time_of_day(day_time, fixed=True)
    REALISTIC = False
    print('realistic constraints: ', REALISTIC)

    agents = sim.get_agents()
    ego = agents[0]
    return calculate_metrics(agents, ego)


"""
Control Agents
"""


@app.route(prefix + 'agents/npc-vehicle/cross-road', methods=['POST'])
def add_npc_cross_road():

    global NPC_QUEUE
    global cars
    global colors
    which_car = cars[random.randint(0, 5)]
    color = colors[random.randint(0, 5)]
    change_lane = int(request.args.get('maintainlane'))
    distance = str(request.args.get('position'))
    colorV = set_color(color)

    ego_transform = sim.get_agents()[0].state.transform
    sx = ego_transform.position.x
    sy = ego_transform.position.y
    sz = ego_transform.position.z

    angle = math.pi
    dist = 20 if distance == 'near' else 50

    point = lgsvl.Vector(sx + dist * math.cos(angle),
                         sy, sz + dist * math.sin(angle))
    state = lgsvl.AgentState()
    state.transform = sim.map_point_on_lane(point)

    generate = get_no_conflict_position(state.position, which_car)
    if generate:
        npc = sim.add_agent(which_car, lgsvl.AgentType.NPC, state, colorV)
        npc.follow_closest_lane(True, 20)
        npc.change_lane(change_lane == 1)

        control_agents_density(npc)
    agents = sim.get_agents()
    ego = agents[0]
    return calculate_metrics(agents, ego)


@app.route(prefix + 'agents/pedestrian/cross-road', methods=['POST'])
def add_pedestrian_cross_road():

    global NPC_QUEUE
    direction = request.args.get('direction')
    ego_transform = sim.get_agents()[0].state.transform
    forward = lgsvl.utils.transform_to_forward(ego_transform)
    right = lgsvl.utils.transform_to_right(ego_transform)

    npc_state = lgsvl.AgentState()

    if direction == 'left':
        offset = - 5.0 * right
    else:
        offset = 5.0 * right

    wp = [lgsvl.WalkWaypoint(sim.map_point_on_lane(ego_transform.position + offset + 30 * forward).position, 1),
          lgsvl.WalkWaypoint(sim.map_point_on_lane(ego_transform.position - offset + 30 * forward).position, 1)]

    npc_state.transform.position = sim.map_point_on_lane(
        ego_transform.position + offset + 30.0 * forward).position

    generate = get_no_conflict_position(
        npc_state.transform.position, 'pedestrian')
    if generate:
        name = pedestrian[random.randint(0, 8)]
        p = sim.add_agent(name, lgsvl.AgentType.PEDESTRIAN, npc_state)
        p.follow(wp, loop=False)

        control_agents_density(p)

    agents = sim.get_agents()
    ego = agents[0]
    return calculate_metrics(agents, ego)


@app.route(prefix + 'agents/npc-vehicle/drive-ahead', methods=['POST'])
def add_npc_drive_ahead():

    global NPC_QUEUE
    global cars
    global colors
    which_lane = request.args.get('lane')
    which_car = cars[random.randint(0, 5)]
    color = colors[random.randint(0, 5)]
    change_lane = int(request.args.get('maintainlane'))
    distance = str(request.args.get('position'))
    colorV = set_color(color)

    ego_transform = sim.get_agents()[0].state.transform
    forward = lgsvl.utils.transform_to_forward(ego_transform)
    right = lgsvl.utils.transform_to_right(ego_transform)

    if distance == 'near':
        offset = 10
        if which_car == 'BoxTruck' or which_car == 'SchoolBus':
            forward = 15 * forward
            right = 4 * right
            speed = 10
        else:
            forward = 12 * forward
            right = 4 * right
            speed = 10
    else:
        offset = 50
        if which_car == 'BoxTruck' or which_car == 'SchoolBus':
            forward = 50 * forward
            right = 4 * right
            speed = 10
        else:
            forward = 50 * forward
            right = 4 * right
            speed = 10

    if which_lane == "left":
        point = ego_transform.position - right + forward
    elif which_lane == "right":
        point = ego_transform.position + right + forward
    elif which_lane == "current":
        point = ego_transform.position + forward
    else:
        point = lgsvl.Vector(ego_transform.position.x + offset * math.cos(0), ego_transform.position.y,
                             ego_transform.position.z + offset * math.sin(0))

    npc_state = lgsvl.AgentState()
    npc_state.transform = sim.map_point_on_lane(point)

    generate = get_no_conflict_position(npc_state.position, which_car)
    if generate:
        npc = sim.add_agent(which_car, lgsvl.AgentType.NPC, npc_state, colorV)
        npc.follow_closest_lane(True, speed)
        npc.change_lane(change_lane == 1)

        control_agents_density(npc)

    agents = sim.get_agents()
    ego = agents[0]
    return calculate_metrics(agents, ego)


@app.route(prefix + 'agents/npc-vehicle/overtake', methods=['POST'])
def add_npc_overtake():
    global NPC_QUEUE
    global cars
    global colors
    which_lane = request.args.get('lane')
    which_car = cars[random.randint(0, 5)]
    color = colors[random.randint(0, 5)]
    change_lane = int(request.args.get('maintainlane'))
    distance = str(request.args.get('position'))
    colorV = set_color(color)

    ego_transform = sim.get_agents()[0].state.transform

    forward = lgsvl.utils.transform_to_forward(ego_transform)
    right = lgsvl.utils.transform_to_right(ego_transform)

    if distance == 'near':
        offset = 10
        if which_car == 'BoxTruck' or which_car == 'SchoolBus':
            forward = 20 * forward
            right = 5 * right
            speed = 20
        else:
            forward = 10 * forward
            right = 4 * right
            speed = 30
    else:
        offset = 50
        if which_car == 'BoxTruck' or which_car == 'SchoolBus':
            forward = 50 * forward
            right = 5 * right
            speed = 20
        else:
            forward = 50 * forward
            right = 4 * right
            speed = 30

    if which_lane == "left":
        point = ego_transform.position - right - forward
    elif which_lane == "right":
        point = ego_transform.position + right - forward
    elif which_lane == "current":
        point = ego_transform.position - forward
    else:
        point = lgsvl.Vector(ego_transform.position.x + offset * math.cos(0), ego_transform.position.y,
                             ego_transform.position.z + offset * math.sin(0))

    npc_state = lgsvl.AgentState()
    npc_state.transform = sim.map_point_on_lane(point)

    generate = get_no_conflict_position(npc_state.position, which_car)
    if generate:
        npc = sim.add_agent(which_car, lgsvl.AgentType.NPC, npc_state, colorV)
        npc.follow_closest_lane(True, speed)
        npc.change_lane(change_lane == 1)

        control_agents_density(npc)

    agents = sim.get_agents()
    ego = agents[0]
    return calculate_metrics(agents, ego)


@app.route(prefix + 'agents/npc-vehicle/drive-opposite', methods=['POST'])
def add_npc_drive_opposite():

    global NPC_QUEUE
    which_car = cars[random.randint(0, 5)]
    color = colors[random.randint(0, 5)]
    change_lane = int(request.args.get('maintainlane'))
    distance = str(request.args.get('position'))
    colorV = set_color(color)

    ego_transform = sim.get_agents()[0].state.transform
    forward = lgsvl.utils.transform_to_forward(ego_transform)
    right = lgsvl.utils.transform_to_right(ego_transform)

    if distance == 'near':
        offset = 20
    else:
        offset = 50

    if which_car == 'BoxTruck' or which_car == 'SchoolBus':
        forward = offset * forward
        right = 8 * right
        speed = 20
    else:
        forward = offset * forward
        right = 8 * right
        speed = 20

    point = ego_transform.position - right + forward

    npc_state = lgsvl.AgentState()
    npc_state.transform = sim.map_point_on_lane(point)

    generate = get_no_conflict_position(npc_state.position, which_car)
    if generate:
        npc = sim.add_agent(which_car, lgsvl.AgentType.NPC, npc_state, colorV)
        npc.follow_closest_lane(True, speed)
        npc.change_lane(change_lane == 1)

        control_agents_density(npc)

    agents = sim.get_agents()
    ego = agents[0]
    return calculate_metrics(agents, ego)


@app.route('/LGSVL/Control/ControllableObjects/TrafficLight', methods=['POST'])
def control_traffic_light():
    ego_transform = sim.get_agents()[0].state.transform
    forward = lgsvl.utils.transform_to_forward(ego_transform)
    position = ego_transform.position + 50.0 * forward
    signal = sim.get_controllable(position, "signal")

    control_policy = "trigger=100;red=5"
    signal.control(control_policy)

    agents = sim.get_agents()
    ego = agents[0]
    return calculate_metrics(agents, ego)


"""
Status APIs
"""


def interpreter_signal(signal_state):
    code = 0
    if signal_state == 'red':
        code = -1
    elif signal_state == 'yellow':
        code = 0
    elif signal_state == 'green':
        code = 1
    return code

def get_line(point_a, point_b):
    a = (point_a['z'] - point_b['z'])
    b = -(point_a['x'] - point_b['x'])
    c = - point_a['x'] * a - point_a['z'] * b
    
    return {
        'a': a,
        'b': b,
        'c': c
    }

def get_apollo_msg():
    global msg_socket

    msg_socket.send(json.dumps(["start_getting_data"]).encode("utf-8"))
    data = msg_socket.recv(2048)

    data = json.loads(data.decode("utf-8"))

    control_info = data["control_info"]
    local_info = data["local_info"]
    pred_info = data["pred_info"]
    per_info = data["per_info"]
    tlight_info = data["tlight_info"]

    return local_info, per_info, pred_info, control_info, tlight_info

def get_apollo_sub_msg():
    global msg_socket

    msg_socket.send(json.dumps(["start_getting_sub_data"]).encode("utf-8"))
    data = msg_socket.recv(2048)

    data = json.loads(data.decode("utf-8"))

    control_info = data["control_info"]
    local_info = data["local_info"]
    tlight_info = data["tlight_info"]
    chassis_info = data["chassis_info"]
    
    # print("Get Apollo Sub Message")
    
    # print("Traffic light info: ", tlight_info)
    # print("Control info: ", control_info)
    # print("Chassis info: ", chassis_info)

    return local_info, control_info, tlight_info, chassis_info

def get_apollo_ctrl_msg():
    global msg_socket

    msg_socket.send(json.dumps(["get_control_msg"]).encode("utf-8"))
    data = msg_socket.recv(2048)

    data = json.loads(data.decode("utf-8"))

    control_info = data["control_info"]
    chassis_info = data["chassis_info"]
    
    # print("Get Apollo Sub Message")
    
    # print("Traffic light info: ", tlight_info)
    # print("Control info: ", control_info)
    # print("Chassis info: ", chassis_info)

    return control_info, chassis_info

def cal_dis(x_a, y_a, z_a, x_b, y_b, z_b):
    return math.sqrt((x_a - x_b) ** 2 + (y_a - y_b) ** 2 + (z_a - z_b) ** 2)

def cal_dis_2d(x_a, y_a, x_b, y_b):
    return math.sqrt((x_a - x_b) ** 2 + (y_a - y_b) ** 2)

def check_modules_status():
    global EGO
    global DREAMVIEW
    
    # print("Create connection")
    # print("Create connection successfully")
    
    
    modules_status = DREAMVIEW.get_module_status()
    
    stop = False
    
    if modules_status['Localization'] == False:
        print(20*'*', 'LOCALIZATION STOPPED', 20*'*')
        DREAMVIEW.enable_module('Localization')
        print(20*'*', 'LOCALIZATION ENABLED', 20*'*')
        stop = True
        
    if modules_status['Prediction'] == False:
        print(20*'*', 'PREDICTION STOPPED', 20*'*')
        DREAMVIEW.enable_module('Prediction')
        print(20*'*', 'PREDICTION ENABLED', 20*'*')
        stop = True
        
    if modules_status['Transform'] == False:
        print(20*'*', 'TRANSFORM STOPPED', 20*'*')
        DREAMVIEW.enable_module('Transform')
        print(20*'*', 'TRANSFORM ENABLED', 20*'*')
        stop = True
        
    if modules_status['Control'] == False:
        print(20*'*', 'CONTROL STOPPED', 20*'*')
        DREAMVIEW.enable_module('Control')
        print(20*'*', 'CONTROL ENABLED', 20*'*')
        stop = True
        
    if modules_status['Perception'] == False:
        print(20*'*', 'PERCEPTION STOPPED', 20*'*')
        DREAMVIEW.enable_module('Perception')
        print(20*'*', 'PERCEPTION ENABLED', 20*'*')
        stop = True
        
    if modules_status['Routing'] == False:
        print(20*'*', 'ROUTING STOPPED', 20*'*')
        DREAMVIEW.enable_module('Routing')
        print(20*'*', 'ROUTING ENABLED', 20*'*')
        stop = True
        
    if modules_status['Planning'] == False:
        print(20*'*', 'PLANNING STOPPED', 20*'*')
        DREAMVIEW.enable_module('Planning')
        print(20*'*', 'PLANNING ENABLED', 20*'*')
        stop = True
        
    if stop:
        time.sleep(300)

@app.route('/LGSVL/Status/Environment/State', methods=['GET'])
def get_environment_state():
    global MID_POINT
    global lanes_map
    global lane_waypoint
    global next_lane_waypoint
    global current_signals
    global signals_map

    agents = sim.get_agents()

    weather = sim.weather
    position = agents[0].state.position
    rotation = agents[0].state.rotation
    signal = sim.get_controllable(position, "signal")
    speed = agents[0].state.speed
    
    # calculate advanced external features

    num_obs = len(agents) - 1
    num_npc = 0

    min_obs_dist = 100000
    speed_min_obs_dist = 1000
    vol_min_obs_dist = 1000
    dist_to_max_speed_obs = 100000

    max_speed = -100000

    for j in range(1, num_obs + 1):
        state_ = agents[j].state
        if isinstance(agents[j], NpcVehicle):
            num_npc += 1

        dis_to_ego = cal_dis(position.x,
                             position.y,
                             position.z,
                             state_.position.x,
                             state_.position.y,
                             state_.position.z)

        if dis_to_ego < min_obs_dist:
            min_obs_dist = dis_to_ego
            speed_min_obs_dist = state_.speed

            bounding_box_agent = agents[j].bounding_box
            size = bounding_box_agent.size
            vol = size.x * size.y * size.z
            vol_min_obs_dist = vol

        if max_speed < state_.speed:
            max_speed = state_.speed
            dist_to_max_speed_obs = dis_to_ego

    # get apollo info
    local_info, per_info, pred_info, control_info, tlight_info = get_apollo_msg()
    print("Get messages")

    # transform ego's position to world coordinate position
    transform = lgsvl.Transform(
        lgsvl.Vector(position.x, position.y,
                     position.z), lgsvl.Vector(rotation.x, rotation.y, rotation.z)
    )
    gps = sim.map_to_gps(transform)
    dest_x = gps.easting
    dest_y = gps.northing

    # orient = gps.orientation
    
    # print("Orientation: ", gps.orientation)

    # Calculate the differences between localization and simulator

    vector_avut = np.array([
        dest_x,
        dest_y,
    ])

    vector_local = np.array([
        local_info['position']['x'],
        local_info['position']['y'],
    ])

    local_diff = np.linalg.norm(vector_local - vector_avut)
                        
    # Specify the mid point between localization's position and simulator ego's position

    vector_mid = np.array([
        (vector_avut[0] + vector_local[0]) / 2,
        (vector_avut[1] + vector_local[1]) / 2
    ])

    gps2 = sim.map_from_gps(
        None, None, vector_mid[1], vector_mid[0], None, None)

    MID_POINT = np.array([
        gps2.position.x,
        gps2.position.y,
        gps2.position.z
    ])

    # Calculate the angle of lcoalization's position and simulator ego's position

    v_x = vector_local[0] - vector_avut[0]
    v_y = vector_local[1] - vector_avut[1]

    local_angle = math.atan2(v_y, v_x)

    if v_x < 0:
        local_angle += math.pi

    # state_dict = {'x': position.x, 'y': position.y, 'z': position.z,
    #               'rx': rotation.x, 'ry': rotation.y, 'rz': rotation.z,
    #               'rain': weather.rain, 'fog': weather.fog, 'wetness': weather.wetness,
    #               'timeofday': sim.time_of_day, 'signal': interpreter_signal(signal.current_state),
    #               'speed': speed, 'throttle': control_info['throttle'], 'brake': control_info['brake'],
    #               'steering_rate': control_info['steering_rate'], 'steering_target': control_info['steering_target'],
    #               'acceleration': control_info['acceleration'], 'gear': control_info['gear']}

    state_dict = {'x': position.x, 'y': position.y, 'z': position.z,
                  'rx': rotation.x, 'ry': rotation.y, 'rz': rotation.z,
                  'rain': weather.rain, 'fog': weather.fog, 'wetness': weather.wetness,
                  'timeofday': sim.time_of_day, 'signal': interpreter_signal(signal.current_state),
                  'speed': speed, 'local_diff': local_diff, 'local_angle': local_angle,
                  'dis_diff': per_info["dis_diff"], 'theta_diff': per_info["theta_diff"],
                  'vel_diff': per_info["vel_diff"], 'size_diff': per_info["size_diff"],
                  'mlp_eval': pred_info['mlp_eval'], 'cost_eval': pred_info['cost_eval'],
                  'cruise_mlp_eval': pred_info['cruise_mlp_eval'],
                  'junction_mlp_eval': pred_info['junction_mlp_eval'],
                  'cyclist_keep_lane_eval': pred_info['cyclist_keep_lane_eval'],
                  'lane_scanning_eval': pred_info['lane_scanning_eval'],
                  'pedestrian_interaction_eval': pred_info['pedestrian_interaction_eval'],
                  'junction_map_eval': pred_info['junction_map_eval'],
                  'lane_aggregating_eval': pred_info['lane_aggregating_eval'],
                  'semantic_lstm_eval': pred_info['semantic_lstm_eval'],
                  'jointly_prediction_planning_eval': pred_info['jointly_prediction_planning_eval'],
                  'vectornet_eval': pred_info['vectornet_eval'],
                  'unknown': pred_info['unknown'], 'throttle': control_info['throttle'],
                  'brake': control_info['brake'], 'steering_rate': control_info['steering_rate'],
                  'steering_target': control_info['steering_target'], 'acceleration': control_info['acceleration'],
                  'gear': control_info['gear'], "num_obs": num_obs, "num_npc": num_npc,
                  "min_obs_dist": min_obs_dist, "speed_min_obs_dist": speed_min_obs_dist,
                  "vol_min_obs_dist": vol_min_obs_dist, "dist_to_max_speed_obs": dist_to_max_speed_obs
                  }

    # state_dict = {'x': position.x, 'y': position.y, 'z': position.z,
    #               'rx': rotation.x, 'ry': rotation.y, 'rz': rotation.z,
    #               'rain': weather.rain, 'fog': weather.fog, 'wetness': weather.wetness,
    #               'timeofday': sim.time_of_day, 'signal': interpreter_signal(signal.current_state),
    #               'speed': speed, 'dis_diff': per_info["dis_diff"], 'theta_diff': per_info["theta_diff"],
    #               'vel_diff': per_info["vel_diff"], 'size_diff': per_info["size_diff"],
    #               }

    # state_dict = {'x': position.x, 'y': position.y, 'z': position.z,
    #               'rx': rotation.x, 'ry': rotation.y, 'rz': rotation.z,
    #               'rain': weather.rain, 'fog': weather.fog, 'wetness': weather.wetness,
    #               'timeofday': sim.time_of_day, 'signal': interpreter_signal(signal.current_state),
    #               'speed': speed, 'local_diff': local_diff, 'local_angle': local_angle}

    # state_dict = {'x': position.x, 'y': position.y, 'z': position.z,
    #               'rx': rotation.x, 'ry': rotation.y, 'rz': rotation.z,
    #               'rain': weather.rain, 'fog': weather.fog, 'wetness': weather.wetness,
    #               'timeofday': sim.time_of_day, 'signal': interpreter_signal(signal.current_state),
    #               'speed': speed, 'mlp_eval': pred_info['mlp_eval'],
    #               'cost_eval': pred_info['cost_eval'],
    #               'cruise_mlp_eval': pred_info['cruise_mlp_eval'],
    #               'junction_mlp_eval': pred_info['junction_mlp_eval'],
    #               'cyclist_keep_lane_eval': pred_info['cyclist_keep_lane_eval'],
    #               'lane_scanning_eval': pred_info['lane_scanning_eval'],
    #               'pedestrian_interaction_eval': pred_info['pedestrian_interaction_eval'],
    #               'junction_map_eval': pred_info['junction_map_eval'],
    #               'lane_aggregating_eval': pred_info['lane_aggregating_eval'],
    #               'semantic_lstm_eval': pred_info['semantic_lstm_eval'],
    #               'jointly_prediction_planning_eval': pred_info['jointly_prediction_planning_eval'],
    #               'vectornet_eval': pred_info['vectornet_eval'],
    #               'unknown': pred_info['unknown']
    #               }

    # state_dict = {'x': position.x, 'y': position.y, 'z': position.z,
    #               'rx': rotation.x, 'ry': rotation.y, 'rz': rotation.z,
    #               'rain': weather.rain, 'fog': weather.fog, 'wetness': weather.wetness,
    #               'timeofday': sim.time_of_day, 'signal': interpreter_signal(signal.current_state),
    #               'speed': speed}

    # state_dict = {'x': position.x, 'y': position.y, 'z': position.z,
    #               'rx': rotation.x, 'ry': rotation.y, 'rz': rotation.z,
    #               'rain': weather.rain, 'fog': weather.fog, 'wetness': weather.wetness,
    #               'timeofday': sim.time_of_day, 'signal': interpreter_signal(signal.current_state),
    #               'speed': speed, "num_obs": num_obs, "num_npc": num_npc,
    #               "min_obs_dist": min_obs_dist, "speed_min_obs_dist": speed_min_obs_dist,
    #               "vol_min_obs_dist": vol_min_obs_dist, "dist_to_max_speed_obs": dist_to_max_speed_obs}

    # state_dict = {'x': position.x, 'y': position.y, 'z': position.z,
    #               'rx': rotation.x, 'ry': rotation.y, 'rz': rotation.z,
    #               'rain': weather.rain, 'fog': weather.fog, 'wetness': weather.wetness,
    #               'timeofday': sim.time_of_day, 'signal': interpreter_signal(signal.current_state),
    #               'speed': speed, 'per': per_confi, 'pred': pred_confi}

    # state_dict = {'x': position.x, 'y': position.y, 'z': position.z,
    #               'rx': rotation.x, 'ry': rotation.y, 'rz': rotation.z,
    #               'rain': weather.rain, 'fog': weather.fog, 'wetness': weather.wetness,
    #               'timeofday': sim.time_of_day, 'signal': interpreter_signal(signal.current_state),
    #               'speed': speed, 'local': local_diff, 'per': per_confi, 'pred': pred_confi}

    # state_dict = {'x': position.x, 'y': position.y, 'z': position.z,
    #               'rain': weather.rain, 'fog': weather.fog, 'wetness': weather.wetness,
    #               'timeofday': sim.time_of_day, 'signal': interpreter_signal(signal.current_state)}
    return json.dumps(state_dict)

def get_sub_state():
    global lanes_map
    global lane_waypoint
    global next_lane_waypoint
    global current_signals
    global signals_map
    global signals_params
    
    agents = sim.get_agents()
    
    position = agents[0].state.position
    rotation = agents[0].state.rotation
    
    transform = lgsvl.Transform(
        lgsvl.Vector(position.x, position.y,
                     position.z), lgsvl.Vector(rotation.x, rotation.y, rotation.z)
    )
    gps = sim.map_to_gps(transform)
    dest_x = gps.easting
    dest_y = gps.northing

    orient = gps.orientation
    
    orient_radians = math.radians(orient)

    addition = {
        'x': math.cos(orient_radians) * 2.3,
        'y': math.sin(orient_radians) * 2.3
    }
    
    current_signals = {}

    # get apollo info
    local_info, control_info, tlight_info, chassis_info = get_apollo_sub_msg()
    brake_percentage = chassis_info['brake_percentage']
    
    local_info['position']['x'] += addition['x']
    local_info['position']['y'] += addition['y']
    
    # calculate road direction
    
    # current lane direction
    for lane in control_info["lane_arr"]:
        id = control_info["lane_arr"][lane]

        if ((lanes_map[id]['central_curve'][0]['x'] <= local_info['position']['x'] and local_info['position']['x'] <= lanes_map[id]['central_curve'][-1]['x'])
                or (lanes_map[id]['central_curve'][0]['x'] >= local_info['position']['x'] and local_info['position']['x'] >= lanes_map[id]['central_curve'][-1]['x'])):
            if ((lanes_map[id]['central_curve'][0]['y'] <= local_info['position']['y'] and local_info['position']['y'] <= lanes_map[id]['central_curve'][-1]['y'])
                    or (lanes_map[id]['central_curve'][0]['y'] >= local_info['position']['y'] and local_info['position']['y'] >= lanes_map[id]['central_curve'][-1]['y'])):
                cnt = 0
                dis = 100000000
                for i in range(0, len(lanes_map[id]['central_curve']) - 1):
                    cur_dis = cal_dis(local_info['position']['x'], local_info['position']
                                      ['y'], 0, lanes_map[id]['central_curve'][i]['x'], lanes_map[id]['central_curve'][i]['y'], 0)
                    if cur_dis < dis:
                        # print("update lane waypoint")
                        dis = cur_dis
                        
                        gps_waypoint = sim.map_from_gps(None, None, lanes_map[id]['central_curve'][i]['y'], lanes_map[id]['central_curve'][i]['x'],  None, None)
                        
                        lane_waypoint['point_f'] = {
                            'x': gps_waypoint.position.x,
                            'y': gps_waypoint.position.y,
                            'z': gps_waypoint.position.z
                        }
                        
                        gps_waypoint = sim.map_from_gps(None, None, lanes_map[id]['central_curve'][i + 1]['y'], lanes_map[id]['central_curve'][i + 1]['x'],  None, None)

                        lane_waypoint['point_l'] = {
                            'x': gps_waypoint.position.x,
                            'y': gps_waypoint.position.y,
                            'z': gps_waypoint.position.z
                        }
                        
                        lane_waypoint['lane_id'] = id

    # next 3s lane direction
    for lane in control_info["lane_arr"]:
        id = control_info["lane_arr"][lane]

        if ((lanes_map[id]['central_curve'][0]['x'] <= control_info['last_point']['x'] and control_info['last_point']['x'] <= lanes_map[id]['central_curve'][-1]['x'])
                or (lanes_map[id]['central_curve'][0]['x'] >= control_info['last_point']['x'] and control_info['last_point']['x'] >= lanes_map[id]['central_curve'][-1]['x'])):
            if ((lanes_map[id]['central_curve'][0]['y'] <= control_info['last_point']['y'] and control_info['last_point']['y'] <= lanes_map[id]['central_curve'][-1]['y'])
                    or (lanes_map[id]['central_curve'][0]['y'] >= control_info['last_point']['y'] and control_info['last_point']['y'] >= lanes_map[id]['central_curve'][-1]['y'])):
                cnt = 0
                dis = 100000000
                for i in range(0, len(lanes_map[id]['central_curve']) - 1):
                    cur_dis = cal_dis(control_info['last_point']['x'], control_info['last_point']
                                      ['y'], 0, lanes_map[id]['central_curve'][i]['x'], lanes_map[id]['central_curve'][i]['y'], 0)
                    if cur_dis < dis:
                        # print("update lane waypoint")
                        
                        dis = cur_dis
                        
                        gps_waypoint = sim.map_from_gps(None, None, lanes_map[id]['central_curve'][i]['y'], lanes_map[id]['central_curve'][i]['x'],  None, None)
                        
                        next_lane_waypoint['point_f'] = {
                            'x': gps_waypoint.position.x,
                            'y': gps_waypoint.position.y,
                            'z': gps_waypoint.position.z
                        }
                        
                        gps_waypoint = sim.map_from_gps(None, None, lanes_map[id]['central_curve'][i + 1]['y'], lanes_map[id]['central_curve'][i + 1]['x'],  None, None)

                        next_lane_waypoint['point_l'] = {
                            'x': gps_waypoint.position.x,
                            'y': gps_waypoint.position.y,
                            'z': gps_waypoint.position.z
                        }
                        
                        next_lane_waypoint['lane_id'] = id
    
    cnt = 0
     
    for tlight in tlight_info:
        id = tlight_info[tlight]['id']
        signal_info = signals_map[id]
        
        if not (id in signals_params):
            
            gps_point = sim.map_from_gps(None, None, signal_info['stop_line'][0]['first_point']['y'], signal_info['stop_line'][0]['first_point']['x'],  None, None)
                        
            first_point = {
                'x': gps_point.position.x,
                'y': gps_point.position.y,
                'z': gps_point.position.z
            }
            
            gps_point = sim.map_from_gps(None, None, signal_info['stop_line'][0]['last_point']['y'], signal_info['stop_line'][0]['last_point']['x'],  None, None)
                        
            last_point = {
                'x': gps_point.position.x,
                'y': gps_point.position.y,
                'z': gps_point.position.z
            }
            
            stop_line_params = get_line(first_point, last_point)
            
            signals_params[str(id)] = stop_line_params
        
        tf_obj = {
            'id': tlight_info[tlight]['id'],
            'color': tlight_info[tlight]['color'],
            'stop_line': signals_params[str(id)]
        }
        current_signals[str(cnt)] = tf_obj
        cnt += 1
        
    # print("Current Signals: ", current_signals)
        
    return control_info['acceleration'], brake_percentage, orient


@app.route('/LGSVL/Status/Realistic', methods=['GET'])
def get_realistic():

    return json.dumps(REALISTIC)


@app.route('/LGSVL/Status/Environment/Weather', methods=['GET'])
def get_weather():
    weather = sim.weather
    weather_dict = {'rain': weather.rain,
                    'fog': weather.fog, 'wetness': weather.wetness}

    return json.dumps(weather_dict)


@app.route('/LGSVL/Status/Environment/Weather/Rain', methods=['GET'])
def get_rain():
    return str(sim.weather.rain)


@app.route('/LGSVL/Status/Environment/TimeOfDay', methods=['GET'])
def get_timeofday():
    return str(sim.time_of_day)


@app.route('/LGSVL/Status/CollisionInfo', methods=['GET'])
def get_loc():
    """
    three types of collision: obstacle, NPC vehicle, pedestrian, None(no collision)
    :return:
    """
    global collision_object
    collision_info = str(collision_object)
    collision_object = None
    collision_type = str(None)
    if collision_info == 'OBSTACLE':
        collision_type = "obstacle"
    if collision_info in npc_vehicle:
        collision_type = "npc_vehicle"
    if collision_info in pedestrian:
        collision_type = "pedestrian"

    print("Collision Info: ", collision_type)

    return collision_type


@app.route('/LGSVL/Status/EGOVehicle/Speed', methods=['GET'])
def get_speed():
    speed = "{:.2f}".format(sim.get_agents()[0].state.speed)
    return speed


@app.route('/LGSVL/Status/EGOVehicle/Position', methods=['GET'])
def get_position():
    position = sim.get_agents()[0].state.position
    pos_dict = {'x': position.x, 'y': position.y, 'z': position.z}
    return json.dumps(pos_dict)


@app.route('/LGSVL/Status/EGOVehicle/Position/X', methods=['GET'])
def get_position_x():
    position = sim.get_agents()[0].state.position
    return "{:.2f}".format(position.x)


@app.route('/LGSVL/Status/EGOVehicle/Position/Y', methods=['GET'])
def get_position_y():
    position = sim.get_agents()[0].state.position
    return "{:.2f}".format(position.y)


@app.route('/LGSVL/Status/EGOVehicle/Position/Z', methods=['GET'])
def get_position_z():
    position = sim.get_agents()[0].state.position
    return "{:.2f}".format(position.z)


"""
RESTful APIs for getting GPS data
"""


@app.route('/LGSVL/Status/GPSData', methods=['GET'])
def get_gps_data():
    gps_json = {}  # dict file, also can be defined by gps_json = dict()
    gps_data = sensors[1].data

    # String format: "{:.2f}".format(gps_data.latitude), "{:.2f}".format(gps_data.longitude)
    gps_json.update({'Altitude': round(gps_data.altitude, 2), 'Latitude': round(gps_data.latitude, 2),
                     'Longitude': round(gps_data.longitude, 2), 'Northing': round(gps_data.northing, 2),
                     'Easting': round(gps_data.easting, 2)})
    return json.dumps(gps_json)


@app.route('/LGSVL/Status/GPS/Latitude', methods=['GET'])
def get_latitude():
    gps_data = sensors[1].data
    latitude = "{:.2f}".format(gps_data.latitude)
    return latitude


@app.route('/LGSVL/Status/GPS/Longitude', methods=['GET'])
def get_longitude():
    gps_data = sensors[1].data
    longitude = "{:.2f}".format(gps_data.longitude)
    return longitude


@app.route('/LGSVL/Status/GPS/Altitude', methods=['GET'])
def get_altitude():
    gps_data = sensors[1].data
    altitude = "{:.2f}".format(gps_data.altitude)
    return altitude


@app.route('/LGSVL/Status/GPS/Northing', methods=['GET'])
def get_northing():
    gps_data = sensors[1].data
    northing = "{:.2f}".format(gps_data.northing)
    return northing


@app.route('/LGSVL/Status/GPS/Easting', methods=['GET'])
def get_easting():
    gps_data = sensors[1].data
    easting = "{:.2f}".format(gps_data.easting)
    return easting


@app.route('/LGSVL/Status/CollisionProbability', methods=['GET'])
def get_c_probability():
    global probability
    c_probability = probability
    probability = 0
    return str(c_probability)

@app.route('/LGSVL/Status/DiversityLevel', methods=['GET'])
def get_diversity_level():
    global diversity_level
    d_level = diversity_level
    diversity_level = 0
    return str(d_level)


@app.route('/LGSVL/Status/ViolationRateReward', methods=['GET'])
def get_violation_rate():
    global vioRate_reward
    c_vioRate = vioRate_reward
    vioRate_reward = 0
    return str(c_vioRate)


@app.route('/LGSVL/Status/HardBrake', methods=['GET'])
def get_hard_brake():
    global speed_list
    acceleration_threshold = 8
    hard_brake = False
    speed = speed_list[0]
    for i in range(1, len(speed_list), 2):
        temp = speed_list[i]
        acceleration = abs(temp - speed) / 1
        speed = temp
        if acceleration >= acceleration_threshold:
            hard_brake = True
            break
    return json.dumps(hard_brake)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8933, debug=False)