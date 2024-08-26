import socket
import time

import cv2
from flask import Flask, request
import os
from datetime import timedelta
import json
import lgsvl
import numpy as np
from collision_utils_origin import pedestrian, npc_vehicle, calculate_measures
from clustering import cluster
import math
import threading
from lgsvl.agent import NpcVehicle
import random
import queue
import pickle

import torch
from lgsvl.dreamview import CoordType
import requests
from numba import jit

########################################
observation_time = 6  # OTP [4, 6, 8, 10]

app = Flask(__name__)

app.config['SECRET_KEY'] = os.urandom(24)
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=7)

# create simulator

sim = lgsvl.Simulator(os.environ.get(
    "SIMUSaveStateLATOR_HOST", "localhost"), 8977)

DREAMVIEW = None
collision_object = None
probability = 0
ETTC = 100
DTO = 100000
JERK = 100
time_step_collision_object = None
sensors = None
DESTINATION = None
EGO = None
CONSTRAINS = True
SAVE_SCENARIO = False
REALISTIC = False
collision_tag = False
EPISODE = 0
CONTROL = False
NPC_QUEUE = queue.Queue(maxsize=10)
collision_speed = 0  # 0 indicates there is no collision occurred.
collision_uid = "No collision"
isCollisionAhead = False
prev_acc = 0
time_offset = -5 # add time offset
pedes_prev_pos = {}

current_lane = {
    'left_boundary': {
        'a': 0,
        'b': 0,
        'c': 0
    },
    'right_boundary': {
        'a': 0,
        'b': 0,
        'c': 0
    }
}

speed_list = []

cars = ['Jeep', 'BoxTruck', 'Sedan', 'SchoolBus', 'SUV', 'Hatchback']
colors = ['pink', 'yellow', 'red', 'white', 'black', 'skyblue']

u = 0.6
prefix = '/deepqtest/lgsvl-api/'

# setup connect to apollo

APOLLO_HOST = '112.137.129.158'  # or 'localhost'
PORT = 8966
DREAMVIEW_PORT = 9988
BRIDGE_PORT = 9090

msg_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_address = (APOLLO_HOST, PORT)

msg_socket.connect(server_address)

map = 'sanfrancisco' # map: tartu, sanfrancisco, borregasave

lanes_map_file = "./map/{}_lanes.pkl".format(map)
lanes_map = None

with open(lanes_map_file, "rb") as file:
    lanes_map = pickle.load(file)

file.close()

# on collision callback function
def on_collision(agent1, agent2, contact):
    name1 = agent1.__dict__.get('name')
    name2 = agent2.__dict__.get('name') if agent2 is not None else "OBSTACLE"
    uid = agent2.__dict__.get('uid') if agent2 is not None else "OBSTACLE"
    print("{} collided with {} at {}".format(name1, name2, contact))
    if agent2 is not None:
        print("AV speed {}, NPC speed {} ".format(agent1.state.speed, agent2.state.speed))
    global collision_object
    global collision_tag
    global collision_speed
    global collision_uid
    global isCollisionAhead
    collision_uid = uid
    collision_object = name2
    collision_tag = True
    
    # raise evaluator.TestException("Ego collided with {}".format(agent2))
    try:
        collision_speed = agent1.state.speed
        
        if name2 != 'OBSTACLE' and not isinstance(agent2, NpcVehicle):
            ego_pos = agent1.state.position
            ego_rot = agent1.state.rotation
            
            yaw_rad = np.deg2rad(ego_rot.y)
            
            ego_dir = ego_rot
    
            ego_dir.x = np.sin(yaw_rad)
            ego_dir.z = np.cos(yaw_rad)
            
            # ego_bbox = agent1.bounding_box
            length = 4.7
            
            # print("Ego length: ", length)
            # print("Ego rot: ", ego_rot)
            # print("Ego pos: ", ego_pos)
            
            vel_ = math.sqrt(ego_dir.x ** 2 + ego_dir.z ** 2)
            ahd_pt = ego_pos + (length / 2) / vel_ * ego_dir 
            
            a = ego_dir.x 
            b = ego_dir.z
            c = - a * ahd_pt.x - b * ahd_pt.z
            
            # print("line: ", a, b, c)
            
            pedes_pos = agent2.state.position
            # print("pedes pos: ", pedes_pos)
            
            val_ego = a * ego_pos.x + b * ego_pos.z +c
            val_pedes = a * pedes_pos.x + b * pedes_pos.z + c
            
            # print("val_ego: ", val_ego)
            # print("val_pedes: ", val_pedes)
            
            if val_ego * val_pedes <= 0:
                isCollisionAhead = True            
            
    except KeyError:
        collision_speed = -1
        print('KeyError')
# check whether there is collision or not

def get_boundary_value(boundary, position):
    return boundary['a'] * position.x + boundary['b'] * position.z + boundary['c']

def get_no_conflict_position(position, car):
    if car == 'BoxTruck' or car == 'SchoolBus':
        sd = 10
    else:
        sd = 8
    generate = True
    if CONSTRAINS:
        agents = sim.get_agents()
        for agent in agents:

            left_val_1 = get_boundary_value(current_lane['left_boundary'], position)
            right_val_1 = get_boundary_value(current_lane['right_boundary'], position)

            left_val_2 = get_boundary_value(current_lane['left_boundary'], agent.transform.position)
            right_val_2 = get_boundary_value(current_lane['right_boundary'], agent.transform.position)

            isConsidered = ((left_val_1 * left_val_2) > 0 and (right_val_1 * right_val_2) > 0)

            if isConsidered and math.sqrt(pow(position.x - agent.transform.position.x, 2) +
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

# calculate measures thread, use in multi-thread
def calculate_measures_thread(npc_state, ego_state, isNpcVehicle, TTC_list, 
                              distance_list, probability_list, collision_tag_=False):

    TTC, distance, probability2 = calculate_measures(npc_state, ego_state, isNpcVehicle)
    
    TTC_list.append(round(TTC, 6))

    distance_list.append(round(distance, 6))
    
    if collision_tag_:
        probability2 = 1
        
    probability_list.append(round(probability2, 6))   
        
# calculate metrics
def calculate_metrics(agents, ego, uid = None):
    global probability
    global ETTC
    global DTO
    global JERK
    global collision_tag

    global SAVE_SCENARIO

    global collision_object
    global collision_speed
    global collision_uid
    global isCollisionAhead
    global prev_acc
    global pedes_prev_pos

    collision_object = None
    collision_speed = 0  # 0 indicates there is no collision occurred.
    collision_speed_ = 0
    collision_type = "None"
    collision_uid = "No collision"
    isCollisionAhead = False
    spd_bf_col = 0
    isFirstCollision = True
    pedes_mov_fw_to = False

    print("Calculating metrics ....")

    ETTC_list = []
    distance_list = []
    probability_list = []
    JERK_list = []
    
    i = 0
    time_step = 0.5
    sliding_step = 0.25
    
    sudden_appearance = False
    overlapping = False
    position_list = {}

    while i < observation_time / time_step:
        
        # check apollo's modules status from dreamview
        # print("Checking modules status ....")
        check_modules_status()
        
        # run simulator
        # print("Running simulator ....")
        for k in range(0, int(time_step / sliding_step)):
            
            spd_bf_col = ego.state.speed
            
            sim.run(time_limit=sliding_step)  # , time_scale=2
            
            collision_info_ = str(collision_object)
            collision_type_ = 'None'
            
            if collision_info_ == 'OBSTACLE':
                collision_type_ = "obstacle"
            if collision_info_ in npc_vehicle:
                collision_type_ = "npc_vehicle"
            if collision_info_ in pedestrian:
                collision_type_ = "pedestrian"
            
            # print("Collision Info: ", collision_type_)
            
            if collision_tag and isFirstCollision:
                print("Ego speed before collision: ", spd_bf_col)
                isFirstCollision = False
                if spd_bf_col < 0.5 and collision_type_ == 'pedestrian':
                    pedes_mov_fw_to = True
            
            if uid:
                if collision_tag:
                    print("Collision at: ", (i * 2 + k) * 0.25, "-", (i * 2 + k + 1) * 0.25)
                    if uid == collision_uid and (i * 2 + k + 1) * 0.25 <= 0.75:
                        sudden_appearance = True
        
        # ego_acc = get_ego_acceleration()
        ego_acc = 5
        # print("Ego Acceleration: ", ego_acc)
        
        JERK_list.append(abs(ego_acc - prev_acc) / 0.5)
        # print("Ego JERK: ", abs(ego_acc - prev_acc) / 0.5)
        prev_acc = ego_acc
        
        ego_state = ego.state
        
        npc_state = []
        isNpcVehicle = []
        
        pos = {}
        
        pos[ego.uid] = {
            'x': ego_state.position.x,
            'y': ego_state.position.y,
            'z': ego_state.position.z,
            'dis_to_ego': 0
        }
        
        for j in range(1, len(agents)):
            state_ = agents[j].state
            isNpc = (isinstance(agents[j], NpcVehicle))
            isNpcVehicle.append(isNpc)
            
            if not isNpc:
                pedes_uid = agents[j].uid
                # print("Pedes_uid: ", pedes_uid)
                
                if pedes_uid in pedes_prev_pos:
                    state_.velocity.x = (state_.transform.position.x - pedes_prev_pos[pedes_uid][0]) / 0.5
                    state_.velocity.y = (state_.transform.position.y - pedes_prev_pos[pedes_uid][1]) / 0.5
                    state_.velocity.z = (state_.transform.position.z - pedes_prev_pos[pedes_uid][2]) / 0.5
                    # state_.speed = math.sqrt(state_.velocity.x ** 2 + state_.velocity.y ** 2 + state_.velocity.z ** 2)
                
                # print("Updated velocity: ", state_.velocity)
                
                pedes_prev_pos[pedes_uid] = np.array([state_.transform.position.x, state_.transform.position.y, state_.transform.position.z])
                # print("Pedes position: ", pedes_prev_pos[pedes_uid])
                
            pos[agents[j].uid] = {
                'x': state_.position.x,
                'y': state_.position.y,
                'z': state_.position.z,
                'dis_to_ego': math.sqrt((state_.position.x - ego_state.position.x) ** 2 + (state_.position.y - ego_state.position.y) ** 2 + (state_.position.z - ego_state.position.z) ** 2)
            }
            
            if not overlapping:
                if abs(ego_state.position.y - state_.position.y) > 0.4:
                    overlapping = True
                
            npc_state.append(state_)
            
        position_list[str(i)] = pos

        thread = threading.Thread(
            target=calculate_measures_thread,
            args=(npc_state, ego_state, isNpcVehicle, ETTC_list, 
                  distance_list, probability_list, collision_tag,)
        )

        thread.start()

        if collision_tag:
            collision_tag = False

        i += 1
        
        if i == int(observation_time / time_step):
            time.sleep(0.5)

    # if SAVE_SCENARIO:
    collision_type, collision_speed_, collision_uid_, isCollisionAhead_ = get_collision_info()
    if collision_speed_ == -1:
        collision_speed_ = speed
        
    print("ETTC List: ", ETTC_list)
    print("DTO List: ", distance_list)
    print("JERK List: ", JERK_list)
    print("PROC List: ", probability_list)
    
    ETTC = round(min(ETTC_list), 6)
    DTO = round(min(distance_list), 6)
    JERK = round(max(JERK_list), 6)
    probability = round(max(probability_list), 6)
    
    return {'ETTC': ETTC_list, 'distance': distance_list, 'JERK': JERK_list, 'collision_uid': collision_uid_, 'probability': probability_list, "sudden_appearance": sudden_appearance, 
            "overlapping": overlapping, 'position_list': position_list, 'generated_uid': uid, 'isCollisionAhead': isCollisionAhead_, 'pedes_mov_fw_to': pedes_mov_fw_to} 


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
    global isCollisionAhead
    global JERK

    collision_info = str(collision_object)
    collision_speed_ = collision_speed

    # collision_object = None
    
    
    isCollisionAhead_ = isCollisionAhead
    collision_speed = 0
    JERK = 0
    collision_type = 'None'
    if collision_info == 'OBSTACLE':
        collision_type = "obstacle"
    if collision_info in npc_vehicle:
        collision_type = "npc_vehicle"
    if collision_info in pedestrian:
        collision_type = "pedestrian"
    return collision_type, collision_speed_, collision_uid, isCollisionAhead_


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
    global prev_lane_id
    global DREAMVIEW
    global prev_acc
    global time_offset
    global pedes_prev_pos
    
    prev_acc = 0
    prev_lane_id = ""
    pedes_prev_pos = {}

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
    sim.set_time_of_day((10 + time_offset) % 24, fixed=True)
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
    global time_offset
    
    time = request.args.get('time_of_day')
    day_time = (10 + time_offset) % 24  # initial time: 10
    if time == 'Morning':
        day_time = (10 + time_offset) % 24
    elif time == 'Noon':
        day_time = (14 + time_offset) % 24 
    elif time == 'Evening':
        day_time = (20 + time_offset) % 24
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
    lane = request.args.get('lane')
    distance = str(request.args.get('position'))
    colorV = set_color(color)

    ego_transform = sim.get_agents()[0].state.transform
    sx = ego_transform.position.x
    sy = ego_transform.position.y
    sz = ego_transform.position.z

    right = lgsvl.utils.transform_to_right(ego_transform)
    forward_ = lgsvl.utils.transform_to_forward(ego_transform)

    # angle = 0
    # dist = 20 if distance == 'near' else 40

    # point = lgsvl.Vector(sx + dist * math.cos(angle),
    #                      sy, sz + dist * math.sin(angle))

    point = None

    if lane == 'right': 
        if distance == 'near':
            if which_car == 'BoxTruck' or which_car == 'SchoolBus':
                point = ego_transform.position + 4 * right + 17 * forward_
            else:
                point = ego_transform.position + 4 * right + 12 * forward_
        else:
            if which_car == 'BoxTruck' or which_car == 'SchoolBus':
                point = ego_transform.position + 4 * right + 30 * forward_
            else:
                point = ego_transform.position + 4 * right + 25 * forward_
    else:
        if distance == 'near':
            if which_car == 'BoxTruck' or which_car == 'SchoolBus':
                point = ego_transform.position - 4 * right + 17 * forward_
            else:
                point = ego_transform.position - 4 * right + 12 * forward_
        else:
            if which_car == 'BoxTruck' or which_car == 'SchoolBus':
                point = ego_transform.position - 4 * right + 30 * forward_
            else:
                point = ego_transform.position - 4 * right + 25 * forward_


    state = lgsvl.AgentState()
    state.transform = sim.map_point_on_lane(point)

    npc = None

    generate = get_no_conflict_position(state.position, which_car)
    if not generate:    
        if distance == 'near':
            # print("Collided, regenerate")
            point -= forward_ * 10
            state.transform = sim.map_point_on_lane(point)
            generate = get_no_conflict_position(state.position, which_car)
            # print("NPC Point:", point)

            # print("NPC Position:", npc_state.transform.position)
        else:
            # print("Collided, regenerate")
            point += forward_ * 20
            state.transform = sim.map_point_on_lane(point)
            generate = get_no_conflict_position(state.position, which_car)
            # print("NPC Point:", point)

            # print("NPC Position:", npc_state.transform.position)
    if generate:
        npc = sim.add_agent(which_car, lgsvl.AgentType.NPC, state, colorV)
        npc.follow_closest_lane(True, 20)
        if change_lane:
            if lane == 'left':
                npc.change_lane(False)
            elif lane == 'right':
                npc.change_lane(True)

        control_agents_density(npc)
    agents = sim.get_agents()
    ego = agents[0]
    
    uid = None
    
    if npc:
        uid = npc.uid
    
    return calculate_metrics(agents, ego, uid)


@app.route(prefix + 'agents/pedestrian/cross-road', methods=['POST'])
def add_pedestrian_cross_road():

    global NPC_QUEUE
    global pedes_prev_pos
    
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
    
    p = None
    
    if generate:
        name = pedestrian[random.randint(0, 8)]
        p = sim.add_agent(name, lgsvl.AgentType.PEDESTRIAN, npc_state)
        
        pedes_uid = p.uid
        # print("Pedes_uid: ", pedes_uid)
        pedes_prev_pos[pedes_uid] = np.array([npc_state.transform.position.x, npc_state.transform.position.y, npc_state.transform.position.z])
        # print("Pedes position: ", pedes_prev_pos[pedes_uid])
        
        p.follow(wp, loop=False)

        control_agents_density(p)
        
    agents = sim.get_agents()
    ego = agents[0]
    
    uid = None
    
    if p:
        uid = p.uid
    
    return calculate_metrics(agents, ego, uid)


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
    forward_ = forward
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
    generate = False
    npc_state.transform = sim.map_point_on_lane(point)
    # npc_state.transform.position = point

    # print("NPC Point:", point)

    # print("NPC Position:", npc_state.transform.position)

    generate = get_no_conflict_position(npc_state.position, which_car)

    if not generate:
        if distance == 'near':
            # print("Collided, regenerate")
            point += forward_ * 10
            npc_state.transform = sim.map_point_on_lane(point)
            generate = get_no_conflict_position(npc_state.position, which_car)
            # print("NPC Point:", point)

            # print("NPC Position:", npc_state.transform.position)
        else:
            # print("Collided, regenerate")
            point -= forward_ * 20
            npc_state.transform = sim.map_point_on_lane(point)
            generate = get_no_conflict_position(npc_state.position, which_car)
            # print("NPC Point:", point)

            # print("NPC Position:", npc_state.transform.position)

    npc = None

    if generate:
        # print("NPC is generated")
        npc = sim.add_agent(which_car, lgsvl.AgentType.NPC, npc_state, colorV)
        npc.follow_closest_lane(True, speed)
        if change_lane:
            if which_lane == 'left':
                npc.change_lane(False)
            elif which_lane == 'right':
                npc.change_lane(True)
            else:
                npc.change_lane(random.choice([True, False]))

        control_agents_density(npc)

    agents = sim.get_agents()
    ego = agents[0]
    
    uid = None
    
    if npc:
        uid = npc.uid
    
    return calculate_metrics(agents, ego, uid)


@app.route(prefix + 'agents/npc-vehicle/overtake', methods=['POST'])
def add_npc_overtake():
    global NPC_QUEUE
    global cars
    global colors
    which_lane = request.args.get('lane')
    which_car = cars[random.randint(0, 5)]
    color = colors[random.randint(0, 5)]
    change_lane = int(request.args.get('maintainlane'))
    # distance = str(request.args.get('position'))
    colorV = set_color(color)

    ego_transform = sim.get_agents()[0].state.transform

    forward = lgsvl.utils.transform_to_forward(ego_transform)
    forward_ = forward
    right = lgsvl.utils.transform_to_right(ego_transform)

    if which_car == 'BoxTruck' or which_car == 'SchoolBus':
        offset = 10
        forward = 10 * forward
        right = 4 * right
        speed = 20
    else:
        offset = 10
        forward = 5 * forward
        right = 4 * right
        speed = 20

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

    if not generate:    
        
        # print("Collided, regenerate")
        point -= forward_ * 10
        npc_state.transform = sim.map_point_on_lane(point)
        generate = get_no_conflict_position(npc_state.position, which_car)
        # print("NPC Point:", point)

        # print("NPC Position:", npc_state.transform.position)
        
    npc = None

    if generate:
        npc = sim.add_agent(which_car, lgsvl.AgentType.NPC, npc_state, colorV)
        npc.follow_closest_lane(True, speed)
        if change_lane:
            if which_lane == 'left':
                npc.change_lane(False)
            elif which_lane == 'right':
                npc.change_lane(True)
            else:
                npc.change_lane(random.choice([True, False]))

        control_agents_density(npc)

    agents = sim.get_agents()
    ego = agents[0]
    
    uid = None
    
    if npc:
        uid = npc.uid
    
    return calculate_metrics(agents, ego, uid)


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
    forward_ = forward
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
    if not generate:    
        if distance == 'near':
            # print("/Collided, regenerate")
            point += forward_ * 10
            npc_state.transform = sim.map_point_on_lane(point)
            generate = get_no_conflict_position(npc_state.position, which_car)
            # print("NPC Point:", point)

            # print("NPC Position:", npc_state.transform.position)
        else:
            # print("Collided, regenerate")
            point -= forward_ * 20
            npc_state.transform = sim.map_point_on_lane(point)
            generate = get_no_conflict_position(npc_state.position, which_car)
            # print("NPC Point:", point)

            # print("NPC Position:", npc_state.transform.position)
    npc = None
            
    if generate:
        npc = sim.add_agent(which_car, lgsvl.AgentType.NPC, npc_state, colorV)
        npc.follow_closest_lane(True, speed)
        npc.change_lane(change_lane == 1)

        control_agents_density(npc)

    agents = sim.get_agents()
    ego = agents[0]
    
    uid = None
    if npc:
        uid = npc.uid
    
    return calculate_metrics(agents, ego, uid)


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

# def get_line(point_a, point_b):
#     a = (point_a['z'] - point_b['z'])
#     b = -(point_a['x'] - point_b['x'])
#     c = - point_a['x'] * a - point_a['z'] * b
    
#     return {
#         'a': a,
#         'b': b,
#         'c': c
#     }

def get_ego_acceleration():
    global msg_socket

    msg_socket.send(json.dumps(["get_acc"]).encode("utf-8"))
    data = msg_socket.recv(1024)

    data = json.loads(data.decode("utf-8"))

    acceleration = data["acceleration"]

    return acceleration

def get_apollo_msg():
    global msg_socket

    msg_socket.send(json.dumps(["start_getting_data"]).encode("utf-8"))
    data = msg_socket.recv(2048)

    data = json.loads(data.decode("utf-8"))

    control_info = data["control_info"]
    local_info = data["local_info"]
    per_info = data["per_info"]

    return local_info, per_info, control_info

def check_modules_status():
    global EGO
    global DREAMVIEW
    
    # print("Create connection")
    # print("Create connection successfully")
    
    
    modules_status = DREAMVIEW.get_module_status()
    
    stop = False
    
    stop_time = 300
    
    if modules_status['Localization'] == False:
        print(20*'*', 'LOCALIZATION STOPPED', 20*'*')
        DREAMVIEW.enable_module('Localization')
        print(20*'*', 'LOCALIZATION ENABLED', 20*'*')
        stop = True
        # stop_time = max(120, stop_time)
        
    if modules_status['Prediction'] == False:
        print(20*'*', 'PREDICTION STOPPED', 20*'*')
        DREAMVIEW.enable_module('Prediction')
        print(20*'*', 'PREDICTION ENABLED', 20*'*')
        stop = True
        # stop_time = max(120, stop_time)
        
    if modules_status['Transform'] == False:
        print(20*'*', 'TRANSFORM STOPPED', 20*'*')
        DREAMVIEW.enable_module('Transform')
        print(20*'*', 'TRANSFORM ENABLED', 20*'*')
        stop = True
        # stop_time = max(120, stop_time)
        
    if modules_status['Control'] == False:
        print(20*'*', 'CONTROL STOPPED', 20*'*')
        DREAMVIEW.enable_module('Control')
        print(20*'*', 'CONTROL ENABLED', 20*'*')
        stop = True
    #    stop_time = max(120, stop_time)
        
    if modules_status['Perception'] == False:
        print(20*'*', 'PERCEPTION STOPPED', 20*'*')
        DREAMVIEW.enable_module('Perception')
        print(20*'*', 'PERCEPTION ENABLED', 20*'*')
        stop = True
        # stop_time = max(300, stop_time)
        
    if modules_status['Routing'] == False:
        print(20*'*', 'ROUTING STOPPED', 20*'*')
        DREAMVIEW.enable_module('Routing')
        print(20*'*', 'ROUTING ENABLED', 20*'*')
        stop = True
        # stop_time = max(120, stop_time)
        
    if modules_status['Planning'] == False:
        print(20*'*', 'PLANNING STOPPED', 20*'*')
        DREAMVIEW.enable_module('Planning')
        print(20*'*', 'PLANNING ENABLED', 20*'*')
        stop = True
        # stop_time = max(120, stop_time)
        
    if stop:
        time.sleep(stop_time)

def cal_dis(x_a, y_a, z_a, x_b, y_b, z_b):
    return math.sqrt((x_a - x_b) ** 2 + (y_a - y_b) ** 2 + (z_a - z_b) ** 2)


@app.route('/LGSVL/Status/Environment/State', methods=['GET'])
def get_environment_state():

    global current_lane

    agents = sim.get_agents()

    weather = sim.weather
    position = agents[0].state.position
    rotation = agents[0].state.rotation
    signal = sim.get_controllable(position, "signal")
    speed = agents[0].state.speed
    
    # calculate advanced external features

    num_obs = len(agents) - 1

    min_obs_dist = 100
    speed_min_obs_dist = 100

    for j in range(1, num_obs + 1):
        state_ = agents[j].state

        dis_to_ego = cal_dis(position.x,
                             position.y,
                             position.z,
                             state_.position.x,
                             state_.position.y,
                             state_.position.z)

        if dis_to_ego < min_obs_dist:
            min_obs_dist = dis_to_ego
            speed_min_obs_dist = state_.speed

    # get apollo info
    local_info, per_info, control_info = get_apollo_msg()
    print("Get messages")

    # extract lane info 

    for lane in control_info["lane_arr"]:
        id = control_info["lane_arr"][lane]

        if ((lanes_map[id]['central_curve'][0]['x'] <= local_info['position']['x'] and local_info['position']['x'] <= lanes_map[id]['central_curve'][-1]['x'])
                or (lanes_map[id]['central_curve'][0]['x'] >= local_info['position']['x'] and local_info['position']['x'] >= lanes_map[id]['central_curve'][-1]['x'])):
            if ((lanes_map[id]['central_curve'][0]['y'] <= local_info['position']['y'] and local_info['position']['y'] <= lanes_map[id]['central_curve'][-1]['y'])
                    or (lanes_map[id]['central_curve'][0]['y'] >= local_info['position']['y'] and local_info['position']['y'] >= lanes_map[id]['central_curve'][-1]['y'])):
                
                left_bound_1 = sim.map_from_gps(None, None, lanes_map[id]['left_boundary'][0]['y'], lanes_map[id]['left_boundary'][0]['x'],  None, None)
                left_bound_2 = sim.map_from_gps(None, None, lanes_map[id]['left_boundary'][-1]['y'], lanes_map[id]['left_boundary'][-1]['x'],  None, None)
                
                left_a = left_bound_2.position.z - left_bound_1.position.z
                left_b = left_bound_1.position.x - left_bound_2.position.x
                left_c = - left_a * left_bound_1.position.x - left_b * left_bound_1.position.z

                right_bound_1 = sim.map_from_gps(None, None, lanes_map[id]['right_boundary'][0]['y'], lanes_map[id]['right_boundary'][0]['x'],  None, None)
                right_bound_2 = sim.map_from_gps(None, None, lanes_map[id]['right_boundary'][-1]['y'], lanes_map[id]['right_boundary'][-1]['x'],  None, None)
                
                right_a = right_bound_2.position.z - right_bound_1.position.z
                right_b = right_bound_1.position.x - right_bound_2.position.x
                right_c = - right_a * right_bound_1.position.x - right_b * right_bound_1.position.z

                current_lane['left_boundary']['a'] = left_a
                current_lane['left_boundary']['b'] = left_b
                current_lane['left_boundary']['c'] = left_c

                current_lane['right_boundary']['a'] = right_a
                current_lane['right_boundary']['b'] = right_b
                current_lane['right_boundary']['c'] = right_c

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

    # Calculate the angle of lcoalization's position and simulator ego's position

    v_x = vector_local[0] - vector_avut[0]
    v_y = vector_local[1] - vector_avut[1]

    local_angle = math.atan2(v_y, v_x)

    if v_x < 0:
        local_angle += math.pi

    weather_state = weather.rain * 10 ** 2 + weather.fog * 10 + weather.wetness

    state_dict = {'x': position.x, 'y': position.y, 'z': position.z,
                  'rx': rotation.x, 'ry': rotation.y, 'rz': rotation.z,
                  'weather': weather_state, 'timeofday': sim.time_of_day, 
                  'signal': interpreter_signal(signal.current_state),
                  'speed': speed, 'local_diff': local_diff, 'local_angle': local_angle,
                  'dis_diff': per_info["dis_diff"], 'theta_diff': per_info["theta_diff"],
                  'vel_diff': per_info["vel_diff"], 'size_diff': per_info["size_diff"],
                  'throttle': control_info['throttle'], 'brake': control_info['brake'], 
                  'steering_rate': control_info['steering_rate'], 'steering_target': control_info['steering_target'], 
                  'acceleration': control_info['acceleration'], "num_obs": num_obs, 
                  "min_obs_dist": min_obs_dist, "speed_min_obs_dist": speed_min_obs_dist,
                  }

    return json.dumps(state_dict)

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
    print("Collisison Object")
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

@app.route('/LGSVL/Status/CollisionObject', methods=['GET'])
def get_collision_object():
    global collision_object
    return str(collision_object)


@app.route('/LGSVL/Status/CollisionUid', methods=['GET'])
def get_collision_uid():
    global collision_uid

    col_uid = collision_uid
    collision_uid = "No Collision"

    return str(col_uid)

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

@app.route('/LGSVL/Status/DistanceToObstacles', methods=['GET'])
def get_distance_to_obstacles():
    global DTO
    dto = DTO
    DTO = 100000
    return str(dto)


@app.route('/LGSVL/Status/EstimatedTimeToCollision', methods=['GET'])
def get_estimated_time_to_collision():
    global ETTC
    ettc = ETTC
    ETTC = 100
    return str(ettc)

@app.route('/LGSVL/Status/Jerk', methods=['GET'])
def get_jerk():
    global JERK
    jerk = JERK
    JERK = 100
    return str(jerk)


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