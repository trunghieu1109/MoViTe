import socket
import time

import cv2
from flask import Flask, request
import os
from datetime import timedelta
import json
import lgsvl
import numpy as np
from collision_utils_short_paper import pedestrian, npc_vehicle, calculate_measures
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
ETTC = 1000
DTO = 100000
JERK = 10000
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
prev_acc = 0

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
def calculate_metrics(agents, ego):
    global probability
    global TTC
    global DTO
    global JERK
    global collision_tag

    global SAVE_SCENARIO

    global collision_object
    global collision_speed
    global collision_uid
    global prev_acc

    collision_object = None
    collision_speed = 0  # 0 indicates there is no collision occurred.
    collision_speed_ = 0
    collision_type = "None"
    collision_uid = "No collision"

    print("Calculating metrics ....")

    ETTC_list = []
    distance_list = []
    probability_list = []
    JERK_list = []
    
    i = 0
    time_step = 0.5

    for i in range (0, int(observation_time / time_step)):
        
        # check apollo's modules status from dreamview
        check_modules_status()
        
        # run simulator
        sim.run(time_limit=time_step) 
        
        ego_acc = get_ego_acceleration()
        
        JERK_list.append(abs(ego_acc - prev_acc) / 0.5)
        
        npc_state = []
        isNpcVehicle = []
        for j in range(1, len(agents)):
            state_ = agents[j].state
            npc_state.append(state_)
            isNpc = (isinstance(agents[j], NpcVehicle))
            isNpcVehicle.append(isNpc)

        ego_state = ego.state

        thread = threading.Thread(
            target=calculate_measures_thread,
            args=(state_list, ego_state, isNpcVehicle, ETTC_list, 
                  distance_list, probability_list, collision_tag,)
        )

        thread.start()

        if collision_tag:
            collision_tag = False

    # if SAVE_SCENARIO:
    collision_type, collision_speed_, collision_uid_ = get_collision_info()
    if collision_speed_ == -1:
        collision_speed_ = speed
    
    TTC = round(min(ETTC_list), 6)
    DTO = round(min(distance_list), 6)
    JERK = round(max(JERK_list), 6)
    probability = round(max(probability_list), 6)
    
    return {'ETTC': ETTC_list, 'distance': distance_list, 'JERK': JERK_list, 'collision_uid': collision_uid_, 'probability': probability_list,} 


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
    global prev_lane_id
    global DREAMVIEW
    global prev_acc
    
    prev_acc = 0
    prev_lane_id = ""

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
    # roadTransform_start = open(
    #     'Transform/transform-road' + road_num + '-start', 'rb')
    # state.transform = torch.load(
    #     './Transform/{}.pt'.format("road" + "4" + "_start"))
    
    with open('position.pkl') as file:
        map_pos = pickle.load(file)
        
    state.transform.position.x = map_pos[scene][road_num]['init_pos']['x']
    state.transform.position.y = map_pos[scene][road_num]['init_pos']['y']
    state.transform.position.z = map_pos[scene][road_num]['init_pos']['z']
    state.transform.rotation.y = map_pos[scene][road_num]['init_pos']['ry']
    state.transform.rotation.x = map_pos[scene][road_num]['init_pos']['rx']

    forward = lgsvl.utils.transform_to_forward(state.transform)

    state.velocity = 3 * forward

    EGO = sim.add_agent("8e776f67-63d6-4fa3-8587-ad00a0b41034", lgsvl.AgentType.EGO, state)
    EGO.connect_bridge(os.environ.get("BRIDGE_HOST", APOLLO_HOST), BRIDGE_PORT)
    
    DREAMVIEW = lgsvl.dreamview.Connection(sim, EGO, APOLLO_HOST, str(DREAMVIEW_PORT))

    sensors = EGO.get_sensors()
    sim.get_agents()[0].on_collision(on_collision)

    data = {'road_num': road_num}

    requests.post(
        f"http://localhost:8933/LGSVL/SetDestination?des_x={map_pos[scene][road_num]['des_pos']['x']}" 
        + f"&des_y={map_pos[scene][road_num]['des_pos']['y']}&des_z={map_pos[scene][road_num]['des_pos']['z']}")

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

def check_modules_status():
    global EGO
    global DREAMVIEW
    
    # print("Create connection")
    # print("Create connection successfully")
    
    
    modules_status = DREAMVIEW.get_module_status()
    
    stop = False
    
    stop_time = 0
    
    if modules_status['Localization'] == False:
        print(20*'*', 'LOCALIZATION STOPPED', 20*'*')
        DREAMVIEW.enable_module('Localization')
        print(20*'*', 'LOCALIZATION ENABLED', 20*'*')
        stop = True
        stop_time = max(120, stop_time)
        
    if modules_status['Prediction'] == False:
        print(20*'*', 'PREDICTION STOPPED', 20*'*')
        DREAMVIEW.enable_module('Prediction')
        print(20*'*', 'PREDICTION ENABLED', 20*'*')
        stop = True
        stop_time = max(120, stop_time)
        
    if modules_status['Transform'] == False:
        print(20*'*', 'TRANSFORM STOPPED', 20*'*')
        DREAMVIEW.enable_module('Transform')
        print(20*'*', 'TRANSFORM ENABLED', 20*'*')
        stop = True
        stop_time = max(120, stop_time)
        
    if modules_status['Control'] == False:
        print(20*'*', 'CONTROL STOPPED', 20*'*')
        DREAMVIEW.enable_module('Control')
        print(20*'*', 'CONTROL ENABLED', 20*'*')
        stop = True
       stop_time = max(120, stop_time)
        
    if modules_status['Perception'] == False:
        print(20*'*', 'PERCEPTION STOPPED', 20*'*')
        DREAMVIEW.enable_module('Perception')
        print(20*'*', 'PERCEPTION ENABLED', 20*'*')
        stop = True
        stop_time = max(300, stop_time)
        
    if modules_status['Routing'] == False:
        print(20*'*', 'ROUTING STOPPED', 20*'*')
        DREAMVIEW.enable_module('Routing')
        print(20*'*', 'ROUTING ENABLED', 20*'*')
        stop = True
        stop_time = max(120, stop_time)
        
    if modules_status['Planning'] == False:
        print(20*'*', 'PLANNING STOPPED', 20*'*')
        DREAMVIEW.enable_module('Planning')
        print(20*'*', 'PLANNING ENABLED', 20*'*')
        stop = True
        stop_time = max(120, stop_time)
        
    if stop:
        time.sleep(stop_time)

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

    state_dict = {'x': position.x, 'y': position.y, 'z': position.z,
                  'rx': rotation.x, 'ry': rotation.y, 'rz': rotation.z,
                  'rain': weather.rain, 'fog': weather.fog, 'wetness': weather.wetness,
                  'timeofday': sim.time_of_day, 'signal': interpreter_signal(signal.current_state),
                  'speed': speed,}

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

@app.route('/LGSVL/Status/DistanceToObstacles', methods=['GET'])
def get_distance_to_obstacles():
    global DTO
    dto = DTO
    DTO = 0
    return str(dto)


@app.route('/LGSVL/Status/EstimatedTimeToCollision', methods=['GET'])
def get_estimated_time_to_collision():
    global ETTC
    ettc = ETTC
    ETTC = 0
    return str(ettc)

@app.route('/LGSVL/Status/Jerk', methods=['GET'])
def get_jerk():
    global JERK
    jerk = JERK
    JERK = 0
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