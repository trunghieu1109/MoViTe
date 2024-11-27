from pathlib import Path

import numpy as np
import pandas as pd
import math
from kneed import KneeLocator
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler
import os
import json
import argparse
import pickle as pkl

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

from diversity_utils import merging_frames
from mpl_toolkits.mplot3d import Axes3D

NUMBER_OF_VEHICLE = 1

title = ['filepath', 'ego_position_x', 'ego_position_y', 'ego_position_z', 'ego_rotation_x', 'ego_rotation_y', 'ego_rotation_z', 'ego_velocity_x', 'ego_velocity_y', 'ego_velocity_z', 'ego_angular_velocity_x', 'ego_angular_velocity_y', 'ego_angular_velocity_z', 'number_of_vehicles',
         'first_npc_position_x', 'first_npc_position_y', 'first_npc_position_z', 'first_npc_rotation_x', 'first_npc_rotation_y', 'first_npc_rotation_z', 'first_npc_velocity_x', 'first_npc_velocity_y', 'first_npc_velocity_z', 'first_npc_angular_velocity_x', 'first_npc_angular_velocity_y', 'first_npc_angular_velocity_z', 'first_npc_type', 'first_npc_collided',
         'second_npc_position_x', 'second_npc_position_y', 'second_npc_position_z', 'second_npc_rotation_x', 'second_npc_rotation_y', 'second_npc_rotation_z', 'second_npc_velocity_x', 'second_npc_velocity_y', 'second_npc_velocity_z', 'second_npc_angular_velocity_x', 'second_npc_angular_velocity_y', 'second_npc_angular_velocity_z', 'second_npc_type', 'second_npc_collided',
         'third_npc_position_x', 'third_npc_position_y', 'third_npc_position_z', 'third_npc_rotation_x', 'third_npc_rotation_y', 'third_npc_rotation_z', 'third_npc_velocity_x', 'third_npc_velocity_y', 'third_npc_velocity_z', 'third_npc_angular_velocity_x', 'third_npc_angular_velocity_y', 'third_npc_angular_velocity_z', 'third_npc_type', 'third_npc_collided',
         'horizontal_entry', 'vertical_entry', 'junction_position', 'boundary', 'direction']

saved_scenarios = {}

def cluster_df(data_scaled, data_selected, k_neighbor):

    knn = NearestNeighbors(n_neighbors=k_neighbor)
    neighbors = knn.fit(data_selected)
    distances, _ = neighbors.kneighbors(data_selected)
    sorted_distances = np.sort(distances, axis=0)[:, -1]

    i = np.arange(len(sorted_distances))
    knee = KneeLocator(
        i,
        sorted_distances,
        S=1.00,
        curve="convex",
        direction="decreasing",
        interp_method='polynomial'
    )
    
    if knee.knee:
        epsilon = sorted_distances[knee.knee]
    else:
        epsilon = sorted_distances[round(len(sorted_distances) / 2)]
        
    db_clusters = DBSCAN(eps=epsilon, min_samples=k_neighbor, metric='cityblock')
    clusters_src = db_clusters.fit_predict(data_scaled)

    return clusters_src

def distance_to_ego(npc, ego):

    npc_pos = npc['position']
    ego_pos = ego['position']

    return math.sqrt((npc_pos['x'] - ego_pos['x']) ** 2 + (npc_pos['y'] - ego_pos['y']) ** 2 + (npc_pos['z'] - ego_pos['z']) ** 2)

def handle_rotation(rotation):
    if rotation > 180:
        return rotation - 360
    
    if rotation < -180:
        return rotation + 360
    
    return rotation

def extract_ego_info(ego_info):

    info = []

    info.append(ego_info['position']['x'])
    info.append(ego_info['position']['y'])
    info.append(ego_info['position']['z'])
    
    info.append(handle_rotation(ego_info['rotation']['x']))
    info.append(handle_rotation(ego_info['rotation']['y']))
    info.append(handle_rotation(ego_info['rotation']['z']))

    info.append(ego_info['velocity']['x'])
    info.append(ego_info['velocity']['y'])
    info.append(ego_info['velocity']['z'])
    
    info.append(ego_info['angular_velocity']['x'])
    info.append(ego_info['angular_velocity']['y'])
    info.append(ego_info['angular_velocity']['z'])

    return info

def extract_npc_info(npc_info):

    info = []

    info.append(npc_info['position']['x'])
    info.append(npc_info['position']['y'])
    info.append(npc_info['position']['z'])

    info.append(handle_rotation(npc_info['rotation']['x']))
    info.append(handle_rotation(npc_info['rotation']['y']))
    info.append(handle_rotation(npc_info['rotation']['z']))

    info.append(npc_info['velocity']['x'])
    info.append(npc_info['velocity']['y'])
    info.append(npc_info['velocity']['z'])
    
    info.append(npc_info['angular_velocity']['x'])
    info.append(npc_info['angular_velocity']['y'])
    info.append(npc_info['angular_velocity']['z'])

    if npc_info['type'] == 'Vehicle': 
        info.append(0)
    else:
        info.append(1)

    if npc_info['Collided_With_Ego'] == False:
        info.append(0)
    else:
        info.append(1)

    return info

def extract_junction_info(junction_info):

    info = []

    info_code = 0
    info_code1 = 0
    info_code2 = 0

    if junction_info['Has_Horizontal_Left_Entry'] == False:
        info_code = info_code + 0
    else:
        info_code = info_code + 1

    if junction_info['Has_Horizontal_Right_Entry'] == False:
        info_code = info_code + 0
    else:
        info_code = info_code + 10

    if junction_info['Has_Vertical_Inverse_Entry'] == False:
        info_code1 = info_code1 + 0
    else:
        info_code1 = info_code1 + 1

    if junction_info['Has_Vertical_Forward_Entry'] == False:
        info_code1 = info_code1 + 0
    else:
        info_code1 = info_code1 + 10

    if junction_info['Junction_Position'] == "out of":
        info_code2 = info_code2 + 0
    elif junction_info['Junction_Position'] == "near":
        info_code2 = info_code2 + 1
    else:
        info_code2 = info_code2 + 2

    info.append(info_code)
    info.append(info_code1)
    info.append(info_code2)

    return info

def extract_lane_info(lane_info):
    info = []

    boundary_type = {
        'Unknown': 0, 
        'Dotted Yellow': 1, 
        'Dotted White': 2, 
        'Solid Yellow': 3, 
        'Solid White': 4, 
        'Double Yellow': 5, 
        'Curb': 6}

    info_code = 0
    info_code2 = 0

    info_code += boundary_type[lane_info['Left_Boundary']]
    info_code += boundary_type[lane_info['Right_Boundary']] * 10

    if lane_info['Left_Lane_Direction'] == 'Opposite Direction to Ego':
        info_code2 = info_code2 + 0
    else:
        info_code2 = info_code2 + 1

    if lane_info['Right_Lane_Direction'] == 'Opposite Direction to Ego':
        info_code2 = info_code2 + 0
    else:
        info_code2 = info_code2 + 10

    info.append(info_code)
    info.append(info_code2)

    return info

def json_to_vec(json_scenario, file_name):

    time_slice_list = []

    for timestep in json_scenario:
        
        time_slice = [file_name]

        ego_info = None
        lane_info = None
        junction_info = None
        npc_info_list = []

        for key in json_scenario[timestep]:
            # print(key)
            if key.startswith("Ego"):
                
                # Ego Features
                
                time_slice += extract_ego_info(json_scenario[timestep][key])
                ego_info = json_scenario[timestep][key]
            elif key.startswith("Lane"):
                lane_info = json_scenario[timestep][key]
            elif key.startswith("Junction"):
                junction_info = json_scenario[timestep][key]
            elif key.startswith("NPC"):
                npc_info_list.append(json_scenario[timestep][key])

        # NPC Features
        
        npc_info_list.sort(key = lambda npc_info : distance_to_ego(npc_info, ego_info), reverse=True)
        
        time_slice.append(len(npc_info_list))

        if len(npc_info_list) > NUMBER_OF_VEHICLE:
            for i in range(0, NUMBER_OF_VEHICLE):
                time_slice += extract_npc_info(npc_info_list[i])

        else:
            for npc_info in npc_info_list:
                time_slice += extract_npc_info(npc_info)

            for i in range(0, NUMBER_OF_VEHICLE - len(npc_info_list)):
                time_slice += 14 * [-1]   

        # Junction Features
        time_slice += extract_junction_info(junction_info)
        
        # Lane Features
        time_slice += extract_lane_info(lane_info)

        time_slice_list.append(time_slice)

    merged_frames = merging_frames([row[1:] for row in time_slice_list])

    return merged_frames, time_slice_list

def cluster(base_dir):
    
    global merged_scenarios

    merged_vec_scenario_list = []
    vec_scenario_list = []

    for root, dirs, files in os.walk(base_dir):
        for directory in dirs:
            
            if directory.startswith("Random") or directory.startswith("DeepCollision"):
                continue
            
            directory_path = os.path.join(root, directory)

            for filename in os.listdir(directory_path):
                
                file_path = os.path.join(directory_path, filename)
                
                if os.path.isfile(file_path): 
                    print(f"Read file: {file_path}")
                    with open(file_path, 'r', encoding='utf-8') as file:
                        content = file.read()

                    json_scenario = json.loads(content)
                    merged_vec_scenario, vec_scenario = json_to_vec(json_scenario, directory + "/" + filename)
                    
                    # vec_scenario.append(directory + "/" + filename)

                    merged_vec_scenario_list += merged_vec_scenario
                    vec_scenario_list += vec_scenario
                    
                    saved_scenarios[directory + "&" + filename] = {
                        'name': directory + "/" + filename,
                        'index': len(merged_vec_scenario_list),
                        'feature': vec_scenario,
                        'merged': merged_vec_scenario,
                        'json': json_scenario
                    }
                    
    # with open("vec_scenario_list.pkl", "wb") as file1:
    #     pkl.dump(vec_scenario_list, file1)
        
    # file1.close()
        
    # with open("merged_vec_scenario_list.pkl", "wb") as file2:
    #     pkl.dump(merged_vec_scenario_list, file2)
        
    # file2.close()
        
    # with open("saved_scenarios.pkl", "wb") as file3:
    #     pkl.dump(saved_scenarios, file3)
        
    # file3.close()
                    
    # with open("vec_scenario_list.pkl", "rb") as file1:
    #     vec_scenario_list = pkl.load(file1)
        
    # file1.close()
        
    # with open("merged_vec_scenario_list.pkl", "rb") as file2:
    #     merged_vec_scenario_list = pkl.load(file2)
        
    # file2.close()
        
    # with open("saved_scenarios.pkl", "rb") as file3:
    #     saved_scenarios = pkl.load(file3)
        
    # file3.close()
    
    # with open("cluster_res.pkl", "rb") as file4:
    #     cluster_res = pkl.load(file4)
        
    # file4.close()

    merged_vec_scenario_list_ = np.array(merged_vec_scenario_list)
    print(merged_vec_scenario_list_.shape)
    
    merged_vec_scenario_list_ = np.unique(merged_vec_scenario_list_, axis=0)
    print("Unique rows in selectedData:", merged_vec_scenario_list_.shape[0])

    cluster_res = cluster_df(merged_vec_scenario_list_, merged_vec_scenario_list_, 225)
    
    data = saved_scenarios['sanfrancisco-road2-scenarios&scenario18.json']
    
    print(cluster_res[data['index']:data['index'] + len(data['merged'])])

    pca = PCA(n_components=3)  
    data_3d = pca.fit_transform(merged_vec_scenario_list_)

    # Visualize dữ liệu 2D
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plot
    scatter = ax.scatter(data_3d[:, 0], data_3d[:, 1], data_3d[:, 2], c=cluster_res, cmap='viridis', s=50)
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.colorbar(scatter, label='Cluster Label')  # Thanh màu

    plt.title('3D Scatter Plot')
    output_file = "dbscan_clusters.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')  # Lưu file với độ phân giải 300 DPI
    plt.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Process input and output files.")
    parser.add_argument("base_dir", help="Scenario Directory Path")

    args = parser.parse_args()

    cluster(args.base_dir)
