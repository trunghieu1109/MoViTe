from clustering import cluster
import time
import numpy as np
import os
import pandas as pd
from tsfresh.feature_extraction import feature_calculators
from lgsvl.agent import NpcVehicle
# from movite_api_server import get_apollo_ctrl_msg

segment_counter = 0
sliding_window_size = 4
num_of_vec = 48
num_of_seg = int(num_of_vec / sliding_window_size)

def hamming_distance(a, b):
    length = len(a)
    
    dis = 0
    
    for i in range(0, length):
        if a[i] != b[i]:
            dis += 1
        else:
            if a[i] == -1:
                dis += 1

    return dis

def calculate_diversity_level(cluster_result):
    global num_of_seg
    segment_len = len(cluster_result)
    
    mapped_current_scenario = cluster_result[segment_len - num_of_seg:segment_len]
    print(mapped_current_scenario)
    
    min_dis = num_of_seg
    
    for i in range(0, int(segment_len / num_of_seg) - 1):

        mapped_scenario = cluster_result[i * num_of_seg:(i + 1) * num_of_seg]
        
        min_dis = min(min_dis, hamming_distance(mapped_current_scenario, mapped_scenario))
        
        print("Targeted Scenario: ", mapped_scenario, hamming_distance(mapped_current_scenario, mapped_scenario))
       
    min_dis /= num_of_seg
        
    if min_dis <= 0.25:
        return -1

    return min_dis


def clustering(clustering_timestamp):
    
    output_file = './merged_state/merged_state_{}.csv'.format(clustering_timestamp)
    
    cluster_result = cluster(output_file)
    
    return cluster_result

def merging_frame(frames, clustering_timestamp):
    
    global segment_counter
    # global violation_segment
    global sliding_window_size
    
    np_frames = np.array(frames)
    
    num_features = np_frames.shape[1]
    num_frames = np_frames.shape[0]
    
    merged_frame_list = []
    
    output_file = './merged_state/merged_state_{}.csv'.format(clustering_timestamp)
    
    # violation_segment.append([])
    
    if not os.path.exists(output_file):
        header = ['ID']
        for j in range(0, num_features * 7):
            header.append(str(j))
        
        header.append('cluster')
        
        pd.DataFrame([header]).to_csv(
                output_file,
                mode='w',
                header=False, index=None)
    
    for j in range(0, int(num_frames / sliding_window_size)):
        # print("(", sliding_window_size * j, ',', sliding_window_size * (j + 1), ')')
        segment_ = [str(segment_counter)]
        # violation_segment[-1].append(segment_counter)
        for i in range(0, num_features):
            feature_list = np_frames[j * sliding_window_size : (j + 1) * sliding_window_size, i]
            # print(feature_list)
            mean = feature_calculators.mean(feature_list)
            maximum = feature_calculators.maximum(feature_list)
            minimum = feature_calculators.minimum(feature_list)
            mean_change = feature_calculators.mean_change(feature_list)
            mean_abs_change = feature_calculators.mean_abs_change(feature_list)
            variance = feature_calculators.variance(feature_list)
            cid_ce = feature_calculators.cid_ce(feature_list, True)
            
            merged_frame = [mean, maximum, minimum, mean_change, mean_abs_change, variance, cid_ce]
        
            segment_ += merged_frame
            
        segment_.append(-1)
            
        pd.DataFrame([segment_]).to_csv(
                output_file,
                mode='a',
                header=False, index=None)
        
        segment_counter += 1
        
