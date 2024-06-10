import numpy as np
import pandas as pd
import argparse
import matplotlib.pyplot as plt

obs_time = 6.0

def metrics_extract(exp_file):
    
    df = pd.read_csv(exp_file)
    
    collision = 0
    potential_collision = 0
    
    isCollision = False
    isPotential = False
    
    cnt_eps_collision = 0
    time_to_collision = 0
    
    avg_ttc = 0
    
    freq = np.zeros(165)
    
    per_info = []
    reward_info = []
    
    prev_position = np.array([0, 0, 0])
    
    prev_uid = ""
    
    distinct_collision = 0
    
    collision_by_random_action = 0
    collision_by_model_choosing_action = 0
    
    for index, row in df.iterrows():
        
        reward = float(row['Collision_Probability'])
        done = row['Done']
        action = row['Action']
        cp_list = row['Collision_Probability_Per_Step']
        state = row['State'][1:-1].split(",")[0].split(" ")
        state_ = [s for s in state if s != ""]
        uid = row["Collision_uid"]
        # type = row["Choosing_Type"]
        
        per_in = None
 
        freq[int(action)] += 1
 
        if reward > 0: 
            if reward == 1.0:
                
                if prev_uid == uid:
                    new_pos = np.array([float(state_[0]), float(state_[1]), float(state_[2])])
                    diff = np.linalg.norm(prev_position - new_pos)
                    
                    if diff > 2:
                        distinct_collision += 1
                        
                else:
                    distinct_collision += 1
                    
                prev_position = np.array([float(state_[0]), float(state_[1]), float(state_[2])])
                prev_uid = uid
                
                collision += 1
                
                # if type == "by model":
                #     collision_by_model_choosing_action += 1
                # else:
                #     collision_by_random_action += 1
                
                if not isCollision:
                    max_val = 0.0
                    index = 0
                    cnt = 0
                    
                    for cp in cp_list[1:-1].split(","):
                        fcp = float(cp)
                        if fcp > max_val:
                            max_val = fcp
                            index = cnt
                            
                        cnt += 1
                        
                    time_to_collision += 0.5 * (index + 1)
                
                isCollision = True
            else:
                potential_collision += 1
                time_to_collision += obs_time
        else:
            time_to_collision += obs_time
                
        if done:
            cnt_eps_collision += int(isCollision)
            avg_ttc += time_to_collision * int(isCollision)
            isCollision = False
            isPotential = False
            time_to_collision = 0
            prev_position = np.array([0, 0, 0])
            
            prev_uid = ""
            
        per_info.append(per_in)
        reward_info.append(reward)
                
    print("Collision: ", collision, "\nPotential Collision: ", potential_collision, "\nDistinct Collision: ", distinct_collision, "\nAverage time to collision: ", avg_ttc / max(cnt_eps_collision, 1))
    print("Collision by model: ", collision_by_model_choosing_action)
    print("Collision randomly: ", collision_by_random_action)
    
    freq /= np.sum(freq)
    
    # print(np.sum(local_info) / len(local_info))
    
    # freq *= 100
    
    # plt.scatter(per_info, reward_info, marker='o', color='b')
    # plt.xlabel('Perception Information')
    # plt.ylabel('Reward')
    # plt.title('Perception Information vs Reward')
    # plt.show()
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process input and output files.")
    parser.add_argument("exp_file", help="Experiment File")

    args = parser.parse_args()

    metrics_extract(args.exp_file)