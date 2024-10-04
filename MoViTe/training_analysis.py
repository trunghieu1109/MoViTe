import numpy as np
import pandas as pd
import argparse
import matplotlib.pyplot as plt

obs_time = 6.0

def cumulative_reward(reward_list):
    
    result = 0
    
    for reward in reward_list[::-1]:
        result = result * 0.9 + reward
        
    return result


def action_analysis(exp_file):
    
    df = pd.read_csv(exp_file)
    
    current_eps = -1
    
    reward_list = []
    
    interval = 15
    
    avg_return = 0
    avg_return_list = []
    
    return_ = []
    
    for index, row in df.iterrows():
        eps = row["Episode"]
        reward = row["Reward"]
        
        if current_eps != eps:
            
            return__ = cumulative_reward(reward_list)
            
            return_.append(return__)
            
            
            
            if eps >= interval:
                avg_return = (avg_return * interval - return_[int(eps) - interval] + return__) / interval
                avg_return_list.append(avg_return)
                
            current_eps = eps
            reward_list = []
            
            
        reward_list.append(reward)
            

                
    
    plt.plot(avg_return_list)
    plt.xlabel("Episode")
    plt.ylabel('Return')
    plt.title('Avg Return in ' + str(interval) + ' eps')
    plt.show()
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process input and output files.")
    parser.add_argument("exp_file", help="Experiment File")
    
    args = parser.parse_args()

    action_analysis(args.exp_file)