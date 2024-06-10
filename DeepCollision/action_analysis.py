import numpy as np
import pandas as pd
import argparse
import matplotlib.pyplot as plt

obs_time = 6.0

def action_analysis(exp_file, n_actions):
    
    df = pd.read_csv(exp_file)
    n_actions_ = int(n_actions)
    
    action_by_model = np.zeros((n_actions_,))
    action_randomly = np.zeros((n_actions_,))
    
    for index, row in df.iterrows():
        action = row['Action']
        type = row["Choosing_Type"]
        if type == "by model":
            action_by_model[int(action)] += 1
        else:
            action_randomly[int(action)] += 1
                
    print("Action chosen by model: ", action_by_model)
    print("Action chosen randomly: ", action_randomly)
    # print(np.sum(local_info) / len(local_info))
    
    # freq *= 100
    
    plt.plot(action_by_model)
    plt.xlabel('Action')
    plt.ylabel('Freq')
    plt.title('Action x Freq')
    plt.show()
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process input and output files.")
    parser.add_argument("exp_file", help="Experiment File")
    parser.add_argument("n_actions", help="Number of actions")

    args = parser.parse_args()

    action_analysis(args.exp_file, args.n_actions)