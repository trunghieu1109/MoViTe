import numpy as np
import pandas as pd
import argparse
import matplotlib.pyplot as plt

obs_time = 6.0

def metrics_extract(exp_file):
    
    df = pd.read_csv(exp_file)
    
    for index, row in df.iterrows():
        
        cp_list = row['Collision_Probability_Per_Step']
        for cp in cp_list[1:-1].split(","):
            pd.DataFrame([[cp]]).to_csv(
                "./reward_result.csv", 
                mode='a',
                header=False, 
                index=None)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process input and output files.")
    parser.add_argument("exp_file", help="Experiment File")

    args = parser.parse_args()

    metrics_extract(args.exp_file)