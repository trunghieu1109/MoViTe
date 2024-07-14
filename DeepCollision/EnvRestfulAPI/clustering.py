from pathlib import Path

import numpy as np
import pandas as pd
from kneed import KneeLocator
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler
from incdbscan import IncrementalDBSCAN
import os

def cluster_df(df: pd.DataFrame):
    data = df.iloc[:, 1:]  # remove the first column
    # print(data.shape)
    
    k_neighbor = 10
    data_scaled = MinMaxScaler().fit_transform(data)
    knn = NearestNeighbors(n_neighbors=k_neighbor)
    neighbors = knn.fit(data_scaled)
    distances, _ = neighbors.kneighbors(data_scaled)
    sorted_distances = np.sort(distances, axis=0)[:, -1]

    i = np.arange(len(sorted_distances))
    knee = KneeLocator(
        i,
        sorted_distances,
        S=1.00,
        curve="convex",
        direction="decreasing",
    )
    
    if knee.knee:
        epsilon = sorted_distances[knee.knee]
    else:
        epsilon = sorted_distances[round(len(sorted_distances) / 2)]
        
    output_file = './epsilon_log.csv'
    
    if not os.path.exists(output_file):
        pd.DataFrame([['epsilon']]).to_csv(
            output_file,
            mode='w',
            header=False, index=None)
        
    pd.DataFrame([[epsilon]]).to_csv(
            output_file,
            mode='a',
            header=False, index=None)

    est = DBSCAN(eps=epsilon, min_samples=k_neighbor, metric="euclidean")
    clusters_src = est.fit_predict(data_scaled)
    
    df["cluster"] = clusters_src

    return clusters_src


def cluster(filename):
    df = pd.read_csv(filename)
    return cluster_df(df)