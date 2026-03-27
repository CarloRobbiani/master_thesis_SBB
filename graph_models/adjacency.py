import pandas as pd
import numpy as np

def create_adj_matrix(station_list_path : str = "stations.csv"):
    """
    Creates an adjacency matrix based on a .csv file
    Returns the adjacency matrix
    """

    df = pd.read_csv(station_list_path, header=None)

    stations = df.iloc[1].tolist()  # second row are abbreviations
    n = len(stations)

    adj = np.zeros((n, n))

    for i in range(n - 1):
        adj[i, i + 1] = 1
        adj[i + 1, i] = 1