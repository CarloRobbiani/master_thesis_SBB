import numpy as np
from torch.utils.data import DataLoader
import torch

from graph_models.stgcn_graph import STGCN, train_gnn
from rail_dataset_class import RailDataset

num_samples = 200
num_nodes = 30
seq_len = 12
num_features = 4
horizon = 12

X = np.random.randn(num_samples, num_nodes, seq_len, num_features)
Y = np.random.randn(num_samples, num_nodes, horizon)

dataset = RailDataset(X, Y)
train_loader = DataLoader(dataset, batch_size=16, shuffle=True)

# dummy graph: ring structure
edges = []
for i in range(num_nodes):
    edges.append([i, (i+1) % num_nodes])
    edges.append([(i+1) % num_nodes, i])

edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

model = STGCN(num_nodes=num_nodes, num_features=num_features, hidden_channels=32, horizon=horizon)
train_gnn(model, train_loader, train_loader, edge_index, epochs=3, lr=1e-3)
