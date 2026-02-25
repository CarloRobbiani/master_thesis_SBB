import torch
import pandas as pd
import numpy as np
from  torch_geometric.data import Data
from torch_geometric.nn import GraphSAGE
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler

# 1. Generate fake data

np.random.seed(42)

num_trains = 5
time_steps = 10

rows = []

for train in range(num_trains):
    current_delay = np.random.randint(0, 5)

    for t in range(time_steps):
        delay_change = np.random.randint(-1, 3)
        current_delay = max(0, current_delay + delay_change)

        rows.append({
            "node_id": len(rows),
            "train_id": train,
            "time": t,
            "track_id": np.random.randint(0, 3),
            "current_delay": current_delay,
            "speed": np.random.uniform(60, 140),
            "headway": np.random.uniform(1, 10),
            "temperature": np.random.uniform(-5, 25),
            "target_delay_5min": current_delay + np.random.randint(-1, 4)
        })

df = pd.DataFrame(rows)


# 2. Build Node features

feature_cols = [
    "current_delay",
    "speed",
    "headway",
    "temperature"
]

X = df[feature_cols].values
y = df["target_delay_5min"].values

scaler = StandardScaler()
X = scaler.fit_transform(X)

x_tensor = torch.tensor(X, dtype=torch.float)
y_tensor = torch.tensor(y, dtype=torch.float)

# 3. Build Edges

edges = []

# A) Temporal edges (same train consecutive time)
for train_id in df["train_id"].unique():
    train_df = df[df["train_id"] == train_id].sort_values("time")
    indices = train_df.index.tolist()
    for i in range(len(indices) - 1):
        edges.append([indices[i], indices[i+1]])

# B) Interaction edges (same track & same time)
for t in df["time"].unique():
    time_df = df[df["time"] == t]
    for track in time_df["track_id"].unique():
        subset = time_df[time_df["track_id"] == track]
        indices = subset.index.tolist()
        for i in range(len(indices)):
            for j in range(i+1, len(indices)):
                edges.append([indices[i], indices[j]])

edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

# 4. Create Graph Object

data = Data(x=x_tensor, edge_index=edge_index, y=y_tensor)

# 5. Define GNN model

class DelayGNN(torch.nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = GraphSAGE(in_channels, 32, num_layers=1)
        self.conv2 = GraphSAGE(32, 16, num_layers=1)
        self.linear = torch.nn.Linear(16, 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)

        x = self.conv2(x, edge_index)
        x = F.relu(x)

        x = self.linear(x)

        return x.squeeze()

model = DelayGNN(in_channels=len(feature_cols))
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# 6. Train model

for epoch in range(100):
    model.train()
    optimizer.zero_grad()

    out = model(data)
    loss = F.mse_loss(out, data.y)

    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

print("Training finished.")