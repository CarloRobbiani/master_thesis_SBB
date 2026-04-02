import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import os
import sys
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

from adjacency import create_adj_matrix
from sklearn.metrics import root_mean_squared_error


class ChebGraphConv(nn.Module):
    def __init__(self, in_channels, out_channels, K):
        super().__init__()
        self.K = K
        self.linear = nn.Linear(in_channels * K, out_channels)

    def forward(self, x, laplacian):
        # x: [B, T, N, F]
        B, T, N, F_in = x.shape

        cheb_polys = [x]

        if self.K > 1:
            cheb_polys.append(torch.einsum("ij,btjf->btif", laplacian, x))

        for k in range(2, self.K):
            cheb_polys.append(
                2 * torch.einsum("ij,btjf->btif", laplacian, cheb_polys[-1])
                - cheb_polys[-2]
            )

        x_cat = torch.cat(cheb_polys, dim=-1)
        return self.linear(x_cat)


class FeatureAttention(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.fc = nn.Linear(feature_dim, feature_dim)

    def forward(self, x):
        # x: [B, T, N, F]
        weights = torch.softmax(self.fc(x), dim=-1)
        return x * weights

class TemporalAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        # x: [B, T, N, F]
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        scores = torch.einsum("btnd,bsnd->btns", Q, K)
        scores = scores / (x.shape[-1] ** 0.5)

        attn = torch.softmax(scores, dim=-1)
        return torch.einsum("btns,bsnd->btnd", attn, V)

class STBlock(nn.Module):
    def __init__(self, in_channels, hidden_dim, K):
        super().__init__()

        self.feature_att = FeatureAttention(in_channels)
        self.temporal_att = TemporalAttention(in_channels)

        self.graph_conv = ChebGraphConv(in_channels, hidden_dim, K)

        self.temporal_conv = nn.Conv2d(
            hidden_dim,
            hidden_dim,
            kernel_size=(3, 1),
            padding=(1, 0)
        )

        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, laplacian):
        # x: [B, T, N, F]

        residual = x

        x = self.feature_att(x)
        x = self.temporal_att(x)

        x = self.graph_conv(x, laplacian)

        x = x.permute(0, 3, 1, 2)
        x = self.temporal_conv(x)
        x = x.permute(0, 2, 3, 1)

        #return self.norm(x + residual)
        return self.norm(x)
    
def compute_laplacian(adj):
    D = torch.diag(torch.sum(adj, dim=1))
    L = D - adj
    D_inv_sqrt = torch.diag(1.0 / torch.sqrt(torch.sum(adj, dim=1) + 1e-6))
    return D_inv_sqrt @ L @ D_inv_sqrt

def prepare_laplacian(station_list_path, device):
    """
    Takes the station_list path that points to a .csv file and returns the 
    laplacian from the adjacency matrix
    """
    adj = torch.tensor(create_adj_matrix(station_list_path))
    laplacian = compute_laplacian(adj).float().to(device)
    lambda_max = torch.linalg.eigvals(laplacian).real.max()
    laplacian = (2 / lambda_max) * laplacian - torch.eye(laplacian.size(0))
    return laplacian

def filter_tensors(data_tensor: torch.tensor, train_end, val_end, timestamps):
    """
    Filter the tensor based on the given timestamps.
    Returns a train, test and val tensor
    """
    # Remove timezone if present
    if isinstance(timestamps, pd.DatetimeIndex):
        if timestamps.tz is not None:
            timestamps = timestamps.tz_convert(None)
    elif isinstance(timestamps, pd.Series):
        if timestamps.dt.tz is not None:
            timestamps = timestamps.dt.tz_convert(None)
    else:
        # If it's a plain Index, try to convert to DatetimeIndex
        timestamps = pd.to_datetime(timestamps, utc=True)
        if hasattr(timestamps, 'tz') and timestamps.tz is not None:
            timestamps = timestamps.tz_convert(None)
    
    train_idx = timestamps < train_end
    val_idx   = (timestamps >= train_end) & (timestamps < val_end)
    test_idx  = timestamps >= val_end

    train = data_tensor[train_idx]
    val = data_tensor[val_idx]
    test = data_tensor[test_idx]

    return train, val, test
    

def create_df_tensors(df: pd.DataFrame):
    """
    Creates tensors from the train Dataframe
    Returns: station_tensor, external_tensor, target_tensor, timestamps
    """
    # Features the stations should have
    station_feature_cols = [
        "EVENT_TYPE",
        "EVENT_SERVED",
        "PLAN_STOP_TYPE", 
        "OPERATION_DAY_PERIOD_IDENTIFIER_COARSE",
        'OPERATION_TRAFFIC_CATEGORY_ABBREVIATION',
        'PLAN_FORMATION_MAXIMAL_VELOCITY',
        "hour_sin",
        "hour_cos",
        "dow_sin",
        "dow_cos"
    ]

    # Features for the external tensor
    external_cols = [
        'tre200s0', 'fkl010z1', 'fu3010z0', 'rre150z0',
       'htoauts0', 'hto000d0'
    ]

    target_col = "DAILY_PLAN_OPERATIONAL_DELAY_SEC"

    # --- Sort ---
    df = df.sort_values(["OPERATION_ACTUAL_TIMESTAMP", "OPERATING_POINT_ABBREVIATION"])


    # ---  Convert Boolean to int ---
    print("converting boolean to int...")
    df["EVENT_SERVED"] = df["EVENT_SERVED"].astype(int)


    # --- Categorical encoding ---
    print("categoircal encoding...")
    exclude_cols = ["OPERATING_POINT_ABBREVIATION", "OPERATION_ACTUAL_TIMESTAMP"]

    cat_cols = [
        col for col in df.select_dtypes(include="object").columns
        if col not in exclude_cols
    ]

    for col in cat_cols:
        df[col] = df[col].astype("category").cat.codes

    # --- Handle missing values ---
    print("handling missing values...")
    df = df.fillna(0)

    # --- Create consistent indices ---
    timestamps = sorted(df["OPERATION_ACTUAL_TIMESTAMP"].unique())
    stations = ["BI","TUE","TWN","LIG","CHAV","POU","NV","LD","CRNE","CORN","SBLB","NE"]

    timestamp_to_idx = {t: i for i, t in enumerate(timestamps)}
    station_to_idx = {s: i for i, s in enumerate(stations)}

    T_total = len(timestamps)
    N = len(stations)

    F = len(station_feature_cols)
    E = len(external_cols)

    # ---  Initialize tensors ---
    station_tensor = np.full((T_total, N, F), np.nan, dtype=np.float32)
    target_tensor = np.full((T_total, N), np.nan, dtype=np.float32)
    external_tensor = np.full((T_total, E), np.nan, dtype=np.float32)

    # --- Fill tensors safely ---
    for _, row in df.iterrows():

        t = timestamp_to_idx[row["OPERATION_ACTUAL_TIMESTAMP"]]
        n = station_to_idx[row["OPERATING_POINT_ABBREVIATION"]]

        # Node features
        station_tensor[t, n, :] = row[station_feature_cols].values

        # Target
        target_tensor[t, n] = row[target_col]
        #target_tensor = np.nan_to_num(target_tensor, nan=0.0)

        # External (same for all stations)
        if np.isnan(external_tensor[t, 0]):
            external_tensor[t, :] = row[external_cols].values

    # --- Handle missing values ---

    target_tensor = np.nan_to_num(target_tensor, nan=0.0)   # Replace NaNs in target NO DELAY = 0

    # forward fill over time
    for f in range(E):
        series = pd.Series(external_tensor[:, f])
        external_tensor[:, f] = series.ffill().fillna(0)

    for n in range(N):
        for f in range(F):
            series = pd.Series(station_tensor[:, n, f])
            station_tensor[:, n, f] = series.ffill().fillna(0)

    return station_tensor, external_tensor, target_tensor, timestamps


def evaluate(model, dataloader, laplacian, criterion, device):

    model.eval()

    total_loss = 0
    total_samples = 0

    all_preds = []
    all_targets = []

    with torch.no_grad():

        for x, e, y in dataloader:

            x = x.to(device)
            e = e.to(device)
            y = y.to(device)
            y = y.permute(0, 2, 1)

            pred = model(x, e, laplacian)

            #loss = criterion(pred, y)
            loss = torch.nn.functional.smooth_l1_loss(pred, y)

            batch_size = x.shape[0]

            total_loss += loss.item() * batch_size
            total_samples += batch_size

            # Collect ONLY valid values for RMSE
            all_preds.append(pred.cpu())
            all_targets.append(y.cpu())

    # Concatenate all batches
    all_preds = torch.cat(all_preds).cpu().numpy()
    all_targets = torch.cat(all_targets).cpu().numpy()

    all_preds = all_preds.reshape(-1)
    all_targets = all_targets.reshape(-1)

    rmse = root_mean_squared_error(all_targets, all_preds)

    return total_loss / total_samples, rmse