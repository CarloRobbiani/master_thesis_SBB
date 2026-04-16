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
from sklearn.preprocessing import StandardScaler


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
    def __init__(self, in_channels, hidden_dim, K, dropout=0.1):
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

        self.dropout = nn.Dropout(p=dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, laplacian):
        # x: [B, T, N, F]

        residual = x

        x = self.feature_att(x)
        x = self.temporal_att(x)

        x = self.graph_conv(x, laplacian)

        x = x.permute(0, 3, 1, 2)
        x = self.temporal_conv(x)
        x = self.dropout(x)
        x = x.permute(0, 2, 3, 1)

        return self.norm(x + residual)

    
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
    laplacian = (2 / lambda_max) * laplacian - torch.eye(laplacian.size(0), device=device)
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


def evaluate(model, dataloader, laplacian, criterion, device):

    model.eval()

    total_loss = 0
    total_samples = 0

    all_preds = []
    all_targets = []

    with torch.no_grad():

        for x, e, y, m in dataloader:

            x = x.to(device)
            e = e.to(device)
            y = y.to(device)
            y = y.permute(0, 2, 1)
            m = m.to(device)
            m = m.permute(0, 2, 1)

            pred = model(x, e, laplacian)

            loss = ((pred - y) ** 2) * m
            loss = loss.sum() / (m.sum() + 1e-6)

            batch_size = x.shape[0]

            total_loss += loss.item() * batch_size
            total_samples += batch_size

            all_preds.append(pred[m].cpu())
            all_targets.append(y[m].cpu())

    all_preds = torch.cat(all_preds).cpu().numpy()
    all_targets = torch.cat(all_targets).cpu().numpy()

    all_preds = all_preds.reshape(-1)
    all_targets = all_targets.reshape(-1)

    rmse = root_mean_squared_error(all_targets, all_preds)

    return total_loss / total_samples, rmse


def load_and_pivot(path, STATION_FEATURE_COLS, EXTERNAL_COLS):
    """
    Load the parquet file and return arrays suitable for the MATGCN model.

    Returns:
        station_arr  : np.ndarray of shape (T, N, F)
        external_arr : np.ndarray of shape (T, E)
        target_arr   : np.ndarray of shape (T, N)
        stations     : list of station ids (length N)
    """
  
    df = pd.read_parquet(path)

    STATION_COL = "OPERATING_POINT_ABBREVIATION"
    DATE_COL    = "OPERATION_PLANNED_TIMESTAMP"
    TARGET_COL  = "DAILY_PLAN_OPERATIONAL_DELAY_SEC"

    # -- temporal encoding --
    df["hour_sin"] = np.sin(2 * np.pi * df["OPERATION_ACTUAL_TIMESTAMP"].dt.hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["OPERATION_ACTUAL_TIMESTAMP"].dt.hour / 24)
    df["dow_sin"]  = np.sin(2 * np.pi * df["OPERATION_ACTUAL_TIMESTAMP"].dt.dayofweek / 7)
    df["dow_cos"]  = np.cos(2 * np.pi * df["OPERATION_ACTUAL_TIMESTAMP"].dt.dayofweek / 7)
    df["hto000d0"] = df["hto000d0"].fillna(0)
    df = df.drop(["date", "days"], axis=1)
    df = df.sort_values([DATE_COL, STATION_COL]).reset_index(drop=True)

    # -- categorical encoding --
    exclude_cols = ["OPERATION_ACTUAL_TIMESTAMP", TARGET_COL, DATE_COL]
    cat_cols = [
        col for col in df.select_dtypes(include="object").columns
        if col not in exclude_cols
    ]
    for col in cat_cols:
        df[col] = df[col].astype("category").cat.codes

    # -- STEP 1: aggregate arrivals and departures separately --

    def _mode(s):
        m = s.mode()
        return m.iloc[0] if len(m) else np.nan

    # Build per-event-type aggregation for features
    cat_station_cols = [
        c for c in STATION_FEATURE_COLS
        if df[c].dtype in (object, "category") or str(df[c].dtype) == "int8"
           or df[c].nunique() < 20
    ]
    num_station_cols = [c for c in STATION_FEATURE_COLS if c not in cat_station_cols]

    agg_dict_feat = {c: _mode for c in cat_station_cols}
    agg_dict_feat.update({c: "mean" for c in num_station_cols})
    agg_dict_feat.update({c: "first" for c in EXTERNAL_COLS})

    # Aggregate departures
    dep_df = df[df["EVENT_TYPE"] == 0] if df["EVENT_TYPE"].dtype != object \
             else df[df["EVENT_TYPE"] == "departure"]
    # compute target mean per event type separately.

    # Re-derive departure/arrival targets from the encoded df
    grp = df.groupby([DATE_COL, STATION_COL])

    # Target: mean departure delay per (timestamp, station)
    dep_code = df["EVENT_TYPE"].unique()  # after encoding, find which code = departure
    # Since encoding is alphabetical: 'arrival'=0, 'departure'=1
    dep_target = (df[df["EVENT_TYPE"] == 1]          # 1 = departure after alpha encoding
                    .groupby([DATE_COL, STATION_COL])[TARGET_COL]
                    .mean()
                    .rename("target_dep"))

    arr_target = (df[df["EVENT_TYPE"] == 0]          # 0 = arrival
                    .groupby([DATE_COL, STATION_COL])[TARGET_COL]
                    .mean()
                    .rename("target_arr"))

    # Context features: aggregate over ALL rows (arrival + departure) per cell
    feat_agg = grp.agg(agg_dict_feat)

    # Combine into one flat table
    combined = feat_agg.join(dep_target).join(arr_target).reset_index()

    # -- build station/external/target dense arrays --
    stations   = sorted(combined[STATION_COL].unique())
    timestamps = sorted(combined[DATE_COL].unique())
    N = len(stations)
    T = len(timestamps)
    print(f"  Timesteps : {T},  Stations : {N}")

    # Station features = original STATION_FEATURE_COLS
    # + arrival delay as an extra autoregressive input feature
    all_station_cols = STATION_FEATURE_COLS + ["target_arr"]
    F  = len(all_station_cols)
    E  = len(EXTERNAL_COLS) if EXTERNAL_COLS else 1

    station_arr  = np.full((T, N, F), np.nan, dtype=np.float32)
    external_arr = np.full((T, E),    np.nan, dtype=np.float32)
    target_arr   = np.full((T, N),    np.nan, dtype=np.float32)

    station_idx = {s: i for i, s in enumerate(stations)}
    time_idx    = {t: i for i, t in enumerate(timestamps)}

    for _, row in combined.iterrows():
        t = time_idx[row[DATE_COL]]
        n = station_idx[row[STATION_COL]]

        station_arr[t, n, :] = [
            row[c] if pd.notna(row.get(c)) else np.nan
            for c in all_station_cols
        ]
        target_arr[t, n] = row["target_dep"] if pd.notna(row.get("target_dep")) else np.nan

        if EXTERNAL_COLS and np.isnan(external_arr[t, 0]):
            external_arr[t, :] = [
                row[c] if pd.notna(row.get(c)) else 0.0
                for c in EXTERNAL_COLS
            ]

    print(f"  Station features ({F}): {all_station_cols}")
    print(f"  External features ({E}): {EXTERNAL_COLS}")
    print(f"  NaNs before fill — station: {np.isnan(station_arr).sum()}, "
          f"external: {np.isnan(external_arr).sum()}, "
          f"target: {np.isnan(target_arr).sum()}")

    # -- forward-fill missing cells (limit = 3 steps) ---
    for e in range(E):
        s = pd.Series(external_arr[:, e])
        external_arr[:, e] = s.ffill().fillna(0).values

    for n in range(N):
        for f in range(F):
            s = pd.Series(station_arr[:, n, f])
            station_arr[:, n, f] = s.ffill(limit=3).fillna(0).values

    for n in range(N):
        s = pd.Series(target_arr[:, n])
        target_arr[:, n] = s.ffill(limit=3).fillna(0).values

    print(f"  NaNs after  fill — station: {np.isnan(station_arr).sum()}, "
          f"external: {np.isnan(external_arr).sum()}, "
          f"target: {np.isnan(target_arr).sum()}")

    return station_arr, external_arr, target_arr, stations


def normalize(train_arr, val_arr, test_arr):
    """Fit scaler on train, apply to all splits. Works on (T, N, F) arrays."""
    T_tr, N, F = train_arr.shape
    scaler = StandardScaler()
    train_2d  = train_arr.reshape(-1, F)
    scaler.fit(train_2d)
    def _transform(arr):
        sh = arr.shape
        return scaler.transform(arr.reshape(-1, F)).reshape(sh)
    return _transform(train_arr), _transform(val_arr), _transform(test_arr), scaler