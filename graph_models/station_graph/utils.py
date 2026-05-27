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

    def forward(self, x, return_weights=False):
        # x: [B, T, N, F]
        weights = torch.softmax(self.fc(x), dim=-1)
        if return_weights:
            return x * weights, weights
        return x * weights


class TemporalAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key   = nn.Linear(hidden_dim, hidden_dim)
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

        self.feature_att  = FeatureAttention(in_channels)
        self.temporal_att = TemporalAttention(in_channels)
        self.graph_conv   = ChebGraphConv(in_channels, hidden_dim, K)

        self.temporal_conv = nn.Conv2d(
            hidden_dim, hidden_dim,
            kernel_size=(3, 1), padding=(1, 0)
        )

        self.dropout = nn.Dropout(p=dropout)
        self.norm    = nn.LayerNorm(hidden_dim)

    def forward(self, x, laplacian, return_att=False):
        # x: [B, T, N, F]
        residual = x

        if return_att:
            x, feat_w = self.feature_att(x, return_weights=True)
        else:
            x = self.feature_att(x)

        x = self.temporal_att(x)
        x = self.graph_conv(x, laplacian)

        x = x.permute(0, 3, 1, 2)
        x = self.temporal_conv(x)
        x = self.dropout(x)
        x = x.permute(0, 2, 3, 1)

        out = self.norm(x + residual)

        if return_att:
            return out, feat_w
        return out


def compute_laplacian(adj):
    D          = torch.diag(torch.sum(adj, dim=1))
    L          = D - adj
    D_inv_sqrt = torch.diag(1.0 / torch.sqrt(torch.sum(adj, dim=1) + 1e-6))
    return D_inv_sqrt @ L @ D_inv_sqrt


def prepare_laplacian(station_list_path, device):
    adj        = torch.tensor(create_adj_matrix(station_list_path))
    laplacian  = compute_laplacian(adj).float().to(device)
    lambda_max = torch.linalg.eigvals(laplacian).real.max()
    laplacian  = (2 / lambda_max) * laplacian - torch.eye(laplacian.size(0), device=device)
    return laplacian


def filter_tensors(data_tensor: torch.Tensor, train_end, val_end, timestamps):
    if isinstance(timestamps, pd.DatetimeIndex):
        if timestamps.tz is not None:
            timestamps = timestamps.tz_convert(None)
    elif isinstance(timestamps, pd.Series):
        if timestamps.dt.tz is not None:
            timestamps = timestamps.dt.tz_convert(None)
    else:
        timestamps = pd.to_datetime(timestamps, utc=True)
        if hasattr(timestamps, 'tz') and timestamps.tz is not None:
            timestamps = timestamps.tz_convert(None)

    train_idx = timestamps < train_end
    val_idx   = (timestamps >= train_end) & (timestamps < val_end)
    test_idx  = timestamps >= val_end

    return data_tensor[train_idx], data_tensor[val_idx], data_tensor[test_idx]


def load_and_pivot(path: str, STATION_FEATURE_COLS, EXTERNAL_COLS, sim = False):
    """
    Load the parquet/csv file and return arrays for the MATGCN model.

    Returns
    -------
        station_arr  : np.ndarray (T, N, F)
        external_arr : np.ndarray (T, E)
        target_arr   : np.ndarray (T, N, 2)  — [dep_delay, arr_delay] in seconds
        stations     : list[str]
    """
    if not sim:
        df = pd.read_parquet(path)
        TARGET_COL = "DAILY_PLAN_OPERATIONAL_DELAY_SEC"
    else:
        if path.endswith(".csv"):
            df = pd.read_csv(path)
        else:
            df = pd.read_parquet(path)
        TARGET_COL = "SIMULATED_DELAY"


    STATION_COL = "OPERATING_POINT_ABBREVIATION"
    DATE_COL    = "OPERATION_PLANNED_TIMESTAMP"

    df["OPERATION_ACTUAL_TIMESTAMP"]  = pd.to_datetime(df["OPERATION_ACTUAL_TIMESTAMP"])
    df["OPERATION_PLANNED_TIMESTAMP"] = pd.to_datetime(df["OPERATION_PLANNED_TIMESTAMP"])

    df["hour_sin"] = np.sin(2 * np.pi * df["OPERATION_ACTUAL_TIMESTAMP"].dt.hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["OPERATION_ACTUAL_TIMESTAMP"].dt.hour / 24)
    df["dow_sin"]  = np.sin(2 * np.pi * df["OPERATION_ACTUAL_TIMESTAMP"].dt.dayofweek / 7)
    df["dow_cos"]  = np.cos(2 * np.pi * df["OPERATION_ACTUAL_TIMESTAMP"].dt.dayofweek / 7)
    df["hto000d0"] = df["hto000d0"].fillna(0)

    for drop_col in ["date", "days"]:
        if drop_col in df.columns:
            df = df.drop(columns=[drop_col])

    df = df.sort_values([DATE_COL, STATION_COL]).reset_index(drop=True)

    exclude_cols = ["OPERATION_ACTUAL_TIMESTAMP", TARGET_COL, DATE_COL]
    cat_cols = [
        col for col in df.select_dtypes(include="object").columns
        if col not in exclude_cols
    ]
    for col in cat_cols:
        df[col] = df[col].astype("category").cat.codes

    def _mode(s):
        m = s.mode()
        return m.iloc[0] if len(m) else np.nan

    cat_station_cols = [
        c for c in STATION_FEATURE_COLS
        if df[c].dtype in (object, "category")
        or str(df[c].dtype) == "int8"
        or df[c].nunique() < 20
    ]
    num_station_cols = [c for c in STATION_FEATURE_COLS if c not in cat_station_cols]

    agg_dict_feat = {c: _mode  for c in cat_station_cols}
    agg_dict_feat.update({c: "mean"  for c in num_station_cols})
    agg_dict_feat.update({c: "first" for c in EXTERNAL_COLS})

    grp = df.groupby([DATE_COL, STATION_COL])

    # Alphabetical encoding: 'arrival'=0, 'departure'=1
    dep_target = (df[df["EVENT_TYPE"] == 1]
                    .groupby([DATE_COL, STATION_COL])[TARGET_COL]
                    .mean().rename("target_dep"))
    arr_target = (df[df["EVENT_TYPE"] == 0]
                    .groupby([DATE_COL, STATION_COL])[TARGET_COL]
                    .mean().rename("target_arr"))

    feat_agg = grp.agg(agg_dict_feat)
    combined = feat_agg.join(dep_target).join(arr_target).reset_index()

    stations   = sorted(combined[STATION_COL].unique())
    timestamps = sorted(combined[DATE_COL].unique())
    N = len(stations)
    T = len(timestamps)
    F = len(STATION_FEATURE_COLS)
    E = len(EXTERNAL_COLS) if EXTERNAL_COLS else 1
    print(f"  Timesteps : {T},  Stations : {N}")

    station_arr  = np.full((T, N, F), np.nan, dtype=np.float32)
    external_arr = np.full((T, E),    np.nan, dtype=np.float32)
    target_arr   = np.full((T, N, 2), np.nan, dtype=np.float32)

    station_idx = {s: i for i, s in enumerate(stations)}
    time_idx    = {t: i for i, t in enumerate(timestamps)}

    for _, row in combined.iterrows():
        t = time_idx[row[DATE_COL]]
        n = station_idx[row[STATION_COL]]
        station_arr[t, n, :] = [
            row[c] if pd.notna(row.get(c)) else np.nan
            for c in STATION_FEATURE_COLS
        ]
        target_arr[t, n, 0] = row["target_dep"] if pd.notna(row.get("target_dep")) else np.nan
        target_arr[t, n, 1] = row["target_arr"] if pd.notna(row.get("target_arr")) else np.nan
        if EXTERNAL_COLS and np.isnan(external_arr[t, 0]):
            external_arr[t, :] = [
                row[c] if pd.notna(row.get(c)) else 0.0
                for c in EXTERNAL_COLS
            ]

    print(f"  Station features ({F}): {STATION_FEATURE_COLS}")
    print(f"  External features ({E}): {EXTERNAL_COLS}")
    print(f"  NaNs before fill — station: {np.isnan(station_arr).sum()}, "
          f"external: {np.isnan(external_arr).sum()}, "
          f"target: {np.isnan(target_arr).sum()}")

    for e in range(E):
        s = pd.Series(external_arr[:, e])
        external_arr[:, e] = s.ffill().fillna(0).values

    for n in range(N):
        for f in range(F):
            s = pd.Series(station_arr[:, n, f])
            station_arr[:, n, f] = s.ffill(limit=3).fillna(0).values

    for n in range(N):
        for c in range(2):
            s = pd.Series(target_arr[:, n, c])
            target_arr[:, n, c] = s.ffill(limit=3).fillna(0).values

    print(f"  NaNs after  fill — station: {np.isnan(station_arr).sum()}, "
          f"external: {np.isnan(external_arr).sum()}, "
          f"target: {np.isnan(target_arr).sum()}")

    return station_arr, external_arr, target_arr, stations


def normalize(train_arr, val_arr, test_arr):
    """Fit StandardScaler on train, apply to all splits. Works on (T, N, F) arrays."""
    T_tr, N, F = train_arr.shape
    scaler      = StandardScaler()
    scaler.fit(train_arr.reshape(-1, F))

    def _transform(arr):
        sh = arr.shape
        return scaler.transform(arr.reshape(-1, F)).reshape(sh)

    return _transform(train_arr), _transform(val_arr), _transform(test_arr), scaler


def normalize_targets(tr_tg, va_tg, te_tg):
    """
    Fit a StandardScaler on the log-transformed training targets (T, N, 2)
    and return normalised splits plus the scaler for inversion at eval time.

    Why a separate scaler?  The station-feature scaler operates on (T*N, F)
    and should not absorb target variance.  Normalising the targets puts the
    two lagged-target channels (appended by DelayDataset) on the same scale
    as every other z-scored feature, making permutation importance fair.

    Call AFTER log-transforming the targets (to_log).
    """
    T, N, C = tr_tg.shape        # C == 2  (dep, arr)
    scaler   = StandardScaler()
    scaler.fit(tr_tg.reshape(-1, C))

    def _t(a):
        sh = a.shape
        return scaler.transform(a.reshape(-1, C)).reshape(sh)

    return _t(tr_tg), _t(va_tg), _t(te_tg), scaler


# ------------------------------------------------------------------------------
# Permutation importance
# ------------------------------------------------------------------------------

def _collect_all_batches(loader):
    """Materialise the full DataLoader into a single tensor triple (x, ext, y)."""
    xs, exts, ys = [], [], []
    for x, ext, y in loader:
        xs.append(x); exts.append(ext); ys.append(y)
    return torch.cat(xs), torch.cat(exts), torch.cat(ys)


def permute_station_feature(x_all, feature_idx):
    """
    Globally permute one feature channel across ALL samples in the dataset.

    This is the critical fix for lagged-target channels: consecutive sliding
    windows share almost identical lag values, so shuffling within a 32-sample
    batch causes near-zero perturbation and makes the feature look unimportant.
    A global shuffle across all T windows produces a meaningful signal.
    """
    x_perm = x_all.clone()
    perm = torch.randperm(x_all.size(0))
    x_perm[..., feature_idx] = x_all[perm, ..., feature_idx]
    return x_perm


def permute_external_feature(ext_all, feature_idx):
    ext_perm = ext_all.clone()
    perm = torch.randperm(ext_all.size(0))
    ext_perm[..., feature_idx] = ext_all[perm, ..., feature_idx]
    return ext_perm


def compute_mae_seconds(model, x_all, ext_all, y_all, laplacian, tg_min,
                        target_scaler=None, batch_size=256):
    """
    Run inference over the full dataset in mini-batches and return MAE in seconds.

    Parameters
    ----------
    x_all, ext_all, y_all : full-dataset CPU tensors
    laplacian             : Laplacian (CPU)
    tg_min                : scalar shift used in to_log()
    target_scaler         : StandardScaler fitted on log targets, or None
    batch_size            : inference chunk size (tune to fit RAM/VRAM)
    """
    model.eval()
    all_pred, all_true = [], []

    with torch.no_grad():
        for start in range(0, x_all.size(0), batch_size):
            xb   = x_all  [start:start + batch_size]
            eb   = ext_all[start:start + batch_size]
            yb   = y_all  [start:start + batch_size]
            pred = model(xb, eb, laplacian, return_att=False).squeeze(2)  # (B, N, 2)
            all_pred.append(pred.numpy())
            all_true.append(yb.numpy())

    preds = np.concatenate(all_pred)   # (total, N, 2)
    trues = np.concatenate(all_true)

    # Invert target z-score if targets were normalised after log-transform
    if target_scaler is not None:
        sh = preds.shape
        preds = target_scaler.inverse_transform(preds.reshape(-1, 2)).reshape(sh)
        trues = target_scaler.inverse_transform(trues.reshape(-1, 2)).reshape(sh)

    pred_sec = np.expm1(preds) + tg_min
    y_sec    = np.expm1(trues) + tg_min

    return float(np.abs(pred_sec - y_sec).mean())


def permutation_importance(
    model,
    loader,
    laplacian,
    station_feature_names,
    external_feature_names,
    tg_min,
    target_scaler=None,
    n_repeats=3,
):
    """
    Global permutation importance with multi-repeat averaging.

    Parameters
    ----------
    station_feature_names : the ORIGINAL F feature names (no lagged channels).
                            The two lagged channels appended by DelayDataset sit
                            at indices F and F+1 and are always reported as
                            'lagged_dep' / 'lagged_arr'.
    n_repeats             : number of independent permutations to average.
                            3 is usually enough; increase for noisier datasets.
    target_scaler         : pass the scaler returned by normalize_targets() if
                            you z-scored the targets after log-transform.
    """
    model.eval()
    laplacian = laplacian.cpu()

    # Materialise the full split once — avoids the per-batch permutation trap
    x_all, ext_all, y_all = _collect_all_batches(loader)

    lagged_names   = ["lagged_dep", "lagged_arr"]
    all_feat_names = list(station_feature_names) + lagged_names

    baseline = compute_mae_seconds(
        model, x_all, ext_all, y_all, laplacian, tg_min,
        target_scaler=target_scaler
    )
    print(f"Baseline MAE (seconds): {baseline:.2f}\n")

    importances = {}

    # ---- station features + lagged channels ----
    for i, name in enumerate(all_feat_names):
        deltas = []
        for _ in range(n_repeats):
            x_perm = permute_station_feature(x_all, i)
            deltas.append(
                compute_mae_seconds(model, x_perm, ext_all, y_all, laplacian,
                                    tg_min, target_scaler=target_scaler)
                - baseline
            )
        delta = float(np.mean(deltas))
        importances[name] = delta
        print(f"  {name:45s}  delta={delta:+.2f} s")

    # ---- external features ----
    for i, name in enumerate(external_feature_names):
        deltas = []
        for _ in range(n_repeats):
            ext_perm = permute_external_feature(ext_all, i)
            deltas.append(
                compute_mae_seconds(model, x_all, ext_perm, y_all, laplacian,
                                    tg_min, target_scaler=target_scaler)
                - baseline
            )
        delta = float(np.mean(deltas))
        importances[name] = delta
        print(f"  {name:45s}  delta={delta:+.2f} s")

    return importances