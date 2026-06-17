# gcn_predictor.py
import torch
import numpy as np
import pickle, json
import pandas as pd
import os
import sys
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from graph_models.station_graph.stationMATGCN import StationMATGCN
from graph_models.station_graph.utils import prepare_laplacian, load_and_pivot
from sim_topology import LINE_ORDER

STATION_FEATURE_COLS = [
    "EVENT_TYPE", "EVENT_SERVED", "PLAN_STOP_TYPE",
    "OPERATION_DAY_PERIOD_IDENTIFIER_COARSE",
    "OPERATION_TRAFFIC_CATEGORY_ABBREVIATION",
    "PLAN_FORMATION_MAXIMAL_VELOCITY",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
]
EXTERNAL_COLS = ["tre200s0", "fkl010z1", "fu3010z0", "rre150z0", "htoauts0", "hto000d0"]
SEQ_LEN = 28


class GCNPredictor:
    """
    Wraps a trained StationMATGCN for inference inside the simulator.
    """

    def __init__(
        self,
        model_path: str,
        scaler_path:  str,
        stats_path: str,
        target_scaler_path: str,
        station_list_path: str,
        hidden_dim=64, K=3, num_blocks=3, horizon=1,
        num_station_features=12, num_external_features=6,
        device: str = "cpu",
    ):
        self.device  = torch.device(device)
        self.seq_len = SEQ_LEN

        with open(scaler_path, "rb") as f:
            self.scaler = pickle.load(f)
        with open(stats_path) as f:
            self.tg_min = json.load(f)["tg_min"]

        with open(target_scaler_path, "rb") as f:
            self.target_scaler = pickle.load(f)

        self.laplacian = prepare_laplacian(station_list_path, self.device)
        self.station_order = LINE_ORDER  # must match training order

        self.model = StationMATGCN(
            num_station_features  = num_station_features,
            num_external_features = num_external_features,
            hidden_dim = hidden_dim,
            K          = K,
            num_blocks = num_blocks,
            horizon    = horizon,
        ).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

    def _build_history_buffer(self, df_day: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """
        Build the (SEQ_LEN, N, F) station buffer and (E,) external snapshot
        from the last SEQ_LEN timesteps of the day's raw data
        """
        import tempfile, os

        # load_and_pivot expects a file path write a temp parquet
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            tmp = f.name
        try:
            df_day.to_parquet(tmp)
            station_arr, external_arr, target_arr, stations = load_and_pivot(
                tmp, STATION_FEATURE_COLS, EXTERNAL_COLS
            )
        finally:
            os.unlink(tmp)

        # Normalize station features with the training scaler
        T, N, F_orig = station_arr.shape 

        # Normalize only the original station features
        station_arr_norm = self.scaler.transform(
            station_arr.reshape(-1, F_orig)
        ).reshape(T, N, F_orig)

        # Append lagged targets after scaling 
        lagged_targets = np.zeros((T, N, 2), dtype=np.float32)
        if T > 1:
            lagged_raw = target_arr[:-1]  # (T-1, N, 2)
            lagged_log = np.log1p(np.clip(lagged_raw - self.tg_min, 0, None))
            sh = lagged_log.shape
            lagged_norm = self.target_scaler.transform(
                lagged_log.reshape(-1, 2)
            ).reshape(sh)
            lagged_targets[1:] = lagged_norm

        # Final shape: (T, N, 12)
        station_arr_full = np.concatenate([station_arr_norm, lagged_targets], axis=-1)

        # Take last SEQ_LEN timesteps
        T, N, F_full = station_arr_full.shape
        if T >= self.seq_len:
            hist = station_arr_full[-self.seq_len:]
            ext  = external_arr[-1]
        else:
            pad  = np.zeros((self.seq_len - T, N, F_full), dtype=np.float32)
            hist = np.concatenate([pad, station_arr_full], axis=0)
            ext  = external_arr[-1] if T > 0 else np.zeros(len(EXTERNAL_COLS))

        return hist.astype(np.float32), ext.astype(np.float32)

    @torch.no_grad()
    def predict_all_stations(self, df_day: pd.DataFrame) -> np.ndarray:
        """
        Run one forward pass. Returns predicted delays in seconds,
        shape (N, 2): col 0 = departure delay, col 1 = arrival delay.
        """
        hist, ext = self._build_history_buffer(df_day)

        x   = torch.tensor(hist).unsqueeze(0).to(self.device)          # (1, SEQ_LEN, N, F)
        ext = torch.tensor(ext).unsqueeze(0).to(self.device)           # (1, E)
        ext = ext.unsqueeze(1).expand(-1, x.shape[1], -1)              # (1, SEQ_LEN, E)

        pred_norm = self.model(x, ext, self.laplacian).squeeze(0).cpu().numpy()  # (N, horizon, 2)
        sh = pred_norm.shape
        pred_log = self.target_scaler.inverse_transform(
            pred_norm.reshape(-1, 2)
        ).reshape(sh)
        pred_sec = np.expm1(pred_log) + self.tg_min
        return pred_sec

    def predict_station_priors(
        self,
        df_day: pd.DataFrame,
    ) -> dict[str, float]:
        """
        Returns {station_abbr: predicted_departure_delay_sec} for every station in LINE_ORDER.
        """
        pred_sec = self.predict_all_stations(df_day)   # (N, horizon, 2)
        station_to_idx = {s: i for i, s in enumerate(self.station_order)}
        priors: dict[str, float] = {}

        for station, idx in station_to_idx.items():
            # col 0 = departure delay
            priors[station] = float(pred_sec[idx, 0, 0])
        return priors
