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
    Call predict_entry_delays() once before RailwaySimulator.run() to get
    a {train_number: delay_sec} dict that seeds each TrainProcess.
    """

    def __init__(
        self,
        model_path:   str,
        scaler_path:  str,
        stats_path:   str,
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
        from the last SEQ_LEN timesteps of the day's raw data — exactly
        replicating the load_and_pivot logic from training.
        """
        import tempfile, os

        # load_and_pivot expects a file path; write a temp parquet
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

        # Normalize only the original station features (what the scaler was fitted on)
        station_arr_norm = self.scaler.transform(
            station_arr.reshape(-1, F_orig)
        ).reshape(T, N, F_orig)

        # Append lagged targets AFTER scaling (they weren't included in scaler fit)
        lagged_targets = np.zeros((T, N, 2), dtype=np.float32)
        if T > 1:
            lagged_targets[1:] = target_arr[:-1]

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
        Station order matches self.station_order
        """
        hist, ext = self._build_history_buffer(df_day)

        x   = torch.tensor(hist).unsqueeze(0).to(self.device)          # (1, SEQ_LEN, N, F)
        ext = torch.tensor(ext).unsqueeze(0).to(self.device)           # (1, E)
        ext = ext.unsqueeze(1).expand(-1, x.shape[1], -1)              # (1, SEQ_LEN, E)

        pred_log = self.model(x, ext, self.laplacian).squeeze(0).cpu().numpy()  # (N, horizon, 2)
        pred_sec = np.expm1(pred_log) + self.tg_min
        return pred_sec

    def predict_entry_delays(
        self,
        df_day:   pd.DataFrame,
        timetable,              
    ) -> dict[int, float]:
        """
        Returns {train_number: predicted_entry_delay_sec} for every train
        in the timetable.  Entry station is the first stop in the schedule
        (BI for BI→NE trains, NE for NE→BI trains).
        Uses the GCN's predicted departure delay at that station.
        """
        pred_sec = self.predict_all_stations(df_day)  # (N, 2)

        station_to_idx = {s: i for i, s in enumerate(self.station_order)}
        entry_delays   = {}

        for schedule in timetable.schedules:
            entry_station = schedule.stops[0].station
            idx = station_to_idx.get(entry_station)
            if idx is None:
                entry_delays[schedule.train_number] = 0.0
                continue
            # col 0 = departure delay — what we want at the origin
            predicted = float(pred_sec[idx,0, 0])
            # Clip negatives: we don't want to start trains early
            entry_delays[schedule.train_number] = max(0.0, predicted)

        return entry_delays