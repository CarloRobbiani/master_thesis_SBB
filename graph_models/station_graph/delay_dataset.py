import torch
from torch.utils.data import Dataset

class DelayDataset(Dataset):
    """
    Sliding-window dataset.
 
    Each sample:
      x        : (SEQ_LEN, N, F)   – station features window
      external : (SEQ_LEN, E)      – external features window
      y        : (N, HORIZON)      – target delay for next HORIZON steps
    """
    def __init__(self, station_arr, external_arr, target_arr, seq_len, horizon):
        self.station  = torch.tensor(station_arr,  dtype=torch.float32)
        self.external = torch.tensor(external_arr, dtype=torch.float32)
        self.target   = torch.tensor(target_arr,   dtype=torch.float32)
        self.seq_len  = seq_len
        self.horizon  = horizon
        # valid start indices
        self.indices  = range(seq_len, len(station_arr) - horizon + 1)
 
    def __len__(self):
        return len(self.indices)
 
    def __getitem__(self, idx):
        i   = self.indices[idx]
        x   = self.station [i - self.seq_len : i]          # (L, N, F)
        ext = self.external[i - self.seq_len : i]           # (L, E)
        # target: next `horizon` steps, averaged across horizon dimension
        y   = self.target  [i : i + self.horizon]           # (H, N)
        y   = y.mean(dim=0)                                  # (N,)  – mean over horizon
        return x, ext, y