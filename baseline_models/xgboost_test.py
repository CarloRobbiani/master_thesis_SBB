import numpy as np

class PersistenceBaseline:
    def fit(self, X, Y):
        return self

    def predict(self, X):
        # predict future delay = last observed delay feature
        # assumes delay is feature index 0
        last_delay = X[:, :, -1, 0]  # (samples, nodes)
        horizon = 12
        return np.repeat(last_delay[:, :, None], horizon, axis=2)
    

from collections import defaultdict

class HistoricalMeanBaseline:
    def __init__(self):
        self.means = defaultdict(lambda: 0.0)

    def fit(self, X, Y, time_ids):
        # time_ids length = num_samples
        sums = defaultdict(float)
        counts = defaultdict(int)

        for i, tid in enumerate(time_ids):
            sums[tid] += Y[i].mean()
            counts[tid] += 1

        for tid in sums:
            self.means[tid] = sums[tid] / counts[tid]

        return self

    def predict(self, X, time_ids, horizon=12):
        preds = np.zeros((X.shape[0], X.shape[1], horizon))
        for i, tid in enumerate(time_ids):
            preds[i, :, :] = self.means[tid]
        return preds
    
import xgboost as xgb

class XGBoostBaseline:
    def __init__(self, params=None):
        self.params = params or {
            "objective": "reg:squarederror",
            "n_estimators": 200,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8
        }
        self.models = []

    def fit(self, X, Y):
        X_flat = X.reshape(X.shape[0] * X.shape[1], -1)
        Y_flat = Y.reshape(Y.shape[0] * Y.shape[1], -1)

        horizon = Y_flat.shape[1]
        self.models = []

        for h in range(horizon):
            model = xgb.XGBRegressor(**self.params)
            model.fit(X_flat, Y_flat[:, h])
            self.models.append(model)

        return self

    def predict(self, X):
        X_flat = X.reshape(X.shape[0] * X.shape[1], -1)

        preds = []
        for model in self.models:
            preds.append(model.predict(X_flat))

        Y_pred = np.stack(preds, axis=1)
        return Y_pred.reshape(X.shape[0], X.shape[1], -1)
    
import torch
import torch.nn as nn

class LSTMForecast(nn.Module):
    def __init__(self, num_features, hidden_dim=64, horizon=12):
        super().__init__()
        self.lstm = nn.LSTM(input_size=num_features, hidden_size=hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, horizon)

    def forward(self, x):
        # x: (batch, seq_len, num_features)
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.fc(last)


def train_lstm(model, train_loader, val_loader, epochs=20, lr=1e-3, device="cpu"):
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        train_loss = 0

        for X, Y in train_loader:
            # X: (batch, nodes, seq_len, features)
            # Y: (batch, nodes, horizon)

            X = X.to(device)
            Y = Y.to(device)

            batch, nodes, seq_len, feats = X.shape

            X = X.reshape(batch * nodes, seq_len, feats)
            Y = Y.reshape(batch * nodes, -1)

            pred = model(X)
            loss = loss_fn(pred, Y)

            opt.zero_grad()
            loss.backward()
            opt.step()

            train_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs} Train Loss={train_loss/len(train_loader):.4f}")



