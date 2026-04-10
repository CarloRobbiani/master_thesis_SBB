import xgboost as xgb
import numpy as np
import pandas as pd
import shap

class XGBoostBaseline:
    def __init__(self, params=None):
        self.params = params or {
            "objective": "reg:squarederror",
            "n_estimators": 500,
            "max_depth": 10,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8
        }
        self.models = []

    def fit(self, X, Y):
        # Accept DataFrame or numpy array
        if isinstance(X, pd.DataFrame):
            X_flat = X.values
            n_samples = X.shape[0]
            n_nodes = 1
        else:
            n_samples = X.shape[0]
            n_nodes = X.shape[1] if len(X.shape) > 1 else 1
            X_flat = X.reshape(n_samples * n_nodes, -1) if len(X.shape) > 2 else X

        if isinstance(Y, (pd.Series, pd.DataFrame)):
            Y_flat = Y.values
            if Y_flat.ndim == 1:
                Y_flat = Y_flat[:, None]
        else:
            Y_flat = Y.reshape(Y.shape[0] * Y.shape[1], -1) if len(Y.shape) > 1 else Y[:, None]

        horizon = Y_flat.shape[1]
        self.models = []

        for h in range(horizon):
            model = xgb.XGBRegressor(**self.params)
            model.fit(X_flat, Y_flat[:, h])
            self.models.append(model)

        return self

    def predict(self, X):
        if isinstance(X, pd.DataFrame):
            X_flat = X.values
            n_samples = X.shape[0]
            n_nodes = 1
        else:
            n_samples = X.shape[0]
            n_nodes = X.shape[1] if len(X.shape) > 1 else 1
            X_flat = X.reshape(n_samples * n_nodes, -1) if len(X.shape) > 2 else X

        preds = []
        for model in self.models:
            preds.append(model.predict(X_flat))

        Y_pred = np.stack(preds, axis=1)
        # Reshape to (samples, nodes, horizon) if possible
        if n_nodes > 1:
            return Y_pred.reshape(n_samples, n_nodes, -1)
        else:
            return Y_pred
        

    def feature_importance(self, feature_names=None, importance_type="gain", aggregate="mean"):
        """
        Extract feature importance for each horizon model.

        Parameters:
            feature_names : list of str
            importance_type : 'gain', 'weight', 'cover'
            aggregate : 'mean' or 'sum'

        Returns:
            pd.DataFrame with importance per feature
        """

        all_importances = []

        for model in self.models:
            booster = model.get_booster()
            score = booster.get_score(importance_type=importance_type)

            # Convert to full vector
            if feature_names is None:
                n_features = model.n_features_in_
                feature_names = [f"f{i}" for i in range(n_features)]

            imp = np.zeros(len(feature_names))
            for i, fname in enumerate(feature_names):
                key = f"f{i}"
                imp[i] = score.get(key, 0.0)

            all_importances.append(imp)

        all_importances = np.array(all_importances)

        if aggregate == "mean":
            final_importance = all_importances.mean(axis=0)
        elif aggregate == "sum":
            final_importance = all_importances.sum(axis=0)

        df = pd.DataFrame({
            "feature": feature_names,
            "importance": final_importance
        }).sort_values("importance", ascending=False)

        return df
    
    def shap_importance(self, X_sample):
        shap_values_all = []

        for model in self.models:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sample)
            shap_values_all.append(np.abs(shap_values))

        shap_values_all = np.array(shap_values_all)

        # Mean across horizons and samples
        mean_shap = shap_values_all.mean(axis=(0,1))

        return mean_shap
