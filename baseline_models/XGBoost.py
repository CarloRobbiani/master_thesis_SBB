import xgboost as xgb
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import mean_absolute_error
import numpy as np

class XGBoostBaseline:
    def __init__(self, params=None):
        self.params = params or {
            "objective": "reg:squarederror",
            "n_estimators": 500,
            "max_depth": 15,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8
        }
        self.models = []
        

    def fit(self, X, Y,  X_val=None, Y_val=None):
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


        # ---- Validation handling ----
        if X_val is not None and Y_val is not None:
            if isinstance(X_val, pd.DataFrame):
                X_val_flat = X_val.values
            else:
                X_val_flat = X_val.reshape(X_val.shape[0] * n_nodes, -1) if len(X_val.shape) > 2 else X_val

            if isinstance(Y_val, (pd.Series, pd.DataFrame)):
                Y_val_flat = Y_val.values
                if Y_val_flat.ndim == 1:
                    Y_val_flat = Y_val_flat[:, None]
            else:
                Y_val_flat = Y_val.reshape(Y_val.shape[0] * Y_val.shape[1], -1) if len(Y_val.shape) > 1 else Y_val[:, None]
        else:
            X_val_flat, Y_val_flat = None, None

        horizon = Y_flat.shape[1]
        self.models = []
        self.eval_results = []

        for h in range(horizon):
            model = xgb.XGBRegressor(**self.params)
            if X_val_flat is not None:
                eval_set = [(X_flat, Y_flat[:, h]), (X_val_flat, Y_val_flat[:, h])]
            else:
                eval_set = [(X_flat, Y_flat[:, h])]

            model.fit(X_flat, Y_flat[:, h], eval_set=eval_set)
            self.models.append(model)
            self.eval_results.append(model.evals_result())

        return self
    
    

    def plot_loss(self, horizon_step=0):
        import matplotlib.pyplot as plt
        results = self.eval_results[horizon_step]

        train_loss = results['validation_0']['rmse']

        plt.figure()
        plt.plot(train_loss, label='Train Loss')

        if 'validation_1' in results:
            val_loss = results['validation_1']['rmse']
            plt.plot(val_loss, label='Validation Loss')

        plt.xlabel('Iterations')
        plt.ylabel('RMSE')
        plt.title(f'Loss Curve (Horizon step {horizon_step})')
        plt.legend()
        plt.grid()
        plt.savefig("images\RMSE_curves_XGboost.png")
        plt.show()

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
        

    def permutation_importance(self, X_sample, Y_sample, n_repeats=5, metric="mae"):

        if isinstance(X_sample, pd.DataFrame):
            X_arr = X_sample.values
            feature_names = X_sample.columns.tolist()
        else:
            X_arr = X_sample
            feature_names = [f"f{i}" for i in range(X_arr.shape[1])]

        baseline_pred = self.predict(X_sample)
        baseline_score = mean_absolute_error(Y_sample, baseline_pred)

        importances = np.zeros((len(feature_names), n_repeats))

        for i in range(len(feature_names)):
            for r in range(n_repeats):
                X_permuted = X_arr.copy()
                X_permuted[:, i] = np.random.permutation(X_permuted[:, i])

                if isinstance(X_sample, pd.DataFrame):
                    X_perm_df = pd.DataFrame(X_permuted, columns=feature_names)
                    perm_pred = self.predict(X_perm_df)
                else:
                    perm_pred = self.predict(X_permuted)

                perm_score = mean_absolute_error(Y_sample, perm_pred)
                importances[i, r] = perm_score - baseline_score  # higher = more important

        mean_imp = importances.mean(axis=1)
        std_imp = importances.std(axis=1)

        return pd.DataFrame({
            "feature": feature_names,
            "importance": mean_imp,
            "std": std_imp
        }).sort_values("importance", ascending=False)
    
    def shap_importance(self, X_sample):
        shap_values_all = []

        for model in self.models:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sample)
            shap_values_all.append(np.abs(shap_values))

        shap_values_all = np.array(shap_values_all)

        # Mean across horizons and samples
        mean_shap = shap_values_all.mean(axis=(0,1))

        return pd.DataFrame({
            "feature": X_sample.columns,
            "importance": mean_shap
        }).sort_values("importance", ascending=False)

        #return mean_shap

    def shap_dependence(self, X_sample, feature):
        """
        Returns feature values and corresponding SHAP contributions
        across all models (horizons).
        contribution means how much delay it adds/removes
        """

        feature_idx = list(X_sample.columns).index(feature)

        values_all = []
        shap_all = []

        for model in self.models:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sample)

            if isinstance(shap_values, list):
                shap_values = np.array(shap_values).mean(axis=0)

            values_all.append(X_sample.iloc[:, feature_idx].values)
            shap_all.append(shap_values[:, feature_idx])

        values_all = np.concatenate(values_all)
        shap_all = np.concatenate(shap_all)

        return pd.DataFrame({
            "feature_value": values_all,
            "shap_value": shap_all
        })
    
def summarize_dependence(dep_df, n_bins=20):
    dep_df["bin"] = pd.qcut(dep_df["feature_value"], q=n_bins, duplicates="drop")

    summary = dep_df.groupby("bin", observed=False).agg({
        "feature_value": "mean",
        "shap_value": "mean"
    }).reset_index(drop=True)

    return summary
