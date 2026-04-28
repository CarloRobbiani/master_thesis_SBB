import xgboost as xgb
import numpy as np
import pandas as pd
import shap

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
            
            """ for fname, val in score.items():
                idx = feature_names.index(fname)
                imp[idx] = val """

            all_importances.append(imp)

        all_importances = np.array(all_importances)

        all_importances = all_importances / all_importances.sum(axis=1, keepdims=True)
        final_importance = all_importances.mean(axis=0)

        """ if aggregate == "mean":
            final_importance = all_importances.mean(axis=0)
        elif aggregate == "sum":
            final_importance = all_importances.sum(axis=0) """

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
