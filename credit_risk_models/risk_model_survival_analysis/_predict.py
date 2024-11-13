import uuid
from pathlib import Path
from tqdm import tqdm
import numpy as np
import pandas as pd
import pickle
from dataclasses import dataclass
from scipy.interpolate import interp1d
from sqlalchemy.types import Double, Uuid, String, DateTime
from lime.lime_tabular import LimeTabularExplainer

from . import _make_dataset
from . import _utils
from . import db
from credit_risk_models.azure_credentials_keyvault.ml_client import (
    get_ml_client,
)


@dataclass
class PredictTask:
    model_name : str
    model_version : str
    prediction_table_name : str
    feat_imps_table_name : str
    dpd_limit : int = 240

    def run(self):
        """Fetch a model from a cloud registry, run prediction and store \
        them on warehouse.
        """
        # We generate the predictions and push them on the warehouse.
        self.ds = _make_dataset.DatasetMaker(is_training=False)
        df = self.ds.dataset

        print(f"Number of on-going loans to be predicted: {df.shape[0]}")
        print(df.info())

        self.model_dict = _load_model(self.model_name, self.model_version)
        model = self.model_dict["model"]

        label_cols = ["event", "duration"]
        id_cols = ["carloan_id", "borrower_id"]
        X = df.drop(columns=label_cols + id_cols)
        
        vectorizer, estimator = model[0], model[-1]
        X_trans = vectorizer.transform(X)

        y_proba = estimator.predict_cumulative_incidence(X_trans)  # (n_samples, n_events, n_time_steps)

        # TODO: use bank provided termination limit of hardcoding it
        self.termination_limit = 150 
        horizon = self.termination_limit - X_trans["loan_age_days"]
        
        indices = np.searchsorted(estimator.time_grid_, horizon)
        y_proba_t = y_proba[np.arange(y_proba.shape[0]), :, indices]  # (n_samples, n_events)
        default_proba_t = y_proba_t[:, 0] + y_proba_t[:, 2]  # (n_samples)

        preds = X_trans.copy()
        preds["loan_id"] = df["carloan_id"]
        preds["prediction_id"] = [uuid.uuid4() for _ in range(preds.shape[0])]
        preds["batch_id"] = uuid.uuid4()
        preds["default_probability"] = default_proba_t
        preds["model_name"] = self.model_dict["model_name"]
        preds["model_version"] = self.model_dict["model_version"]
        preds["date"] = pd.Timestamp.now().strftime(_utils.UTC_DATETIME_FORMAT)

        prediction_sql_dtype = {
            "prediction_id": Uuid(),
            "batch_id": Uuid(),
            "model_name": String(),
            "model_version": String(),
            "loan_id": String(),
            "default_probability": Double(),
            "date": DateTime(),
        }
        _write_table(preds, prediction_sql_dtype, self.prediction_table_name)

        X_trans["prediction_id"] = preds["prediction_id"]
        feat_imps = self._get_feat_imps(X_trans, estimator, y_proba_t, horizon)
        feat_imps["feat_imp_id"] = [uuid.uuid4() for _ in range(feat_imps.shape[0])]
        feat_imps.sort_values("prediction_id", inplace=True)
        feat_mapping_sql_dtype = {
            "feat_imp_id": Uuid(),
            "prediction_id": Uuid(),
            "name": String(),
            "value": String(),
            "contribution": Double(),
        }
        _write_table(feat_imps, feat_mapping_sql_dtype, self.feat_imps_table_name)
    
    def _get_feat_imps(self, X_trans, estimator, y_proba_t, horizon):

        estimator.set_params(show_progressbar=False)

        preds = X_trans.melt(
            id_vars="prediction_id", var_name="name",
        ).sort_values("prediction_id")

        prediction_ids = X_trans.pop("prediction_id")

        # TODO: Does it needs to use X_train instead? To investigate.
        explainer = LimeTabularExplainer(
            training_data=X_trans.values,
            feature_names=X_trans.columns.tolist(),
            mode="classification",
            discretize_continuous=False,
        )

        # We only care about feature importance for the default classes.
        label_indices = np.argmax(y_proba_t[:, [0, 2]], axis=1) * 2

        print("Getting feature importance from Lime")
        results = []
        for idx in tqdm(range(X_trans.shape[0])):
            
            # Has to be set for predict proba
            estimator.set_params(time_horizon=horizon.iloc[idx])

            exp = explainer.explain_instance(
                X_trans.values[idx, :],
                estimator.predict_proba,
                num_features=X_trans.shape[1],
                labels=(0, 1, 2),
            )

            label = label_indices[idx]
            feats_contrib = exp.as_list(label=label)

            for (feat_name, contrib) in feats_contrib:
                results.append(
                    dict(
                        prediction_id=prediction_ids.iloc[idx],
                        name=feat_name,
                        contribution=contrib,
                    )
                )

        results = pd.DataFrame(results)
        preds = preds.merge(results, on=["prediction_id", "name"], how="left")

        return preds


def _load_model(model_name, model_version):
    ml_client = get_ml_client()

    download_path = Path(".") / "downloaded_model"
    model_info = ml_client.models.get(
        name=model_name,
        version=model_version,
    )
    ml_client.models.download(
        name=model_name,
        version=model_version,
        download_path=download_path,
    )
    artifact_dir_path = Path(download_path) / model_name

    # Get the last training run
    run_path = sorted(artifact_dir_path.glob("training_run_*"))[-1]

    model_path = run_path / "model.pkl"
    model = pickle.load(open(model_path, "rb"))

    return dict(
        model=model,
        model_name=model_info.name,
        model_version=model_info.version,
    )


def _write_table(df, sql_dtype, table_name):
    df = df[list(sql_dtype)]

    return db.DBSourceRisk().write_df(
        df,
        table_name=table_name,
        schema="risks",
        dtype=sql_dtype,
    )
