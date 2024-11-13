import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.base import check_is_fitted

FOLDER_DATETIME_FORMAT = "%Y-%m-%d_%H_%M_%S"
UTC_DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"


class CumulativeIncidencePipeline(Pipeline):
    def predict_cumulative_incidence(self, X, times=None):
        Xt = X
        for _, _, transformer in self._iter(with_final=False):
            Xt = transformer.transform(Xt)
        return self.steps[-1][1].predict_cumulative_incidence(Xt, times)
    
    @property
    def time_grid(self):
        model = self.steps[-1][1]
        check_is_fitted(model, "time_grid_")
        return model.time_grid_


def make_recarray(y):
    # This is an annoying trick to make scikit-survival happy.
    event = y["event"].values
    duration = y["duration"].values
    return np.array(
        [(event[i], duration[i]) for i in range(y.shape[0])],
        dtype=[("e", bool), ("t", float)],
    )


def get_n_events(event):
    return len(set(event.unique()) - {0})


def check_not_null(df, cols):
    for col in cols:
        n_nulls = df[col].isna().sum()
        if n_nulls > 0:
            raise ValueError(f"Column {col} has {n_nulls} null values.")


def check_is_dataframe(df, name):
    if not hasattr("__dataframe__", df):
        raise ValueError(f"'{name}' must be a dataframe, got {type(df)}.")


def check_missing_columns(df, cols, name):
    expected_cols = set(cols)
    missing = expected_cols.difference(df.columns)
    if len(missing) > 0:
        raise ValueError(f"columns {missing} are missing from '{name}'.")


def check_no_duplicate_id(df, id_col, name):
    duplicates = df.shape[0] - df[id_col].nunique()
    if duplicates > 0:
        raise ValueError(
            f"{id_col} in {name} must be unique. Got {duplicates} duplicates."
        )
