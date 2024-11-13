import pickle
import pandas as pd
from pathlib import Path
from dataclasses import dataclass

from skrub import TableVectorizer, GapEncoder
from sklearn.preprocessing import OrdinalEncoder
from hazardous import SurvivalBoost

from . import _make_dataset
from . import _utils
from . import _plots
from . import _logs


@dataclass
class TrainTask(_logs.LogsMixin):
    model_name: str
    dpd_limit: int = 240

    def run(self):
        """Train a model locally and store it on disk."""
        now = pd.Timestamp.now().strftime(_utils.FOLDER_DATETIME_FORMAT)
        self.path_folder = Path(f"training_run_{now}")
        self.path_folder.mkdir(exist_ok=True)        
        
        self.ds = _make_dataset.DatasetMaker(
            is_training=True,
            max_n_draw=3,
        )
        df = self.ds.dataset

        label_cols = ["event", "duration"]
        id_cols = ["carloan_id", "borrower_id"]
        X = df.drop(columns=label_cols + id_cols)
        y = df[label_cols]

        # make a plot and save it on disk
        _plots.plot_event_distribution(y, self.path_folder)

        self.estimator = self._get_estimator()
        self.estimator.fit(X, y)
        self._save_model()
    
    def _get_estimator(self):
        return _utils.CumulativeIncidencePipeline(
            [
                ("tv", TableVectorizer(
                    low_cardinality=OrdinalEncoder(),
                    high_cardinality=GapEncoder(),
                )),
                ("model", SurvivalBoost(n_iter=100, learning_rate=0.05, max_depth=5)),
            ]
        )

    def _save_model(self):
        filename = self.path_folder / "model.pkl"

        pickle.dump(self.estimator, open(filename, "wb"))
        self._log_info("dumped", filename)
        return
