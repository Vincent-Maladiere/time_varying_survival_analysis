"""
This notebook is a proof of concept for the projet. It exposes the labels,
the features and the modeling strategy used.
This was shown to non-technical stakeholders with cell outputs generated.
"""
# %%
from credit_risk_models.risk_model_survival_analysis._make_dataset import DatasetMaker
from skrub import TableReport

# This class performs all feature preprocessing necessary, and store each individual
# table in cache. You can inspect it using 'dir(ds)'.
ds = DatasetMaker(is_training=True, max_n_draw=3)

# Making a copy is not mandatory, but it allows us to alter 'df' while keeping
# the original values in the DatasetMaker.
df = ds.dataset.copy()

# A handful companion to visualise datasets.
TableReport(df).open()

# %%
# We display our event distribution, where 0 represents surviving to any event, 1
# represents experiencing a reimbursement and 2 represents experiencing a car sold not
# paid.
# The x-axis represent the age of the loan (in days) at which the event happened,
# and for on-going loans (in blue), the current age of the loan.
from credit_risk_models.risk_model_survival_analysis import _plots

_plots.plot_event_distribution(df)

# %%
from sklearn.model_selection import GroupShuffleSplit
from skrub import TableVectorizer, MinHashEncoder, GapEncoder
from sklearn.preprocessing import OrdinalEncoder
from hazardous import SurvivalBoost

from credit_risk_models.risk_model_survival_analysis import _utils

# We use skrub TableVectorizer as a handler for categorical columns:
# - for categorical columns with cardinality <= 30, it uses scikit-learn OrdinalEncoder.
# - for categorical columns with cardinality > 30, it uses skrub MinHashEncoder.
#
# To ease our preprocessing, we chain in a Pipeline the TableVectorizer with our
# survival estimator from hazardous, SurvivalBoost.
#
# Since in hazardous our methods for training and predicting are 'fit' and
# 'predict_cumulative_incidence', we have to customize the Pipeline class a little,
# and use CumulativeIncidencePipeline.
#
# See more on TableVectorizer at: https://skrub-data.org/stable/reference/generated/skrub.TableVectorizer.html
# See more on SurvivalBoost at: https://soda-inria.github.io/hazardous/generated/hazardous.SurvivalBoost.html

model = _utils.CumulativeIncidencePipeline(
    [
        ("tv", TableVectorizer(
            low_cardinality=OrdinalEncoder(),
            high_cardinality=GapEncoder(),
        )),
        ("model", SurvivalBoost(n_iter=100, learning_rate=0.05, max_depth=5)),
    ]
)

label_cols = ["event", "duration"]
id_cols = ["carloan_id", "borrower_id"]
X = df.drop(columns=label_cols + id_cols)
y = df[label_cols]

# To account for our numerous time-varying features, we have generated multiple samples
# of the same loans, at different age. For evaluation, it's best to group together
# the different instances from the same dealer (and thus from the same loans).
# Therefore, if dealer_1 are in the train set, none of their loan will be in the test
# set.
# GroupShuffleSplit allows us to just this: a train test split while grouping the
# instances by dealer.
#
gss = GroupShuffleSplit(test_size=0.2)
train_indices, test_indices = next(gss.split(X, y, groups=df["borrower_id"]))
X_train, y_train, X_test, y_test = (
    X.iloc[train_indices],
    y.iloc[train_indices],
    X.iloc[test_indices],
    y.iloc[test_indices],
)
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

model.fit(X_train, y_train)

# %%
# SurvivalBoost outputs probabilities estimates for the different events of interest.
# Its output has 3 dimensions: (n_samples, n_events, n_time_steps)
# - n_samples is the number of samples in X_test
# - n_events is the number of events we consider in this survival analysis setting.
#   * The class 0 represents surviving to (i.e. not having) any event
#   * The class 1 represents experiencing the reimbursement event
#   * The class 2 represents experiencing the car sold not paid event
# - n_time_steps is the number of points we use in our time grid (100 by default).
#   Here a step is equal to multiple days.

y_proba = model.predict_cumulative_incidence(X_test)
y_proba.shape

# %%
# FIXME
# _plots.plot_mean_cifs(y_train, y_proba, model.time_grid)

# %%
# The C-index represents the ability of the model to rank loans by risk.
# Higher is better.
#
# Given pairs of loans (i, j), where i has experienced the event of interest before j,
# the C-index counts the number of time the model correctly predicted than the
# probability of the event of interest (reimbursment or car sold not paid) for
# the loan i is higher than the corresponding probability for the loan j.
#
# A perfect model has a C-index of 1, and a random model has a C-index of 0.5.
#
# Here, we pass a list a quantile to assess the performance of the model for loans
# that experienced events before some quantile of the time grid. 0.25 means we only
# compute the C-index for loans which experienced events on the first 25% of
# the time grid.
from credit_risk_models.risk_model_survival_analysis import _metrics

truncation_quantiles = [0.25, 0.5, 0.75]
c_indices = _metrics.c_index(
    y_train, y_test, y_proba, model.time_grid, truncation_quantiles
)
c_indices

# %%
# The accuracy in time represents the ability of the model to rank the different risks
# of the same loans. Higher is better.
#
# Given a loan, it counts the number of time the highest predicted probability
# corresponds to the ground truth class.
import numpy as np

quantiles = np.arange(0.125, 1, 0.125)
_plots.plot_accuracy_in_time(y_train, y_test, y_proba, model.time_grid, quantiles)

# %%
# The brier score represents the ability of the model to predict a probability close
# to the ground truth. It measures both calibration and discriminative power.
# Lower is better.

ibs = _metrics.integrated_brier_score(y_train, y_test, y_proba, model.time_grid)
print(f"SurvivalBoost ibs: {ibs}")

# Get the probabilities of the marginal estimator.
y_proba_aj = _metrics._get_proba_aj(
    y_train, n_samples=y_test.shape[0], time_grid=model.time_grid
)
ibs_aj = _metrics.integrated_brier_score(y_train, y_test, y_proba_aj, model.time_grid)
print(f"Aalen-Johanson ibs: {ibs_aj}")

# %%
# Plot individual predictions and features.
_plots.plot_individual_pred(
    indices=[100],
    df=df,
    y_test=y_test,
    y_proba=y_proba,
    time_grid=model.time_grid,
)

# %%
# Marginal feature importance

model[-1].set_params(show_progressbar=False)
_plots.plot_permutation_importance(
    model, X_test, y_test, n_jobs=4
)

# %%
# Conditional feature importance
from matplotlib import pyplot as plt
from lime.lime_tabular import LimeTabularExplainer

vectorizer, estimator = model[0], model[-1]
X_train_trans = vectorizer.transform(X_train)
X_test_trans = vectorizer.transform(X_test)
estimator.set_params(time_horizon=50)  # Has to be set for predict proba

explainer = LimeTabularExplainer(
    training_data=X_train_trans.values,
    feature_names=X_train_trans.columns.tolist(),
    mode="classification",
    discretize_continuous=False,
)
exp = explainer.explain_instance(
    X_test_trans.values[101, :],
    estimator.predict_proba,
    num_features=20,
    labels=(0, 1, 2),
)
exp.as_pyplot_figure(label=0)
plt.show()

exp.as_pyplot_figure(label=1)
plt.show()

exp.as_pyplot_figure(label=2)
plt.show()

# %%

exp.as_list(label=1)
