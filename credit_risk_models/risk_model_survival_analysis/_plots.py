import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import plotly.express as px
from lifelines import AalenJohansenFitter
from sklearn.inspection import permutation_importance

from . import _metrics


def plot_event_distribution(y, path_folder=None):

    fig, ax = plt.subplots(dpi=300)
    palette = sns.color_palette(palette="colorblind")

    ax = sns.histplot(
        y.sort_values("event"),
        x="duration",
        hue="event",
        multiple="stack",
        ax=ax,
        palette=palette,
    )
    ax.axvline(x=150, color="gray", linestyle="--")
    ax.axvline(x=180, color="black", linestyle="--")
    ax.set_xlabel("Days")
    ax.set_ylabel("Total")
    ax.set_title("Event distribution")
    plt.show()

    filename = "fig_y_distribution.png"
    if path_folder is not None:
        filename = path_folder / filename
    plt.savefig(filename)


def plot_mean_cifs(y_train, y_proba, time_grid, model_name="SurvBoost"):

    fig, axes = plt.subplots(figsize=(8, 4), ncols=3, dpi=200, sharey=True)
    cifs = []

    for event_id, ax in zip([1, 2], axes[1:]):
        aj = AalenJohansenFitter(calculate_variance=False)
        aj.fit(y_train["duration"], y_train["event"], event_of_interest=event_id)
        aj.plot_cumulative_density(ax=ax, label="Marginal AJ", linestyle="--")
        cifs.append(aj.cumulative_density_[f"CIF_{event_id}"].values.reshape(-1, 1))

        ax.plot(
            time_grid,
            y_proba[:, event_id, :].mean(axis=0),
            label=model_name,
        )
        ax.set_title(f"CIF event {event_id}")
        ax.legend()

    min_samples = min([cif.shape[0] for cif in cifs])
    marginal_surv = 1 - np.hstack([cif[:min_samples] for cif in cifs]).sum(axis=1)
    axes[0].plot(
        aj.cumulative_density_.index,
        marginal_surv,
        label="Marginal AJ",
        linestyle="--",
    )
    axes[0].plot(
        time_grid,
        y_proba[:, 0, :].mean(axis=0),
        label=model_name,
    )
    axes[0].set_title("Survival probability")
    axes[0].legend()
    plt.show()


def plot_accuracy_in_time(
    y_train,
    y_test,
    y_proba,
    time_grid,
    quantiles,
    model_name="SurvivalBoost",
):
    y_proba_aj = _metrics._get_proba_aj(y_train, y_test.shape[0], time_grid)

    acc_in_time, taus = _metrics.accuracy_in_time(
        y_test, y_proba, time_grid, quantiles=quantiles
    )
    acc_in_time_aj, _ = _metrics.accuracy_in_time(
        y_test, y_proba_aj, time_grid, quantiles=quantiles
    )
    mean = np.mean(acc_in_time)
    mean_aj = np.mean(acc_in_time_aj)

    fig, ax = plt.subplots()
    ax.plot(
        taus, acc_in_time, marker="o", label=f"{model_name} - mean: {mean:.3f}"
    )
    ax.plot(
        taus, acc_in_time_aj, marker="o", label=f"Aalen-Johanson - mean: {mean_aj:.3f}"
    )
    ax.set_ylim([0, 1])
    ax.set_title("Accuracy in time")
    ax.legend()
    plt.show()


def plot_individual_pred(
    indices,
    df,
    y_test,
    y_proba,
    time_grid,
    events_of_interest=("On-going", "Reimbursed", "Default"),
):
    # Select the rows marked by indices in y_test, and use their corresponding indices
    # as selectors. These indices are also part of the original df index.
    indices = np.atleast_1d(indices).tolist()
    mask = y_test.iloc[indices].index.tolist()

    n_events = len(events_of_interest)
    fig, axes = plt.subplots(figsize=(10, 5), ncols=n_events, sharey=True)
    for idx in indices:
        for event_id, event_name in enumerate(events_of_interest):
            event, duration = y_test.iloc[idx][["event", "duration"]]
            axes[event_id].plot(
                time_grid,
                y_proba[idx, event_id, :],
                label=f"event: {event}, duration: {duration}",
            )
            axes[event_id].set_title(event_name)
    plt.legend()
    plt.show()

    print(y_test.loc[mask])
    print(df.loc[mask].T)


def plot_permutation_importance(model, X_test, y_test, **params):

    result = permutation_importance(
        model,
        X_test,
        y_test,
        **params
    )

    result = pd.DataFrame(
        dict(
            feature_names=X_test.columns,
            std=result.importances_std,
            importances=result.importances_mean,
        )
    ).sort_values("importances", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 9), dpi=200)
    result.plot.barh(
        y="importances",
        x="feature_names",
        title="Feature Importances",
        xerr="std",
        fontsize=12,
        ax=ax,
    )
    plt.tight_layout()
    plt.show()
