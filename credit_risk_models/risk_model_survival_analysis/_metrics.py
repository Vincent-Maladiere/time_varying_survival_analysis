from tqdm import tqdm
from collections import defaultdict

import numpy as np
from scipy.interpolate import interp1d
from lifelines import AalenJohansenFitter
from sksurv.metrics import concordance_index_ipcw
from hazardous.utils import check_y_survival
from hazardous.metrics import integrated_brier_score_incidence

from . import _utils


def c_index(y_train, y_test, y_pred, time_grid, truncation_quantiles):

    c_indices = defaultdict(list)
    n_events = y_pred.shape[1]
    for event_id in range(1, n_events):    
        y_train_binary, y_test_binary = y_train.copy(), y_test.copy()
        y_train_binary["event"] = (y_train_binary["event"] == event_id)
        y_train_binary["event"] = (y_test_binary["event"] == event_id)

        taus = np.quantile(time_grid, truncation_quantiles)
        taus = tqdm(
            taus,
            desc=f"c-index at tau for event {event_id}",
            total=len(taus),
        )
        for tau in taus:
            tau_idx = np.searchsorted(time_grid, tau)
            y_pred_at_t = y_pred[:, event_id, tau_idx]
            ct_index, _, _, _, _ = concordance_index_ipcw(
                _utils.make_recarray(y_train_binary),
                _utils.make_recarray(y_test_binary),
                y_pred_at_t,
                tau=tau,
            )
            c_indices[event_id].append(round(ct_index, 4))

    return c_indices


def accuracy_in_time(y_test, y_pred, times, quantiles=None, taus=None):
    event_true, _ = check_y_survival(y_test)

    if y_pred.ndim != 3:
        raise ValueError(
            "'y_pred' must be a 3D array with shape (n_samples, n_events, n_times), got"
            f" shape {y_pred.shape}."
        )
    
    if y_pred.shape[0] != event_true.shape[0]:
        raise ValueError(
            "'y_true' and 'y_pred' must have the same number of samples, "
            f"got {event_true.shape[0]} and {y_pred.shape[0]} respectively."
        )
    
    times = np.atleast_1d(times)
    if y_pred.shape[2] != times.shape[0]:
        raise ValueError(
            f"'times' length ({times.shape[0]}) "
            f"must be equal to y_pred.shape[2] ({y_pred.shape[2]})."
        )

    if quantiles is not None:
        if taus is not None:
            raise ValueError("'quantiles' and 'taus' can't be set at the same time.")

        quantiles = np.atleast_1d(quantiles)
        if any(quantiles < 0) or any(quantiles > 1):
            raise ValueError(f"quantiles must be in [0, 1], got {quantiles}.")
        taus = np.quantile(times, quantiles)

    elif quantiles is None and taus is None:
        n_quantiles = min(times.shape[0], 8)
        quantiles = np.linspace(1 / n_quantiles, 1, n_quantiles)
        taus = np.quantile(times, quantiles)

    acc_in_time = []

    for tau in taus:
        mask_past_censored = (y_test["event"] == 0) & (y_test["duration"] < tau)

        tau_idx = np.searchsorted(times, tau)
        y_pred_at_t = y_pred[:, :, tau_idx]
        y_pred_class = y_pred_at_t[~mask_past_censored, :].argmax(axis=1)

        y_test_class = y_test["event"] * (y_test["duration"] < tau)
        y_test_class = y_test_class.loc[~mask_past_censored]

        acc_in_time.append((y_test_class.values == y_pred_class).mean())

    return acc_in_time, taus


def integrated_brier_score(y_train, y_test, y_proba, time_grid):
    ibs = {}
    n_events = _utils.get_n_events(y_train["event"])
    for event_id in (1, n_events):
        ibs_event = integrated_brier_score_incidence(
            y_train,
            y_test,
            y_proba[:, event_id, :],
            times=time_grid,
            event_of_interest=event_id,
        )
        ibs[event_id] = round(ibs_event, 4)
    return ibs


def _get_proba_aj(y_train, n_samples, time_grid):
    """Estimate probabilities for the marginal Aalen-Johansen estimator.

    This probabilities are identical for all samples, they are useful to compute
    the accuracy in time or proper scoring rules (brier score, log loss).
    """
    cifs = []
    for event_id in range(1, 3):
        aj = AalenJohansenFitter(calculate_variance=False)
        aj.fit(y_train["duration"], y_train["event"], event_of_interest=event_id)

        cif = aj.cumulative_density_[f"CIF_{event_id}"].values
        times = aj.cumulative_density_.index.round()

        cif = interp1d(x=times, y=cif)(time_grid)
        cif = np.vstack([cif] * n_samples)
        cifs.append(cif[None, :, :])

    cif = np.concatenate(cifs, axis=0)
    surv = (1 - cif.sum(axis=0))[None, :, :]
    cif = np.concatenate([surv, *cifs], axis=0).swapaxes(0, 1)

    return cif
