import pytest
import pandas as pd

from credit_risk_models.risk_model_survival_analysis import _behavior


def test_audit():

    audit = _behavior._get_audit()

    # Check that there is no duplicate in our audit ids
    assert audit.shape[0] == audit.dropna(subset='audit_id').shape[0]



def test_dd():

    dd = _behavior._get_dd()

    # Check that there is no duplicate in our dd ids
    assert dd.shape[0] == dd.dropna(subset='dd_id').shape[0]



def test_loan():

    audit_loan = _behavior.get_audit_overdue()

    audit_overdue_mean_by_state = (
        audit_loan.groupby("loan_state")["audit_overdue"]
        .mean()
        .to_dict()
    )
    expected = {
        0: 0.028,
        80: 0.159,
        100: 0.061,
    }
    for k, v in audit_overdue_mean_by_state.items():
        assert pytest.approx(expected[k], abs=0.03) == v
