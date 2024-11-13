import pandas as pd
from pandas.testing import assert_frame_equal
import pytest

from credit_risk_models.risk_model_survival_analysis._loans import (
    get_loans, Risks
)
from credit_risk_models.risk_model_survival_analysis._make_dataset import (
    _agg_join_audit_loan, _agg_join_audit_dealer, _agg_join_labels_dealer
)

@pytest.fixture
def sample_loans():
    """Create a fake loans dataset for testing purposes.
    """
    loans = pd.DataFrame({
        "borrower_id": [1, 1, 2, 2],
        "carloan_id": [1, 2, 3, 4],
        "risks": [
            Risks.reimbursed.value,
            Risks.reimbursed.value,
            Risks.car_sold_np.value,
            Risks.on_going.value,
        ],
        "loan_duration": [20, 80, 40, 60],
        "loan_created_date": pd.to_datetime(
            ["2021-01-01", "2022-01-01", "2021-06-01", "2022-06-01"]
        ),
        "loan_end_date": pd.to_datetime(
            ['2021-01-21', '2022-03-22', '2021-07-11', None]
        ),
        "loan_reimbursed_date": pd.to_datetime(
            ['2021-01-21', '2022-03-22', None, None]
        )
    })

    return loans


@pytest.fixture
def sample_audits():
    """Create a fake audits dataset for testing purposes
    """
    audits = pd.DataFrame({
        "carloan_id": [1, 2, 2, 3, 3, 4, 4, 4],
        "audit_id": [1, 2, 3, 4, 5, 6, 7, 8],
        "audit_scheduled_for_from": pd.to_datetime([
            "2021-01-01", "2022-01-01", "2022-03-01", "2021-06-05",
            "2021-06-20", "2022-06-01", "2022-06-05", "2022-06-30",
        ]),
        "audit_due_date": pd.to_datetime([
            "2021-01-20", "2022-02-01", "2022-03-20", "2021-06-15",
            "2021-07-03", "2022-06-10", "2022-06-15", "2022-07-01",
        ]),
        "audit_dpd14_date": pd.to_datetime([
            "2021-01-20", "2022-02-01", "2022-03-20", "2021-06-15",
            "2021-07-03", "2022-06-10", "2022-06-15", "2022-07-01",
        ]) + pd.Timedelta(days=1),
        "audit_end_date": pd.to_datetime([
            "2021-01-05", "2022-01-15", "2022-03-15", "2021-06-20",
            "2021-07-05", "2022-06-05", "2022-06-11", None,
        ]),
        # audits with only 0 are either ongoing or overdue
        "approved": [1, 0, 1, 0, 0, 0, 1, 0],
        "rejected": [0, 1, 0, 0, 0, 0, 0, 0],
        "cancelled": [0, 0, 0, 0, 0, 1, 0, 0],
    })
    return audits


@pytest.fixture
def loans():
    """Fetch the labels from the database.
    """ 
    return get_loans()


def test_agg_join_audit_loan(sample_loans, sample_audits):
    sample_loans = sample_loans.copy()
    sample_audits = sample_audits.copy()

    sample_loans["observation_date"] = pd.to_datetime([
        "2021-01-10", "2022-02-01", "2021-06-30", "2022-07-05",
    ])

    expected_single_obs = pd.DataFrame({
        "borrower_id": [1, 1, 2, 2],
        "carloan_id": [1, 2, 3, 4],
        "risks": [
            Risks.reimbursed.value,
            Risks.reimbursed.value,
            Risks.car_sold_np.value,
            Risks.on_going.value,
        ],
        "loan_duration": [20, 80, 40, 60],
        "loan_created_date": pd.to_datetime(
            ["2021-01-01", "2022-01-01", "2021-06-01", "2022-06-01"]
        ),
        "loan_end_date": pd.to_datetime(
            ['2021-01-21', '2022-03-22', '2021-07-11', None]
        ),
        "loan_reimbursed_date": pd.to_datetime([
            '2021-01-21', '2022-03-22', None, None,
        ]),
        "observation_date": pd.to_datetime(
            ["2021-01-10", "2022-02-01", "2021-06-30", "2022-07-05"]
        ),
        "loan_n_past_audits": [1, 1, 2, 3],
        "loan_n_audit_late": [0, 0, 1, 1],
        "loan_n_audit_approved": [1, 0, 0, 1],
        "loan_n_audit_rejected": [0, 1, 0, 0],
        "loan_ratio_audit_late": [0, 0, 1/2, 1/3],
        "loan_ratio_audit_approved": [1., 0, 0, 1/3],
        "loan_ratio_audit_rejected": [0, 1., 0, 0],
    })

    single_obs = _agg_join_audit_loan(sample_loans, sample_audits)

    assert_frame_equal(single_obs, expected_single_obs)


def test_agg_join_audit_dealer(sample_loans, sample_audits):
    sample_loans = sample_loans.copy()
    sample_audits = sample_audits.copy()

    sample_loans["observation_date"] = pd.to_datetime([
        "2021-01-10", "2022-02-01", "2021-06-30", "2022-07-05",
    ])

    expected_single_obs = pd.DataFrame({
        "borrower_id": [1, 1, 2, 2],
        "carloan_id": [1, 2, 3, 4],
        "risks": [
            Risks.reimbursed.value,
            Risks.reimbursed.value,
            Risks.car_sold_np.value,
            Risks.on_going.value,
        ],
        "loan_duration": [20, 80, 40, 60],
        "loan_created_date": pd.to_datetime(
            ["2021-01-01", "2022-01-01", "2021-06-01", "2022-06-01"]
        ),
        "loan_end_date": pd.to_datetime(
            ['2021-01-21', '2022-03-22', '2021-07-11', None]
        ),
        "loan_reimbursed_date": pd.to_datetime(
            ['2021-01-21', '2022-03-22', None, None]
        ),
        "observation_date": pd.to_datetime(
            ["2021-01-10", "2022-02-01", "2021-06-30", "2022-07-05"]
        ),
        "dealer_n_past_audits": [1, 2, 2, 5],
        "dealer_n_audit_overdue": [0, 0, 1, 3],
        "dealer_n_audit_approved": [1, 1, 0, 1],
        "dealer_n_audit_rejected": [0, 1, 0, 0],
        "dealer_ratio_audit_overdue": [0, 0, 1/2, 3/5],
        "dealer_ratio_audit_approved": [1, 1/2, 0, 1/5],
        "dealer_ratio_audit_rejected": [0, 1/2, 0, 0],
    })

    single_obs = _agg_join_audit_dealer(sample_loans, sample_audits)

    assert_frame_equal(single_obs, expected_single_obs)


def test_agg_join_labels_dealer(sample_loans):
    sample_loans = sample_loans.copy()

    sample_loans["observation_date"] = pd.to_datetime([
        "2021-01-10", "2022-02-01", "2021-06-30", "2022-07-05",
    ])

    expected_single_obs = pd.DataFrame({
        "borrower_id": [1, 1, 2, 2],
        "carloan_id": [1, 2, 3, 4],
        "risks": [
            Risks.reimbursed.value,
            Risks.reimbursed.value,
            Risks.car_sold_np.value,
            Risks.on_going.value,
        ],
        "loan_duration": [20, 80, 40, 60],
        "loan_created_date": pd.to_datetime(
            ["2021-01-01", "2022-01-01", "2021-06-01", "2022-06-01"]
        ),
        "loan_end_date": pd.to_datetime(
            ['2021-01-21', '2022-03-22', '2021-07-11', None]
        ),
        "loan_reimbursed_date": pd.to_datetime(
            ['2021-01-21', '2022-03-22', None, None]
        ),
        "observation_date": pd.to_datetime([
            "2021-01-10", "2022-02-01", "2021-06-30", "2022-07-05",
        ]),
        "dealer_n_cars_financed": [1, 2, 1, 2],
        "dealer_avg_reimbursment_days": [0., 20., 0., 0.],
        "dealer_n_cars_reimbursed": [0, 1, 0, 0],
        "dealer_n_maturity_reached": [0, 0, 0, 0],
        "dealer_n_cars_sold_np": [0, 0, 0, 1],
        "dealer_n_loan_ongoing": [1, 1, 1, 1],
        "dealer_ratio_reimbursed": [0, 1/2, 0, 0],
        "dealer_ratio_maturity_reached": [0., 0., 0., 0.],
        "dealer_ratio_cars_sold_np": [0, 0, 0, 1/2],
    })

    single_obs = _agg_join_labels_dealer(sample_loans)

    assert_frame_equal(single_obs, expected_single_obs)
