from functools import cached_property
from dataclasses import dataclass
import pandas as pd
from tqdm import tqdm
from sklearn.utils import check_random_state

from . import _automative
from . import _loans
from . import _company_data
from . import _audits
from . import db

DATASET_COLS = [
    "carloan_id",
    "borrower_id",
    "event",
    "duration",
    "loan_age_days",
    "loan_amount",
    "car_make",
    "car_model",
    "car_transmission_type",
    "car_source",
    "country_code",
    "n_days_since_founded",
    "owner_age_year",
    "loan_n_past_audits",
    "loan_n_audit_overdue",
    "loan_n_audit_approved",
    "loan_n_audit_rejected",
    "loan_ratio_audit_overdue",
    "loan_ratio_audit_approved",
    "loan_ratio_audit_rejected",
    "dealer_n_past_audits",
    "dealer_n_audit_overdue",
    "dealer_n_audit_approved",
    "dealer_n_audit_rejected",
    "dealer_ratio_audit_overdue",
    "dealer_ratio_audit_approved",
    "dealer_ratio_audit_rejected",    
    "dealer_n_cars_financed",
    "dealer_avg_reimbursment_days",
    "dealer_n_cars_reimbursed",
    "dealer_n_maturity_reached",
    "dealer_n_cars_sold_np",
    "dealer_n_loan_ongoing",
    "dealer_ratio_reimbursed", 
    "dealer_ratio_maturity_reached",
    "dealer_ratio_cars_sold_np",
]
LABEL_COLS = ["event", "duration"]

# Remove fraudulent customers, because KYC has been improved and they do not
# represent current customers for which we predict loan defaults.
FRAUD_COMPANIES = [
    "A41282393",
    "B01747534",
    "B90059809",
    "B85866655",
    "B90343799",
    "B39833975",
    "B85573715",
    "B42845503",
    "B90031667",
    "B61482477",
    "B91135376",
    "B28462778",
    "B42984351",
    "B93128452",
    "B98651433",
    "B39875968",
    "B16706012",
    "B74401696",
    "B42984369",
]


@dataclass
class DatasetMaker:
    is_training: bool = True
    draw_sample_period: int = 30
    random_state: int = 42
    max_n_draw: int = 3
    verbose: bool = True

    def push_dataset(self):
        df = self.dataset
        db.DBSourceRisk().write_df(
            df,
            table_name="p0_features",
            schema="risks",
        )

    def get_X_y(self):
        df = self.dataset
        y = df[LABEL_COLS]
        X = df.drop(LABEL_COLS, axis=1)

        return X, y
    
    @cached_property
    def dataset(self):

        loans_observations = (
            self.loans_observations.merge(
                self.cars,
                # Using additional ids in keys to avoid having suffixes _x, _y
                # in the final merged dataframe.
                on=["carloan_id", "borrower_id", "collateral_id"],
                how="left",
            )
            .merge(self.companies, on="borrower_id", how="left")
            .rename(columns={
                "risks": "event",
                "target_duration": "duration",
            })
        )

        # Remove "Other default" as the definition of default is unclear for them,
        # and remove loans created before the 2024-03-01.
        mask = (
            (loans_observations["event"] != _loans.Risks.other_default)
            & (loans_observations["loan_created_date"] > pd.Timestamp("2024-03-01"))
        )
        loans_observations = loans_observations.loc[mask]

        # Map event to integers
        loans_observations["event"] = loans_observations["event"].map({
            _loans.Risks.on_going.value: 0,
            _loans.Risks.maturity_reached.value: 0,
            _loans.Risks.reimbursed.value: 1,
            _loans.Risks.car_sold_np.value: 2,
            _loans.Risks.audit_overdue: 2,
            _loans.Risks.dd_overdue: 2,
        })
                
        # Remove fraud users
        mask = ~loans_observations["company_registration_number"].isin(FRAUD_COMPANIES)
        loans_observations = loans_observations.loc[mask]
                
        return loans_observations[DATASET_COLS].reset_index(drop=True)

    @cached_property
    def loans(self):
        return _loans.get_loans()

    @cached_property
    def audits(self):
        return _audits.get_audits()

    @cached_property
    def due_diligences(self):
        return _audits.get_dd()

    @cached_property
    def cars(self):
        return _automative.get_automative()

    @cached_property
    def companies(self):
        return _company_data.get_company_data()

    @property
    def _features(self):
        return list(set(DATASET_COLS).difference(LABEL_COLS))

    @cached_property
    def loans_observations(self):
        """Draw observation dates for all loans, proportionally to their length."""
        if self.is_training:
            return self._loans_observations_train
        else:
            return self._loans_observations_test
        
    @cached_property
    def _loans_observations_train(self):

        loans = self.loans.copy()
        rng = check_random_state(self.random_state)

        # Set loan duration relative to today for on-going loans. This is only use to
        # compute the number of samples to be made.
        loans.loc[loans["is_ongoing"], "loan_duration"] = (
            pd.Timestamp.now() - loans.loc[loans["is_ongoing"]]["loan_created_date"]
        ).dt.days

        # Loan duration is capped to 149 days.
        loans["loan_duration_clipped"] = loans["loan_duration"].clip(
            upper=_loans.TC_LIMIT
        )

        loans["n_observation_draw"] = (
            loans["loan_duration_clipped"] // self.draw_sample_period
        )
        max_n_draw = min(loans["n_observation_draw"].max(), self.max_n_draw)

        iter_ = range(int(max_n_draw) + 1)
        if self.verbose:
            iter_ = tqdm(iter_)

        loans_obs = []
        for n_draw in iter_:
            if n_draw == 0:
                # All loans have at least one observation: their creation date.
                single_obs = loans.copy()
                single_obs["observation_date"] = single_obs["loan_created_date"]
            else:
                single_obs = loans.query("n_observation_draw >= @n_draw").reset_index(
                    drop=True
                )
                sampled_days = rng.randint(
                    low=0, high=single_obs["loan_duration_clipped"], dtype="int32"
                )
                single_obs["sampled_days"] = pd.to_timedelta(sampled_days, unit="D")
                single_obs["observation_date"] = (
                    single_obs["loan_created_date"] + single_obs["sampled_days"]
                )

            single_obs["loan_age_days"] = (
                single_obs["observation_date"] - single_obs["loan_created_date"]
            ).dt.days

            single_obs["target_duration"] = (
                single_obs["loan_duration"] - single_obs["loan_age_days"]
            )

            loans_obs.append(self._compute_aggregate(single_obs))

        loans_obs = (
            pd.concat(loans_obs, axis=0)
            .sort_values(["carloan_id", "observation_date"])
            .reset_index(drop=True)
        )
        return loans_obs

    @cached_property
    def _loans_observations_test(self):
        
        loans = self.loans.copy()

        # We can't remove closed loans at this stage because we need them to derive
        # features for on-going loans.
        loans["observation_date"] = pd.Timestamp.now()
        loans_obs = self._compute_aggregate(loans)

        # Now that we build our desired features, we only keep on-going loans for
        # testing.
        mask = (loans_obs["risks"] == _loans.Risks.on_going)
        loans_obs = loans_obs.loc[mask]

        loans_obs["loan_age_days"] = (
            loans_obs["observation_date"] - loans_obs["loan_created_date"]
        ).dt.days

        # For consistency, we shouldn't train the model using these.
        loans_obs["loan_duration"] = (
            pd.Timestamp.now() - loans_obs["loan_created_date"]
        ).dt.days
        loans_obs["target_duration"] = (
            loans_obs["loan_duration"] - loans_obs["loan_age_days"]
        ) 

        loans_obs = (
            loans_obs
            .sort_values(["carloan_id", "observation_date"])
            .reset_index(drop=True)
        )

        return loans_obs

    def _compute_aggregate(self, single_obs):

        single_obs = _agg_join_audit_loan(single_obs, self.audits.copy())
        single_obs = _agg_join_audit_dealer(single_obs, self.audits.copy())
        single_obs = _agg_join_labels_dealer(single_obs)

        return single_obs


def _agg_join_audit_loan(single_obs, audits):
    """Aggregate audit features at the loan level.
    """
    cols = ["carloan_id", "observation_date"]
    audits = audits.merge(single_obs[cols], on="carloan_id")

    # Only keep audits when we can observe their "scheduled from" date
    # and that are not cancelled.
    obs_date = audits["observation_date"]
    mask = (
        (audits["audit_scheduled_for_from"] <= obs_date)
        & (~audits["audit_cancelled"])
    )
    n_past_audits = (
        audits.loc[mask].groupby("carloan_id").size().rename("loan_n_past_audits")
    )

    audits["is_audit_overdue"] = _get_audit_overdue_mask(audits, obs_date)
    loan_n_audit_overdue = (
        audits.groupby("carloan_id")[["is_audit_overdue"]]
        .sum()
        .rename(
            columns={
                "is_audit_overdue": "loan_n_audit_overdue",
            }
        )
    )

    mask = audits["audit_submission_date"] < audits["observation_date"]
    rename_cols = {
        "audit_approved": "loan_n_audit_approved",
        "audit_rejected": "loan_n_audit_rejected",
    }
    totals = (
        audits.loc[mask]
        .groupby("carloan_id")[list(rename_cols)]
        .sum()
        .rename(columns=rename_cols)
    )

    # concat() uses the carloan_id index to align the different dataframes.
    group = pd.concat(
        [n_past_audits, loan_n_audit_overdue, totals], axis=1
    ).reset_index()

    group["loan_ratio_audit_overdue"] = (
        group["loan_n_audit_overdue"] / group["loan_n_past_audits"]
    )
    group["loan_ratio_audit_approved"] = (
        group["loan_n_audit_approved"] / group["loan_n_past_audits"]
    )
    group["loan_ratio_audit_rejected"] = (
        group["loan_n_audit_rejected"] / group["loan_n_past_audits"]
    )

    cols = [
        "carloan_id",
        "loan_n_past_audits",
        "loan_n_audit_overdue",
        "loan_n_audit_approved",
        "loan_n_audit_rejected",
        "loan_ratio_audit_overdue",
        "loan_ratio_audit_approved",
        "loan_ratio_audit_rejected",
    ]
    single_obs = single_obs.merge(group[cols], on="carloan_id", how="left")
    single_obs[cols] = single_obs[cols].fillna(0)

    return single_obs


def _agg_join_audit_dealer(single_obs, audits):
    """Aggregate audit features at the dealer level.
    """
    cols = ["carloan_id", "borrower_id"]
    audits = audits.merge(single_obs[cols], on="carloan_id")

    group = []
    for carloan_id, borrower_id, obs_date in single_obs[
        ["carloan_id", "borrower_id", "observation_date"]
    ].values:
        borrower_audits = audits.query("borrower_id == @borrower_id").reset_index(
            drop=True
        )

        mask_past_audits = (
            (borrower_audits["audit_scheduled_for_from"] < obs_date)
            & (~borrower_audits["audit_cancelled"])
        )
        n_past_audits = mask_past_audits.sum()

        n_audit_overdue = _get_audit_overdue_mask(borrower_audits, obs_date).sum()

        mask = (
            mask_past_audits
            & (borrower_audits["audit_submission_date"] < obs_date)
        )
        n_audit_approved = (borrower_audits["audit_approved"] & mask).sum()
        n_audit_rejected = (borrower_audits["audit_rejected"] & mask).sum()

        group.append(
            dict(
                carloan_id=carloan_id,
                dealer_n_past_audits=n_past_audits,
                dealer_n_audit_overdue=n_audit_overdue,
                dealer_n_audit_approved=n_audit_approved,
                dealer_n_audit_rejected=n_audit_rejected,
            )
        )

    group = pd.DataFrame(group)
    group["dealer_ratio_audit_overdue"] = (
        group["dealer_n_audit_overdue"] / group["dealer_n_past_audits"]
    )
    group["dealer_ratio_audit_approved"] = (
        group["dealer_n_audit_approved"] / group["dealer_n_past_audits"]
    )
    group["dealer_ratio_audit_rejected"] = (
        group["dealer_n_audit_rejected"] / group["dealer_n_past_audits"]
    )

    cols = [
        "carloan_id",
        "dealer_n_past_audits",
        "dealer_n_audit_overdue",
        "dealer_n_audit_approved",
        "dealer_n_audit_rejected",
        "dealer_ratio_audit_overdue",
        "dealer_ratio_audit_approved",
        "dealer_ratio_audit_rejected",  
    ]
    single_obs = single_obs.merge(group[cols], on="carloan_id", how="left")
    single_obs[cols] = single_obs[cols].fillna(0)

    return single_obs


def _agg_join_labels_dealer(single_obs):
    """Aggregate labels at the dealer level.
    """
    # TODO: this function takes 10s to run, improve or find the bottleneck.
    group = []
    for carloan_id, borrower_id, obs_date in single_obs[
        ["carloan_id", "borrower_id", "observation_date"]
    ].values:
        borrower_loans = single_obs.query("borrower_id == @borrower_id")

        n_cars_financed = (borrower_loans["loan_created_date"] < obs_date).sum()

        # Null dates are not taken into account in this average
        mask = borrower_loans["loan_reimbursed_date"] < obs_date
        avg_reimbursment_days = (
            borrower_loans.loc[mask]["loan_reimbursed_date"]
            - borrower_loans.loc[mask]["loan_created_date"]
        ).dt.days.mean()

        mask = borrower_loans["loan_end_date"] < obs_date
        n_cars_sold_np = (
            (borrower_loans["risks"] == _loans.Risks.car_sold_np) & mask
        ).sum()

        n_maturity_reached = (
            (borrower_loans["risks"] == _loans.Risks.maturity_reached) & mask
        ).sum()

        n_cars_reimbursed = (
            (borrower_loans["risks"] == _loans.Risks.reimbursed) & mask
        ).sum()

        # A loan is on-going when we haven't observed the end date yet.
        n_loan_ongoing = (
            (borrower_loans["loan_created_date"] < obs_date)
            & (~(borrower_loans["loan_end_date"] < obs_date))
        ).sum()

        group.append(
            dict(
                carloan_id=carloan_id,
                dealer_n_cars_financed=n_cars_financed,
                dealer_avg_reimbursment_days=avg_reimbursment_days,
                dealer_n_cars_reimbursed=n_cars_reimbursed,
                dealer_n_maturity_reached=n_maturity_reached,
                dealer_n_cars_sold_np=n_cars_sold_np,
                dealer_n_loan_ongoing=n_loan_ongoing,
            )
        )

    group = pd.DataFrame(group)
    group["dealer_ratio_reimbursed"] = (
        group["dealer_n_cars_reimbursed"] / group["dealer_n_cars_financed"]
    )
    group["dealer_ratio_maturity_reached"] = (
        group["dealer_n_maturity_reached"] / group["dealer_n_cars_financed"]
    )
    group["dealer_ratio_cars_sold_np"] = (
        group["dealer_n_cars_sold_np"] / group["dealer_n_cars_financed"]
    )

    cols = [
        "carloan_id",
        "dealer_n_cars_financed",
        "dealer_avg_reimbursment_days",
        "dealer_n_cars_reimbursed",
        "dealer_n_maturity_reached",
        "dealer_n_cars_sold_np",
        "dealer_n_loan_ongoing",
        "dealer_ratio_reimbursed", 
        "dealer_ratio_maturity_reached",
        "dealer_ratio_cars_sold_np",
    ]
    single_obs = single_obs.merge(group[cols], on="carloan_id", how="left")
    single_obs[cols] = single_obs[cols].fillna(0)

    return single_obs


def _get_audit_overdue_mask(audits, obs_date):
    """An audit is overdue when we can observe its due date, it hasn't been submitted \
    before the due date and is not cancelled.
    """
    return (
        (audits["audit_due_date"] < obs_date)
        &  (
                ~(audits["audit_submission_date"] < audits["audit_due_date"])
                & (~audits["audit_cancelled"])
        )       
    )
