"""
This file defines the labels for our survival analysis or
binary classification model.

It uses a patch downloaded from Airtable to reconcile missing
reimbursment dates in the Warehouse.
"""
import pandas as pd
from enum import StrEnum

from . import db
from . import _utils

# Terms and conditions limit to reimburse, in days.
TC_LIMIT = 149


# All risks
class Risks(StrEnum):
    on_going = "0 - Ongoing"
    reimbursed = "1 - Reimbursed in time"
    maturity_reached = "2 - Maturity reached"
    car_sold_np = "3 - Collateral sold"
    dd_overdue = "4 - Due Diligence overdue"
    audit_overdue = "5 - Audit overdue"
    other_default = "6 - Other default"


# Risks collapsed in a single "default" class
class SingleRisks(StrEnum):
    on_going = "0 - Ongoing"
    reimbursed = "1 - Reimbursed in time"
    default = "2 - Default"


def get_loans(dpd_limit=60):
    """Compute the labels 'is_default' and 'loan_duration'.

    Parameters
    ----------
    dpd_limit : int, unused.
        Days past due limit.
        The tolerance, in days, before a loan is considered a default
        after the date is past due.

    Returns
    -------
    loans : pandas.DataFrame, whose columns are:
        - carloan_id : str
        - borrower_id : str
        - collateral_id : str
        - is_default : bool
            The event label. Relative to day_past_due_limit.
        - risks : str
            * "0 - Ongoing"
            * "1 - Reimbursed in times"
            * "2 - Maturity reached"
            * "3 - Collateral sold"
            * "4 - Other default"
        - single_risks : str
            * "0 - Ongoing"
            * "1 - Reimbursed in times"
            * "2 - Default"
        - is_ongoing : bool
        - loan_duration : int32
            The duration between the creation date and the 
        - loan_created_date : datetime
        - loan_end_date : datetime
            The combination of the date of reimbursment and termination date.
        - loan_maturity_date : datetime
            The creation date + 149 days.
        - loan_reimbursed_date : datetime
        - terminated_at : datetime
        - loan_span : int
            The duration between the creation and the due date of the loan.
        - loan_state : int
            0 is on going, 80 is a failure and 100 is a reimbursement
        - termination_reason : str
        - raw_termination_reason : str
    """
    loan_status = _get_car_loan_status()
    car_loans = _get_car_loans()

    loans = loan_status.merge(car_loans, on="carloan_id")

    # Remove the few status 100 with no reimbursment date.
    mask = (loans["loan_state"] == 100) & (loans["loan_reimbursed_date"].isnull())
    loans = loans.loc[~mask]

    # Some terminated loans don't have a terminated_at date.
    # In this situation, we use the updated maturity date.
    mask = (
        (loans["termination_reason"].notnull())
        & (loans["terminated_at"].isnull())
    )
    loans.loc[mask, "terminated_at"] = loans.loc[mask]["loan_maturity_date"]

    # We set the end of the loan as the minimum between the termination date and
    # the reimbursment case.
    # - When the loan is reimbursed in due times, the end date is the date of
    #   reimbursment
    # - When the loan is not reimbursed in due times, it is first terminated, then
    #   might be reimbursed. Therefore the end date is the termination date.
    # - On going loans don't have an end date (not terminated, neither reimbursed yet).
    loans["loan_end_date"] = (
        loans[["terminated_at", "loan_reimbursed_date"]].min(axis=1)
    )

    loans["loan_duration"] = (
        loans["loan_end_date"] - loans["loan_created_date"]
    ).dt.days

    loans["raw_maturity_date"] = loans["loan_maturity_date"]
    loans["loan_maturity_date"] = (
        loans["loan_created_date"] + pd.Timedelta(days=TC_LIMIT)
    )
    loans["is_default"] = loans["termination_reason"].notnull().astype("int32")
    loans["is_ongoing"] = (
        loans["terminated_at"].isnull() & loans["loan_reimbursed_date"].isnull()
    )
    
    loans["risks"] = loans["termination_reason"].apply(_gather_risks) 
    loans.loc[loans["is_ongoing"], "risks"] = Risks.on_going.value
    
    # A loan is reimbursed in times when it is not on going, nor default.
    reimbursed_mask = (~loans["is_ongoing"]) & (~loans["is_default"])
    loans.loc[reimbursed_mask, "risks"] = Risks.reimbursed.value

    # Gather all defaults into a single category
    loans["single_risks"] = loans["risks"].replace(
        [
            Risks.maturity_reached.value,
            Risks.car_sold_np.value,
            Risks.dd_overdue.value,
            Risks.audit_overdue.value,
            Risks.other_default.value,
        ],
        value=SingleRisks.default.value,
    )

    _utils.check_no_duplicate_id(loans, id_col="carloan_id", name="loans")
    # TODO add check lower_or_equal on dates

    cols = [
        "carloan_id", "borrower_id", "collateral_id", "is_default", "is_ongoing",
        "risks", "single_risks", "loan_duration", "loan_created_date",
        "loan_end_date", "loan_maturity_date", "loan_reimbursed_date",
        "terminated_at", "loan_state", "termination_reason", "raw_termination_reason",
        "raw_maturity_date",
    ]
    return loans[cols]


def _get_car_loan_status():
    missing_reimbursement = db.DBSourceRisk().fetch(
        "SELECT * FROM missing_reimbursement"
    )

    carloan_status = db.DBSourceWH().fetch("SELECT * FROM car_loan_status")

    # Gather the missing reimbursment dates on car_loan_status by first merging
    # both table, then combining the reimbursment columns.
    loans = carloan_status.merge(
        missing_reimbursement[["carloan_id", "Car reimbursed date"]],
        on="carloan_id",
        how="left",
    ).drop_duplicates(subset="carloan_id")

    loans["Car reimbursed date"] = pd.to_datetime(
        loans["Car reimbursed date"], format="%d/%m/%Y %H:%M"
    )
    loans["loan_reimbursed_date"] = loans["loan_reimbursed_date"].combine_first(
        loans["Car reimbursed date"]
    )
    
    loans.drop(columns=["Car reimbursed date"], inplace=True)

    # Remove the timezone UTC.
    for col in "loan_reimbursed_date", "loan_created_date", "loan_maturity_date":
        loans[col] = loans[col].dt.tz_localize(None)

    loans = loans.rename(columns={"car_collateral_id": "collateral_id"})

    return loans


def _get_car_loans():
    names = {
        "id": "carloan_id",
        "terminationreason": "termination_reason",
        "terminatedat": "terminated_at",
    }
    car_loans = db.DBSourceWH().fetch(
        "SELECT * FROM cars_carloans", columns_renaming=names
    )

    # Aggregate terminationreason with typos.
    car_loans["raw_termination_reason"] = car_loans["termination_reason"]
    car_loans["termination_reason"] = (
        car_loans["termination_reason"]
        .replace("", None)
        .replace(
            [
                '.*(maturity|Maturity).*', '.*imit.*reached.*',
                '.*(due date|DUE DATE).*', '.*(loan|LOAN).*(due|DUE).*',
                'overdue', '.*Dealer has defaulted on loan.*',
            ],
            value=Risks.maturity_reached.value,
            regex=True
        )
        .replace(
            ".*(audit|Audit).*",
            value=Risks.audit_overdue.value,
            regex=True,
        )
        .replace(
            ".*diligence.*",
            value=Risks.dd_overdue.value,
            regex=True,
        )
        .replace(
            "collateral sold",
            value=Risks.car_sold_np.value,
        )
        .replace(
            '.*(reimbursment|REIMBURSMENT).*(requested|REQUESTED).*',
            value="Reimbursment Requested",
            regex=True,
        )
        .replace(
            '.*stock financing.*',
            value="Stock financing for reimbursment",
            regex=True,
        )
    )

    car_loans['terminated_at'] = car_loans['terminated_at'].dt.tz_localize(None) 

    return car_loans


def _gather_risks(x):
    if x is None:
        return None
    elif x in [
        Risks.audit_overdue,
        Risks.dd_overdue,
        Risks.maturity_reached,
        Risks.car_sold_np,
    ]:
        return x
    else:
        return Risks.other_default.value
    