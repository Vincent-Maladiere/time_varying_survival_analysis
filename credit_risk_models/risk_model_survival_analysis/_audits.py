import pandas as pd

from . import db
from . import _utils


def get_audits():
    """Fetch audits.

    The primary id of this table is audit_id.
    Each car loan has at most one due diligence, and can have multiple audits.

    An audit is either:
    - on going (no submission or cancellation date)
    - cancelled (it has a cancellation date)
    - submitted and accepted (it has a submitted date and is not rejected)
    - submitted and rejected (it has a submitted date and is rejected)

    Returns
    -------
    audit : pd.DataFrame, whose columns are:
        - audit_id : str
            The primary id.
        - carloan_id : str
            The id of a loan.
        - collateral_id : str
            The id of the collateral.
        - audit_scheduled_for_from : datetime
            The start of the range in which the audit is due.
        - audit_due_date : datetime 
            The end of the range in which the audit is due.
        - audit_dpd14_date : datetime
            The audit overdue date, defined as due date + 14 days.
        - audit_cancellation_date : datetime
            The date when the audit has been cancelled.
        - audit_submission_date : datetime
            The date when the audit has been submitted.
        - audit_approval_date : datetime
        - audit_end_date : datetime
            audit_submission_taken_at or cancellation_taken_at (or NaT)
        - audit_approved : bool
        - audit_rejected : bool
        - audit_cancelled : bool
        - audit_state : int
            The current state of the audit.
            0 is created, 1 is scheduled, 20 is submitted, 100 is approved,
            200 is rejected and 300 is cancelled.
            Warning, this value can change and only reflects the last state.
    """
    query = """
        select
            id as audit_id,
            loanid as carloan_id,
            collateralid as collateral_id,
            scheduledfor_from as audit_scheduled_for_from,
            scheduledfor_to as audit_due_date,
            cancellation_takenat as audit_cancellation_date,
            submission_takenat as audit_submission_date, 
            approval_result as audit_approval_result,
            approval_takenat as audit_approval_date,
            state as audit_state
        from cars_carcollateralaudits
    """
    audits = db.DBSourceWH().fetch(query)

    # Remove the timezone
    cols = "audit_submission_date", "audit_cancellation_date", "audit_approval_date"
    for col in cols:
        audits[col] = audits[col].dt.tz_localize(None)

    for col in "audit_due_date", "audit_scheduled_for_from":
        audits[col] = pd.to_datetime(audits[col])
    
    audits["audit_dpd14_date"] = audits["audit_due_date"] + pd.Timedelta(days=14)

    audits["audit_end_date"] = audits["audit_submission_date"].combine_first(
        audits["audit_cancellation_date"]
    )

    # 100% overlap with the loan_state.
    audits["audit_approved"] = audits["audit_approval_result"] == True
    audits["audit_rejected"] = audits["audit_approval_result"] == False
    
    # Note: 11% of audits state = 300 don't have a cancellation date.
    audits["audit_cancelled"] = audits["audit_state"] == 300

    _utils.check_no_duplicate_id(audits, id_col="audit_id", name="audits")

    cols = [
        "audit_id", "carloan_id", "collateral_id",
        "audit_scheduled_for_from", "audit_due_date", "audit_dpd14_date",
        "audit_cancellation_date", "audit_submission_date", "audit_approval_date",
        "audit_end_date", "audit_approved", "audit_rejected", "audit_cancelled",
        "audit_state",
    ]
    return audits[cols]


def get_dd():
    """Fetch due diligence.

    Returns
    -------
    dd_id : str
    collateral_id : str
    dd_due_date : datetime
    dd_submission_taken_at : datetime
    car_source : int
    dd_state : int
    """
    query = """
        select
            id as dd_id,
            collateralid as collateral_id,
            createdat as dd_created_at,
            duedate as dd_due_date,
            submission_takenat as dd_submission_taken_at,
            carsource_companyinfo_companytype as car_source,
            state as dd_state,
            approved as dd_approved

        from cars_carcollateralduediligences
    """
    dd = db.DBSourceWH().fetch(query)

    # Remove the timezone
    for col in "dd_submission_taken_at", "dd_due_date":
        dd[col] = dd[col].dt.tz_localize(None)

    # FIXME
    # Similar compute for dd overdue, except there is almost no on-going loans
    # # with pending due diligence.
    # loan["dd_overdue"] = (
    #     (loan["dd_due_date"] < loan["dd_submission_taken_at"])
    #     & (loan["car_source"] == 0)
    # )

    _utils.check_no_duplicate_id(dd, id_col="dd_id", name="due_diligence")

    cols = [
        "dd_id",
        "collateral_id",
        "dd_created_at",
        "dd_due_date",
        "dd_submission_taken_at",
        "car_source",
        "dd_state",
        "dd_approved",
    ]
    return dd[cols]


