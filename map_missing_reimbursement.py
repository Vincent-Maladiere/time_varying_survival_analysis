"""This script is a one-off patch to bring the missing reimbursement
dates from the Airtable Cars table to the car_loans_status table
in the Warehouse.

Running this script will create a CSV in the data directory.
"""
import pandas as pd

from credit_risk_models.risk_model_survival_analysis.db import DBSourceWH


def main():
    matched = map_missing_reimbursement_to_airtable()
    matched.to_csv("data/matched_airtable_loans.csv", index=False)


def map_missing_reimbursement_to_airtable():
    """Match loans from Airtable to loans in the warehouse \
        whose reimbursed date is missing.

    Note that 17 out of 403 loans are still unmatched.

    Returns
    -------
    matched : pandas.DataFrame, whose columns are:
    - Car ID : str, unique ID from Airtable Cars
    - carloan_id : str, unique ID from car_loan_status
    - reimbursed_date : str, from Airtable Cars
    """
    missing_loans = _get_missing_car_loans()
    airtable_loans = _get_airtable_loans()

    matched = _match(missing_loans, airtable_loans)

    return matched.drop_duplicates()


def _get_missing_car_loans():
    """Fetch the loans with missing reimbursement \
        from the Warehouse table car_loan_status.
    """
    loans_st = DBSourceWH().fetch("SELECT * from car_loan_status")
    mask = (
        (loans_st['loan_state'] == 100)
        & (loans_st['loan_reimbursed_date'].isna())
    )
    missing = loans_st.loc[mask].reset_index(drop=True)

    missing["price"] = (
        missing["loan_principal_amount"].astype(float).map('€{:.2f}'.format)
    )

    bad_price_id = 'carloan_7f025ff972d540c2ad515fc64010c7cd'
    missing.loc[missing["carloan_id"] == bad_price_id, "price"] = "€47500.00"

    return missing


def _get_airtable_loans():
    """Fetch airtable loans using a static CSV downloaded \
        from Airtable Cars.
    """
    cars = pd.read_csv("data/airtable_Infinit_Cars.csv", low_memory=False)
    cars = cars.loc[cars['Back Office Status'] == '🤑 Reimbursed']

    return cars


def _match(missing_loans, airtable_loans):
    """Merge missing loans to airtables using multiple columns as keys.
    """
    # First, we use 5 columns to match both tables. During this operation,
    # me must avoid any duplicate, i.e. all matches have to be unique.
    cars_keys = [
        'Legal company number (from Master Dealer)',
        "Make (from Car requests_Automated)",
        'Model (from Car requests_Automated)',
        'Vin (from CarId) (from Car requests_Automated)',
        "Financed Amount",
    ]
    missing_keys = [
        'company_registration_number',
        "car_make",
        "car_model",
        'car_vin',
        'price',
    ]

    first_matched = missing_loans.merge(
        airtable_loans,
        left_on=missing_keys,
        right_on=cars_keys,
        how='inner',
    )

    # Then, we select the Warehouse loans that couldn't be match.
    matched_ids = first_matched["carloan_id"].tolist()
    mask = ~missing_loans["carloan_id"].isin(matched_ids)
    still_missing = missing_loans.loc[mask]

    # We perform a new matching tentative, this time excluding
    # the legal company number, which can be None for some entries.
    missing_keys = [
        "car_make",
        "car_model",
        'car_vin',
        "price"
    ]
    car_keys = [
        "Make (from Car requests_Automated)",
        'Model (from Car requests_Automated)',
        'Vin (from CarId) (from Car requests_Automated)',
        "Financed Amount",
    ]

    second_matched = still_missing.merge(
        airtable_loans,
        left_on=missing_keys,
        right_on=car_keys,
        how="inner",
    )

    # Finally, we concatenate vertically both matches.
    cols = ["Car ID", "carloan_id", "Car reimbursed date"]
    matched = pd.concat([first_matched, second_matched], axis=0)

    return matched[cols]


def test_duplicate_matches():
    """Check 1:1 relationship between matched warehouse and airtable loans.
    """
    df = map_missing_reimbursement_to_airtable()

    # Check that each airtable loan is linked to a single warehouse loan.
    group = df.groupby(["Car ID"]).agg(somelist=("carloan_id", set))
    group["n_duplicate"] = group["somelist"].str.len()
    
    assert group["n_duplicate"].max() == 1
    
    # Check that each warehouse loan is linked to a single airtable loan.
    group = df.groupby(["carloan_id"]).agg(somelist=("Car ID", set))
    group["n_duplicate"] = group["somelist"].str.len()
    
    assert group["n_duplicate"].max() == 1


def test_missing_matches():
    """Check that 17 warehouse loans are still unmatched to Airtable.
    """
    df = map_missing_reimbursement_to_airtable()
    matched_ids = df["carloan_id"].unique().tolist()

    missing_loans = _get_missing_car_loans()
    mask = ~missing_loans["carloan_id"].isin(matched_ids)
    missing_loans = missing_loans.loc[mask]

    assert missing_loans.shape[0] == 17


if __name__ == "__main__":
    main()