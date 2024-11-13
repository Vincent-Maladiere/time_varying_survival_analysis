import pandas as pd

from . import db


def get_automative():
    """
    Returns
    -------
    loan : pd.Dataframe whose columns are:
        - carloan_id : str
        - borrower_id : str
        - collateral_id : str
        - loan_created_at : datetime
        - loan_amount : float32
        - currency : str
        - car_make : str
        - car_model : str
        - car_transmission_type : bool
        - car_source : int
            0 is online, 2 is stock finance
    """
    loan = _get_loan()
    car = _get_car()
    collateral = _get_collateral()

    loan = (
        loan.merge(car, on="carloan_id", how="left")
            .merge(collateral, on="collateral_id", how="left")
    )
    return loan.drop_duplicates()


def _get_loan():
    query = "SELECT * FROM cars_carloans"
    columns_renaming = {
        "id": "carloan_id",
        "borrowerid": "borrower_id",
        "collateralid": "collateral_id",
        "createdat": "created_at",
        "principal_amount": "loan_amount",
        "principal_currency": "currency",
    }
    loan = db.DBSourceWH().fetch(query, columns_renaming)

    # Created in days not in ns timezone
    loan["created_at"] = (
        loan["created_at"].dt.round("h").dt.tz_localize(None)
    )
    loan["loan_amount"] = loan["loan_amount"].astype("float32")

    return loan


def _get_collateral():
    query = "SELECT * from cars_carcollateralduediligences"
    columns_renaming = {
        "collateralid": "collateral_id",
        "carsource_companyinfo_companytype": "car_source",
    }
    collateral = db.DBSourceWH().fetch(query, columns_renaming)
    return collateral


def _get_car():
    query = "SELECT * from car_loan_status"
    cols = [
        "carloan_id",
        "car_make",
        "car_model",
        "car_transmission_type",
        "car_first_registration_date",
    ]
    car = db.DBSourceWH().fetch(query)[cols]

    # FIXME: How do we get the age of the car?
    # We need to find this information in cars_cars, but there isn't a good
    # key to match on tables like cars_carloan.
    # The column "cars_cars.vin" has 70% of missing data.
    #
    # car["car_first_registration_date"] = pd.to_datetime(
    #     car["car_first_registration_date"], format="%Y-%m-%d", errors="coerce",
    # )
    # car["car_age_days"] = (
    #     pd.Timestamp.now() - car["car_first_registration_date"]
    # ).dt.days

    return car

