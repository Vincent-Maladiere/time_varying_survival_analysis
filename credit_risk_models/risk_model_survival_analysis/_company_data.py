import pandas as pd

from . import db


def get_company_data():
    """
    Returns
    -------
    dataframe : pandas.DataFrame, whose columns are:
    - borrower_id : str
    - company_name : str
    - company_registration_number : str
    - country_code : str
    - credit_limit : int32
        Warning: There is no historic data of the credit limit in the warehouse.
        To trace back the evolution of updates, we have to look at Airtable.
        This might concern up to 10 or 20 customers.
    - owner_age_year : int32
    - n_days_since_founded : int32
    """
    company = _get_company_data()
    credit = _get_credit_limit()
    owner = _get_owner()
    commercial_partner = _get_commercial_partner()

    company = (
        company.merge(credit, on="borrower_id")
        .merge(owner, on="borrower_id")
        .merge(commercial_partner, on="borrower_id")
    )

    return company


def _get_company_data():
    columns_renaming = {
        "id": "borrower_id",
        "companyname": "company_name",
        "companyregistrationnumber": "company_registration_number",
        "countrycode": "country_code",
        "foundingdate": "founding_date",
    }
    query = "SELECT * FROM cars_companies"
    companies = db.DBSourceWH().fetch(query, columns_renaming)

    companies = companies.replace("-", None)
    
    # Create n_days_since_founded
    companies["founding_date"] = pd.to_datetime(
        companies["founding_date"], format="%Y-%m-%d", errors="coerce",
    )
    companies = companies.loc[companies["founding_date"].notnull()]
    
    # FIXME: Compute n_days_since_founded using the loan creation date instead of "now".
    companies["n_days_since_founded"] = (
        pd.Timestamp.today() - companies["founding_date"]
    ).dt.days.astype("int32")
    companies.drop(columns="founding_date", inplace=True)

    return companies


def _get_owner():
    columns_renaming = {
        "id": "borrower_id",
        "ownerpersonaldata_birthdate": "owner_birthdate",
    }
    query = "SELECT * from plafond_companies"
    owner = db.DBSourceWH().fetch(query, columns_renaming)

    owner["owner_birthdate"] = pd.to_datetime(
        owner["owner_birthdate"], format="%Y-%m-%d", errors="coerce"
    )
    # FIXME: Compute owner age using the loan creation date instead of "now".
    owner["owner_age_year"] = (
        (pd.Timestamp.today() - owner["owner_birthdate"])
    .dt.days // 365).astype("int32")

    owner.drop(columns=["owner_birthdate"], inplace=True)

    return owner


def _get_credit_limit():
    columns_renaming = {
        "companyid": "borrower_id",
        "grantedamount_amount": "credit_limit",
    }
    query = "SELECT * FROM plafond_companyplafondledger"
    credit = db.DBSourceWH().fetch(query, columns_renaming)

    credit["credit_limit"] = credit["credit_limit"].astype("int32")

    return credit


def _get_commercial_partner():
    query = """
        SELECT
            companyid AS borrower_id,
            commercialpartner AS commercial_partner
        FROM plafond_plafonds
    """
    plafond = db.DBSourceWH().fetch(query)
    return plafond