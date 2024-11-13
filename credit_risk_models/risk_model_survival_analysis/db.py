from time import time
import pandas as pd
import psycopg2
from sqlalchemy import create_engine
from dataclasses import dataclass, asdict

from credit_risk_models.azure_credentials_keyvault.db_credentials import (
    get_db_credentials,
)
from . import _logs


def _register_age_type_psycopg():
    """Prevent psycopg from raising an error when the age is out of bound.

    This register a function as a new format for date parsing.

    https://stackoverflow.com/questions/40184556/psycopg2-erroring-out-when-reading-dates-out-of-range
    """
    date2str = psycopg2.extensions.new_type(
        psycopg2.extensions.DATE.values,
        "DATE2STR",
        lambda value, curs: str(value) if value is not None else None,
    )
    psycopg2.extensions.register_type(date2str)


@dataclass
class DBSource(_logs.LogsMixin):
    """A simple PostgreSQL connector for dataframes."""

    host: str
    port: str
    dbname: str
    user: str
    password: str

    def __post_init__(self):
        self.conn = psycopg2.connect(**asdict(self))
        self.engine = create_engine(
            f"postgresql+psycopg2://{self.user}:{self.password}"
            f"@{self.host}:{self.port}/{self.dbname}"
        )
        _register_age_type_psycopg()

    def fetch(self, query, columns_renaming=None):
        """Use a SQL query to fetch a dataframe.

        Parameters
        ----------
        query : str
            The SQL query.

        columns_renaming : dict, default=None
            The mapping to rename and select columns, to keep
            the SQL query as minimal as possible.

        Returns
        -------
        df : pandas.DataFrame
        """
        start = time()
        with self.conn.cursor() as cursor:
            cursor.execute(query)
            rows = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description]
        end = time()
        msg = f"-- Took {end-start:.1f}s"
        self._log_info("fetched", query, msg)

        df = pd.DataFrame(rows, columns=columns)

        if columns_renaming is not None:
            cols = list(columns_renaming.values())
            df = df.rename(columns=columns_renaming)[cols]

        return df

    def write_df(
        self,
        dataframe,
        table_name,
        schema=None,
        if_exists="replace",
        index=False,
        dtype=None,
    ):
        """Write a dataframe to a table.

        Use sqlalchemy and pandas.DataFrame.to_sql.

        Parameters
        ----------
        dataframe : pandas.DataFrame
            The table to write.

        table_name : str
            Name of SQL table.

        schema : str
            Specify the schema (if database flavor supports this).
            If None, use default schema.

        if_exists : {'replace', 'append', 'fail'}, default='replace'
            How to behave if the table already exists.
            - fail: Raise a ValueError.
            - replace: Drop the table before inserting new values.
            - append: Insert new values to the existing table.

        index : bool, default=False
            Write DataFrame index as a column.
        """
        n_rows = dataframe.to_sql(
            table_name,
            schema=schema,
            con=self.engine,
            if_exists=if_exists,
            index=index,
            dtype=dtype,
        )
        path = f"{schema}.{table_name}" if schema is not None else table_name
        print(dataframe.dtypes)
        self._log_info("wrote", path)
        return n_rows

    def delete(self, table_name, schema=None):
        """Delete a table.

        Parameters
        ----------
        table_name : str
            Name of SQL table.

        schema : str, default=None
            Specify the schema (if database flavor supports this).
            If None, use default schema.
        """
        path = f"{schema}.{table_name}" if schema is not None else table_name
        with self.conn.cursor() as cursor:
            cursor.execute(f"DROP TABLE {path}")
        self._log_info("deleted", path)

    def _log_info(self, action, path, extra=""):
        print(f"{self.__class__.__name__} {action} {path} {extra}")

    def _fetch_credentials(self):
        credentials = get_db_credentials()
        return {
            "host": credentials["host"],
            "port": credentials["port"],
            "dbname": credentials["dbname"],
            "user": credentials[self.user_key],
            "password": credentials[self.password_key],
        }


class DBSourceWH(DBSource):
    """The main Warehouse connector, with read only rights."""

    user_key = "etl_user"
    password_key = "etl_password"

    def __init__(self):
        credentials = self._fetch_credentials()
        super().__init__(**credentials)


class DBSourceRisk(DBSource):
    """The risk Warehouse connector, with writing rights."""

    user_key = "risk_user"
    password_key = "risk_password"

    def __init__(self):
        credentials = self._fetch_credentials()
        super().__init__(**credentials)
