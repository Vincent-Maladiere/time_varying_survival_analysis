from azure.identity import ManagedIdentityCredential, DefaultAzureCredential
from azure.keyvault.secrets import SecretClient
from credit_risk_models.azureml_pipelines.azureml_config import VAULT_URL, CLIENT_ID


def get_db_credentials(vault_url: str = VAULT_URL):
    """Retrieve database credentials from Azure Key Vault"""

    # Try DefaultAzureCredential first (faster for local development)
    try:
        credential = DefaultAzureCredential()
        secret_client = SecretClient(vault_url=vault_url, credential=credential)
        # Test if it works
        secret_client.get_secret("PSQL-HOST")
        print("Using DefaultAzureCredential")
    except:
        # Fallback to ManagedIdentityCredential (for AzureML)
        credential = ManagedIdentityCredential(client_id=CLIENT_ID)
        secret_client = SecretClient(vault_url=vault_url, credential=credential)
        print("Using ManagedIdentityCredential")

    # Get secrets
    credentials = {
        "host": secret_client.get_secret("PSQL-HOST").value,
        "port": secret_client.get_secret("PSQL-PORT").value,
        "dbname": secret_client.get_secret("PSQL-DB").value,
        "etl_user": secret_client.get_secret("PSQL-ETL-USER").value,
        "etl_password": secret_client.get_secret("PSQL-ETL-PASSWORD").value,
        "risk_user": secret_client.get_secret("PSQL-RISK-USER").value,
        "risk_password": secret_client.get_secret("PSQL-RISK-PASSWORD").value,

    }

    return credentials
