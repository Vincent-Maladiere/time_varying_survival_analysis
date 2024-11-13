import os
from azure.identity import DefaultAzureCredential, ManagedIdentityCredential
from azure.ai.ml import MLClient

from credit_risk_models.azureml_pipelines.azureml_config import (
    SUBSCRIPTION_ID,
    RESOURCE_GROUP,
    WORKSPACE_NAME,
    CLIENT_ID,
)


def get_ml_client():
    try:
        # Use ManagedIdentityCredential with the Client ID for user-assigned identity
        credential = ManagedIdentityCredential(client_id=CLIENT_ID)
        ml_client = MLClient(
            credential=credential,
            subscription_id=SUBSCRIPTION_ID,
            resource_group_name=RESOURCE_GROUP,
            workspace_name=WORKSPACE_NAME,
        )
        # Test the client
        ml_client.workspaces.get(WORKSPACE_NAME)
        print("Successfully authenticated with ManagedIdentityCredential")
        return ml_client
    except Exception as e:
        print(f"ManagedIdentityCredential failed: {str(e)}")

    try:
        # Fallback to DefaultAzureCredential
        credential = DefaultAzureCredential()
        ml_client = MLClient(
            credential=credential,
            subscription_id=SUBSCRIPTION_ID,
            resource_group_name=RESOURCE_GROUP,
            workspace_name=WORKSPACE_NAME,
        )
        # Test the client
        ml_client.workspaces.get(WORKSPACE_NAME)
        print("Successfully authenticated with DefaultAzureCredential")
        return ml_client
    except Exception as e:
        print(f"DefaultAzureCredential failed: {str(e)}")

    raise Exception(
        "Failed to authenticate with both ManagedIdentityCredential and DefaultAzureCredential"
    )
