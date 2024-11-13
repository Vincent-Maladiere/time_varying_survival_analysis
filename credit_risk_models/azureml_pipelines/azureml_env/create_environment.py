# %%
import sys
import os
from azure.ai.ml.entities import Environment, BuildContext
from credit_risk_models.azure_credentials_keyvault.ml_client import get_ml_client


def create_azureml_environment(
    name: str,
    description: str,
    version: str,
    build_path: str,
    dockerfile_path: str = "Dockerfile",
) -> Environment:
    """
    Create or update an Azure ML environment.

    This function creates a new Azure ML environment or updates an existing one
    using the specified parameters. It uses a Dockerfile to define the environment.

    Args:
        name (str): The name of the environment.
        description (str): A brief description of the environment.
        dockerfile_path (str): The path to the Dockerfile, relative to the build_path.
        version (str, optional): The version of the environment.
        build_path (str, optional): The path to the directory containing the Dockerfile.
                                    Defaults to "../../" (assuming we're in credit_risk_models/azureml_pipelines).

    Returns:
        Environment: The created or updated Azure ML environment.

    Raises:
        Exception: If there's an error in creating or updating the environment.
    """
    try:
        # Create the Azure ML client
        ml_client = get_ml_client()

        # Define the environment
        azureml_env = Environment(
            name=name,
            description=description,
            build=BuildContext(
                path=build_path,
                dockerfile_path=dockerfile_path,
            ),
            version=version,
        )

        # Create or update the environment
        created_env = ml_client.environments.create_or_update(azureml_env)

        print(f"Environment created with name: {created_env.name}")
        print(f"Environment version: {created_env.version}")

        return created_env

    except Exception as e:
        print(f"Error creating environment: {str(e)}")
        raise


# %%

# to create a new environment, run this file
if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "..", "..", ".."))
    env = create_azureml_environment(
        name="credit_risk_surv_env",
        description="Environment for credit risk modeling",
        version="0.5",
        dockerfile_path="Dockerfile",
        build_path=project_root,
    )

# %%
