# %%
from datetime import datetime
from azure.ai.ml import dsl

from credit_risk_models.azureml_pipelines.survboost_prediction_component import (
    survboost_prediction_component
)
from credit_risk_models.azure_credentials_keyvault.ml_client import get_ml_client


def push_prediction_pipeline():
    pipeline_prediction_km = credit_risk_prediction_pipeline(
        model_name="SurvivalBoost",
        model_version="1",
        prediction_table_name="loan-default-risk-predictions-test",
        feat_imps_table_name="loan-default-risk-features-test",
        dpd_limit=240,
    )
    ml_client = get_ml_client()
    pipeline_job = ml_client.jobs.create_or_update(
        pipeline_prediction_km,
        experiment_name="credit-risk-prediction-survboost",
        force_rerun=True,
        tags={"last_updated": str(datetime.now())},
    )
    ml_client.jobs.stream(pipeline_job.name)


@dsl.pipeline(compute="test-risk-model-cluster")
def credit_risk_prediction_pipeline(
    model_name,
    model_version,
    prediction_table_name,
    feat_imps_table_name,
    dpd_limit,
):
    # Using feature_extraction_component like a python call with its own inputs.
    survboost_prediction_step = survboost_prediction_component(
        model_name=model_name,
        model_version=model_version,
        prediction_table_name=prediction_table_name,
        feat_imps_table_name=feat_imps_table_name,
        dpd_limit=dpd_limit,
    )
    return {}

# %%

push_prediction_pipeline()
# %%
