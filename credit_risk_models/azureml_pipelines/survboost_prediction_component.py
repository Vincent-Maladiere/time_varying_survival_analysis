from azure.ai.ml import command
from azure.ai.ml import Input, Output


survboost_prediction_component = command(
    name="survboost_prediction",
    display_name="SurvivalBoost prediction",
    description="Predict using the SurvivalBoost",
    inputs={
        "model_name": Input(type="string"),
        "model_version": Input(type="string"),
        "prediction_table_name": Input(type="string"),
        "feat_imps_table_name": Input(type="string"),
        "dpd_limit": Input(type="integer", default=240),
    },
    outputs={},
    code="../../",  # The source folder of the component
    command=(
        # Debug information
        "echo 'Current directory:' && pwd && "
        "echo 'Directory contents:' && ls -la && "
        # Configure poetry
        "poetry config virtualenvs.create false && "
        
        # clone hazardous package & install
        "cd hazardous && pip install . && cd .. && "
        # Add hazardous package to the project
        "poetry add ./hazardous && "
        
        # Clean installation with debug output
        "poetry install --no-interaction --verbose && "
        # Verify installation
        "poetry run python -c 'import credit_risk_models; print(\"Package imported successfully\")' && "
        # Run inference        
        "poetry run python -m credit_risk_models.risk_model_survival_analysis.survboost_prediction \
        --model_name ${{inputs.model_name}} \
        --model_version ${{inputs.model_version}} \
        --prediction_table_name ${{inputs.prediction_table_name}} \
        --feat_imps_table_name ${{inputs.feat_imps_table_name}} \
        --dpd_limit ${{inputs.dpd_limit}}"
    ),
    environment="credit_risk_surv_env:0.5",
)

print(survboost_prediction_component)
