# How to work with AzureML 

## SETUP
### create the environment
- run the create_environment.py file to create the environment
- update the environment name, description, version in the create_environment.py file
- it will refer to the docker file in the root

### create the compute cluster
- for now, create the compute cluster manually in the AzureML studio

## PIPELINE
### data inputs
- can be a reference from AML Data Assets

### data outputs 
- intermediate outputs can be passed to the next step 

### model registry
- will depend on the type (TODO)

### components 
- feature_extraction_component.py : creates the features for the credit risk model
- training_component.py : trains the credit risk model
- batch_inference_component.py : does the batch inference for the credit risk model
- register components in AML for reuse (later)


### pipeline
- credit_risk_training_pipeline.py (for data extractionn + train steps)

