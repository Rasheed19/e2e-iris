import pandas as pd
from typing_extensions import Annotated
from zenml import step
from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import (
    MLFlowModelDeployer,
)
from zenml.integrations.mlflow.services import MLFlowDeploymentService
from zenml.client import Client


@step(enable_cache=False)
def prediction_service_loader(
    config: dict,
    running: bool = True,
) -> Annotated[MLFlowDeploymentService, "mlflow_deployment_service"]:

    # get the MLflow model deployer stack component
    model_deployer = MLFlowModelDeployer.get_active_model_deployer()

    # fetch existing services with same configuration as config
    existing_services = model_deployer.find_model_server(running=running, config=config)

    if not existing_services:
        raise RuntimeError(
            f"No MLflow prediction service deployed with config "
            f"'{config}' is currently running."
        )

    return existing_services[0]


@step(enable_cache=False)
def get_prediction(
    service: MLFlowDeploymentService,
    data: pd.DataFrame,
) -> Annotated[pd.DataFrame, "predictions"]:

    # get prediction service configuration
    service_config = service.config.dict()

    # get preprocess pipeline coressponding to the service version
    client = Client()
    preprocess_pipeline = client.get_artifact_version(
        name_id_or_prefix="preprocess_pipeline", version=service_config["model_version"]
    ).load()
    data = preprocess_pipeline.transform(data)

    prediction = service.predict(data)

    return prediction
