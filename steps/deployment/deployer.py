from typing import Optional
from typing_extensions import Annotated
from zenml import ArtifactConfig, get_step_context, step
from zenml.client import Client
from zenml.integrations.mlflow.services.mlflow_deployment import (
    MLFlowDeploymentService,
)
from zenml.integrations.mlflow.steps.mlflow_deployer import (
    mlflow_model_registry_deployer_step,
)
from zenml.logger import get_logger

from utils.helper import dump_yaml_file

logger = get_logger(__name__)


@step(enable_cache=False)
def deployment_deploy() -> Annotated[
    Optional[MLFlowDeploymentService],
    ArtifactConfig(name="mlflow_deployment", is_deployment_artifact=True),
]:

    if Client().active_stack.orchestrator.flavor == "local":
        model = get_step_context().model

        # deploy predictor service
        deployment_service = mlflow_model_registry_deployer_step.entrypoint(
            registry_model_name=model.name,
            registry_model_version=model.run_metadata["model_registry_version"].value,
            replace_existing=True,
        )
        logger.info("Dumping deployment service config in 'configs' folder...")
        deployment_service_config = deployment_service.config.dict()
        dump_yaml_file(
            data={"config": deployment_service_config},
            relative_path="configs/deployment_service_config.yaml",
        )
        logger.info(f"Deployment service config: {deployment_service_config}")

        deployment_service.start(timeout=60)
    else:
        logger.warning("Skipping deployment as the orchestrator is not local.")
        deployment_service = None

    return deployment_service
