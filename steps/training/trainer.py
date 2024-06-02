import pandas as pd
from sklearn.base import ClassifierMixin
from typing_extensions import Annotated
import mlflow
from zenml import ArtifactConfig, get_step_context, step
from zenml.client import Client
from zenml.integrations.mlflow.experiment_trackers import (
    MLFlowExperimentTracker,
)
from zenml.integrations.mlflow.steps.mlflow_registry import (
    mlflow_register_model_step,
)
from zenml.logger import get_logger

logger = get_logger(__name__)

experiment_tracker = Client().active_stack.experiment_tracker

if not experiment_tracker or not isinstance(
    experiment_tracker, MLFlowExperimentTracker
):
    raise RuntimeError(
        "Your active stack needs to contain a MLFlow experiment tracker for "
        "this example to work."
    )


@step(experiment_tracker=experiment_tracker.name, enable_cache=False)
def model_trainer(
    dataset_trn: pd.DataFrame,
    target: str,
    model: ClassifierMixin,
    name: str,
) -> Annotated[
    ClassifierMixin,
    ArtifactConfig(name="model", is_model_artifact=True),
]:

    logger.info(f"Training model...")

    mlflow.sklearn.autolog()
    model.fit(X=dataset_trn.drop(columns=[target]), y=dataset_trn[target])

    # register mlflow model
    mlflow_register_model_step.entrypoint(
        model,
        name=name,
    )
    # keep track of mlflow version for future use
    model_registry = Client().active_stack.model_registry
    if model_registry:
        version = model_registry.get_latest_model_version(name=name, stage=None)
        if version:
            model_ = get_step_context().model
            model_.log_metadata({"model_registry_version": version.version})

    return model
