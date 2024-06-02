from zenml import Model, get_step_context, step
from zenml.logger import get_logger

from utils.promote_in_model_registry import promote_in_model_registry

logger = get_logger(__name__)


@step(enable_cache=False)
def promote_with_metric_compare(
    latest_metric: float,
    current_metric: float,
    mlflow_model_name: str,
    target_env: str,
) -> None:

    should_promote = True

    # Get model version numbers from Model Control Plane
    latest_version = get_step_context().model
    current_version = Model(name=latest_version.name, version=target_env)

    current_version_number = current_version.number

    if current_version_number is None:
        logger.info("No current model version found - promoting latest")
    else:
        logger.info(
            f"Latest model metric={latest_metric:.6f}\n"
            f"Current model metric={current_metric:.6f}"
        )
        if latest_metric >= current_metric:
            logger.info(
                "Latest model version outperformed current version - promoting latest"
            )
        else:
            logger.info(
                "Current model version outperformed latest version - keeping current"
            )
            should_promote = False

    if should_promote:
        # Promote in Model Control Plane
        model = get_step_context().model
        model.set_stage(stage=target_env, force=True)
        logger.info(f"Current model version was promoted to '{target_env}'.")

        # Promote in Model Registry
        latest_version_model_registry_number = latest_version.run_metadata[
            "model_registry_version"
        ].value
        if current_version_number is None:
            current_version_model_registry_number = latest_version_model_registry_number
        else:
            current_version_model_registry_number = current_version.run_metadata[
                "model_registry_version"
            ].value
        promote_in_model_registry(
            latest_version=latest_version_model_registry_number,
            current_version=current_version_model_registry_number,
            model_name=mlflow_model_name,
            target_env=target_env.capitalize(),
        )
        promoted_version = latest_version_model_registry_number
    else:
        promoted_version = current_version.run_metadata["model_registry_version"].value

    logger.info(
        f"Current model version in `{target_env}` is `{promoted_version}` registered in Model Registry"
    )

    return None
