import click
import os
from datetime import datetime as dt
from zenml.logger import get_logger

from pipelines import training_pipeline, deployment_pipeline


logger = get_logger(__name__)


@click.command(
    help="""
Entry point for running pipelines.
"""
)
@click.option(
    "--no-cache",
    is_flag=True,
    default=False,
    help="Disable caching for the pipeline run.",
)
@click.option(
    "--fail-on-accuracy-quality-gates",
    is_flag=True,
    default=False,
    help="Whether to fail the pipeline run if the model evaluation step "
    "finds that the model is not accurate enough.",
)
@click.option(
    "--only-deployment",
    is_flag=True,
    default=False,
    help="Whether to run only deployment pipeline.",
)
@click.option(
    "--test-size",
    default=0.2,
    type=click.FloatRange(0.0, 1.0),
    help="Proportion of the dataset to include in the test split.",
)
@click.option(
    "--min-train-accuracy",
    default=0.8,
    type=click.FloatRange(0.0, 1.0),
    help="Minimum training accuracy to pass to the model evaluator.",
)
@click.option(
    "--min-test-accuracy",
    default=0.8,
    type=click.FloatRange(0.0, 1.0),
    help="Minimum test accuracy to pass to the model evaluator.",
)
def main(
    no_cache: bool = False,
    fail_on_accuracy_quality_gates: bool = False,
    only_deployment: bool = False,
    test_size: float = 0.2,
    min_train_accuracy: float = 0.8,
    min_test_accuracy: float = 0.8,
) -> None:

    pipeline_args = {}
    if no_cache:
        pipeline_args["enable_cache"] = False

    if only_deployment:

        # run deployment pipeline
        pipeline_args["config_path"] = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "configs",
            "deployer_config.yaml",
        )
        pipeline_args["run_name"] = (
            f"e2e_iris_deployment_run_{dt.now().strftime('%Y_%m_%d_%H_%M_%S')}"
        )
        deployment_pipeline.with_options(**pipeline_args)()
        logger.info("Deployment pipeline finished successfully!")

        return None

    # run training pipeline
    run_args_train = {
        "test_size": test_size,
        "min_train_accuracy": min_train_accuracy,
        "min_test_accuracy": min_test_accuracy,
        "fail_on_accuracy_quality_gates": fail_on_accuracy_quality_gates,
    }
    pipeline_args["config_path"] = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "configs",
        "train_config.yaml",
    )
    pipeline_args["run_name"] = (
        f"e2e_iris_training_run_{dt.now().strftime('%Y_%m_%d_%H_%M_%S')}"
    )
    training_pipeline.with_options(**pipeline_args)(**run_args_train)
    logger.info("Training pipeline finished successfully!")

    return None


if __name__ == "__main__":
    main()
