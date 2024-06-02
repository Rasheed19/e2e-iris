from zenml import pipeline

from steps import deployment_deploy


@pipeline
def deployment_pipeline() -> None:
    deployment_deploy()

    return None
