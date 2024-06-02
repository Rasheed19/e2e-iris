from zenml import pipeline
from typing import Any

from steps import (
    data_loader,
    train_data_splitter,
    data_preprocessor,
    model_trainer,
    hyperparameter_tuner,
    model_evaluator,
    compute_performance_metrics_on_current_data,
    promote_with_metric_compare,
)


@pipeline
def training_pipeline(
    param_grid: dict[str, Any],
    target_env: str,
    test_size: float = 0.2,
    min_train_accuracy: float = 0.0,
    min_test_accuracy: float = 0.0,
    fail_on_accuracy_quality_gates: bool = False,
):

    dataset, target = data_loader()

    dataset_trn, dataset_tst = train_data_splitter(
        dataset=dataset,
        test_size=test_size,
    )

    dataset_trn, dataset_tst, preprocess_pipeline = data_preprocessor(
        dataset_trn=dataset_trn,
        dataset_tst=dataset_tst,
        target=target,
    )

    best_model = hyperparameter_tuner(
        param_grid=param_grid,
        dataset_trn=dataset_trn,
        dataset_tst=dataset_tst,
        target=target,
    )

    model = model_trainer(
        dataset_trn=dataset_trn,
        target=target,
        model=best_model,
    )

    model_evaluator(
        model=model,
        dataset_trn=dataset_trn,
        dataset_tst=dataset_tst,
        target=target,
        min_train_accuracy=min_train_accuracy,
        min_test_accuracy=min_test_accuracy,
        fail_on_accuracy_quality_gates=fail_on_accuracy_quality_gates,
    )

    latest_metric, current_metric = compute_performance_metrics_on_current_data(
        dataset_tst=dataset_tst,
        target_env=target_env,
        after=["model_evaluator"],
    )

    promote_with_metric_compare(
        latest_metric=latest_metric,
        current_metric=current_metric,
        target_env=target_env,
    )

    return None
