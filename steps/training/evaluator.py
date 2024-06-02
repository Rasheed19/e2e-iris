import pandas as pd
import mlflow
from sklearn.base import ClassifierMixin
from sklearn.metrics import accuracy_score
from zenml import step
from zenml.client import Client
from zenml.logger import get_logger

logger = get_logger(__name__)

experiment_tracker = Client().active_stack.experiment_tracker


@step(experiment_tracker=experiment_tracker.name, enable_cache=False)
def model_evaluator(
    model: ClassifierMixin,
    dataset_trn: pd.DataFrame,
    dataset_tst: pd.DataFrame,
    target: str,
    min_train_accuracy: float = 0.0,
    min_test_accuracy: float = 0.0,
    fail_on_accuracy_quality_gates: bool = False,
) -> None:

    X_train = dataset_trn.drop(columns=[target])
    y_train = dataset_trn[target]
    X_test = dataset_tst.drop(columns=[target])
    y_test = dataset_tst[target]

    trn_acc = accuracy_score(y_true=y_train, y_pred=model.predict(X_train))
    logger.info(f"Train accuracy={trn_acc * 100:.2f}%")
    mlflow.log_metric("training_score", trn_acc)

    tst_acc = accuracy_score(y_true=y_test, y_pred=model.predict(X_test))
    logger.info(f"Test accuracy={tst_acc * 100:.2f}%")
    mlflow.log_metric("testing_accuracy_score", tst_acc)

    messages = []
    if trn_acc < min_train_accuracy:
        messages.append(
            f"Train accuracy {trn_acc * 100:.2f}% is below {min_train_accuracy * 100:.2f}% !"
        )
    if tst_acc < min_test_accuracy:
        messages.append(
            f"Test accuracy {tst_acc * 100:.2f}% is below {min_test_accuracy * 100:.2f}% !"
        )
    if fail_on_accuracy_quality_gates and messages:
        raise RuntimeError(
            "Model performance did not meet the minimum criteria:\n"
            + "\n".join(messages)
        )
    else:
        for message in messages:
            logger.warning(message)

    return
