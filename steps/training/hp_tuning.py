from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.base import ClassifierMixin
from typing_extensions import Annotated
from zenml import step, log_artifact_metadata
from zenml.logger import get_logger
import pandas as pd

logger = get_logger(__name__)


@step(enable_cache=False)
def hyperparameter_tuner(
    param_grid: dict,
    dataset_trn: pd.DataFrame,
    dataset_tst: pd.DataFrame,
    target: str,
) -> Annotated[ClassifierMixin, "hp_result"]:

    logger.info("Running Hyperparameter tuning...")

    grid_search = GridSearchCV(
        estimator=GradientBoostingClassifier(),
        param_grid=param_grid,
        scoring="accuracy",
        cv=3,
        refit=True,
    )

    X_train = dataset_trn.drop(columns=[target])
    y_train = dataset_trn[target]
    X_test = dataset_tst.drop(columns=[target])
    y_test = dataset_tst[target]

    grid_search.fit(X=X_train, y=y_train)
    test_score = accuracy_score(
        y_true=y_test,
        y_pred=grid_search.best_estimator_.predict(X_test),
    )

    log_artifact_metadata(
        metadata={
            "metric": {
                "cv_accuracy": grid_search.best_score_,
                "test_accuracy": test_score,
            }
        },
        artifact_name="hp_result",
    )

    return grid_search.best_estimator_
