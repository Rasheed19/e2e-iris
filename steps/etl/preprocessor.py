import pandas as pd
from typing import Annotated
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from zenml import step

from utils.helper import DataFrameCaster


@step(enable_cache=False)
def data_preprocessor(
    dataset_trn: pd.DataFrame,
    dataset_tst: pd.DataFrame,
    target: str,
) -> tuple[
    Annotated[pd.DataFrame, "dataset_trn"],
    Annotated[pd.DataFrame, "dataset_tst"],
    Annotated[Pipeline, "preprocess_pipeline"],
]:

    y_train = dataset_trn[target].values
    y_test = dataset_tst[target].values
    dataset_trn = dataset_trn.drop(columns=[target])
    dataset_tst = dataset_tst.drop(columns=[target])

    preprocess_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer()),
            ("scaler", StandardScaler()),
            ("cast", DataFrameCaster(dataset_trn.columns)),
        ]
    )
    dataset_trn = preprocess_pipeline.fit_transform(dataset_trn)
    dataset_tst = preprocess_pipeline.transform(dataset_tst)

    dataset_trn[target] = y_train
    dataset_tst[target] = y_test

    return dataset_trn, dataset_tst, preprocess_pipeline
