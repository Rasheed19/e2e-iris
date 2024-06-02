import pandas as pd
from typing_extensions import Annotated
from zenml import step
from sklearn.model_selection import train_test_split


@step(enable_cache=False)
def train_data_splitter(dataset: pd.DataFrame, test_size: float = 0.2) -> tuple[
    Annotated[pd.DataFrame, "dataset_trn"],
    Annotated[pd.DataFrame, "dataset_tst"],
]:

    dataset_trn, dataset_tst = train_test_split(
        dataset,
        test_size=test_size,
        random_state=42,
        shuffle=True,
    )
    dataset_trn = pd.DataFrame(dataset_trn, columns=dataset.columns)
    dataset_tst = pd.DataFrame(dataset_tst, columns=dataset.columns)

    return dataset_trn, dataset_tst
