from sklearn.datasets import load_iris
import pandas as pd
from typing_extensions import Annotated
from zenml import step
from zenml.logger import get_logger

logger = get_logger(__name__)


@step(enable_cache=False)
def data_loader() -> (
    tuple[Annotated[pd.DataFrame, "dataset"], Annotated[str, "target"]]
):

    target = "target"

    raw_data = load_iris(as_frame=True)
    dataset = raw_data.data
    dataset[target] = raw_data.target
    logger.info(f"Dataset with {len(dataset)} records loaded!")

    return dataset, target
