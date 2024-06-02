import pandas as pd
import os
from shiny.types import ImgData
from shiny import ui
from pathlib import Path

from steps import prediction_service_loader, get_prediction
from .helper import load_yaml_file


def is_number(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


def prepare_single_prediction_data(
    sepal_length: str, sepal_width: str, petal_length: str, petal_width: str
) -> pd.DataFrame | str:
    data = pd.DataFrame(
        {
            "sepal length (cm)": [sepal_length],
            "sepal width (cm)": [sepal_width],
            "petal length (cm)": [petal_length],
            "petal width (cm)": [petal_width],
        }
    )

    is_valid = [
        is_number(n) for n in [sepal_length, sepal_width, petal_length, petal_width]
    ]

    if all(is_valid):
        return data

    return "One or some of the inputs are invalid. All inputs must be float."


def prepare_batch_prediction_data(uploaded_data: pd.DataFrame) -> pd.DataFrame | str:

    valid_column_names = [
        "sepal length (cm)",
        "sepal width (cm)",
        "petal length (cm)",
        "petal width (cm)",
    ]

    if valid_column_names != list(uploaded_data.columns):
        return (
            "Column names are invalid or not ordered correctly. "
            f"Column names must be and ordered as {valid_column_names}."
        )

    # Check for strings
    contains_strings = uploaded_data.map(lambda x: isinstance(x, str)).any().any()

    # Check for NaN values
    contains_nan = uploaded_data.isnull().any().any()

    if contains_strings | contains_nan:
        return (
            "Some values in the uploaded CSV contains strings and/or "
            "NaN values. Please check the file and re-upload."
        )

    return uploaded_data


def get_iris_dictionary() -> dict[int, str]:
    return dict(zip([0, 1, 2], ["setosa", "versicolor", "virginica"]))


def get_merged_prediction_data(prepared_data: pd.DataFrame) -> pd.DataFrame:
    config_path = os.path.join(
        os.path.realpath(os.path.join(os.path.dirname(__file__), "..")),
        "configs",
        "deployment_service_config.yaml",
    )
    config = load_yaml_file(path=config_path)
    service = prediction_service_loader(config=config["config"], running=True)
    predictions = get_prediction(service, prepared_data)

    prepared_data["predicted iris"] = predictions
    prepared_data["predicted iris"] = prepared_data["predicted iris"].map(
        get_iris_dictionary()
    )

    return prepared_data


def load_image(image_path: Path) -> ImgData:

    img: ImgData = {
        "src": image_path,
        "height": "200px",
        "width": "350px",
    }
    return img


def github_text() -> ui.Tag:
    return ui.markdown(
        """
        _The source code for this dashboard can be found in 
        this [link](https://github.com/Rasheed19/e2e-iris).
        This project makes use of the [ZenML](https://www.zenml.io/) machine learning 
        oprations (MLOps) structure to develop both the model steps
        and pipelines. The dashboard is built using the [shiny](https://shiny.posit.co/py/) python
        framework._
        """
    )


def about_prediction_service() -> ui.Tag:
    return ui.markdown(
        """This prediction service is obtained from 
            training the gradient boost model on the Iris
            dataset downloaded from the scikit-learn library.
            The data contains the lengths and widths of the 
            sepal and petal of three irises namely setosa, versicolor,
            and virginica. More information about this dataset
            can be found [here](https://en.wikipedia.org/wiki/Iris_flower_data_set).
            The model is served using the MLflow deployment service via 
            the ZenML model deployer stack.
            """
    )
