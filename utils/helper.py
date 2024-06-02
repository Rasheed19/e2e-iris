from typing import Any
import yaml
import pickle
import os
import pandas as pd


def load_yaml_file(path: str) -> dict[Any, Any]:
    with open(path, "r") as file:
        data = yaml.safe_load(file)

    return data


def dump_yaml_file(data: dict[Any, Any], relative_path: str) -> None:
    parent = os.path.realpath(os.path.join(os.path.dirname(__file__), ".."))
    with open(f"{parent}/{relative_path}", "w") as file:
        yaml.dump(data, file, default_flow_style=False)

    return None


def dump_data(data: Any, relative_path: str) -> None:
    parent = os.path.realpath(os.path.join(os.path.dirname(__file__), ".."))
    with open(f"{parent}/{relative_path}", "wb") as fp:
        pickle.dump(data, fp)

    return None


class DataFrameCaster:
    """Support class to cast type back to pd.DataFrame in sklearn Pipeline."""

    def __init__(self, columns):
        self.columns = columns

    def fit(self, *args, **kwargs):
        return self

    def transform(self, X):
        return pd.DataFrame(X, columns=self.columns)
