from dataclasses import dataclass
import pandas as pd
from sklearn.base import BaseEstimator


@dataclass
class SavedModel:
    model: BaseEstimator
    training_data: pd.DataFrame
    predictor_columns: list[str]
    response_column: str
    codes: pd.DataFrame
