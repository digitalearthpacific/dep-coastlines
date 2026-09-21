from dataclasses import dataclass
from joblib import dump

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator

from dep_coastlines.config import CLOUD_MODEL_FILE
from dep_coastlines.cloud_model.prep_training_data import TRAINING_DATA_FILE


@dataclass
class SavedModel:
    model: BaseEstimator
    training_data: pd.DataFrame
    predictor_columns: list[str]
    response_column: str
    codes: pd.DataFrame


def train(training_data):
    codes = pd.DataFrame.from_records(
        [
            ("land", 1, 1, 0, 1, "#064a00"),
            ("clean_water", 9, 2, 0, 2, "#ff0000"),
            ("water", 2, 3, 0, 2, "#ff0000"),
            ("deep_ocean", 3, 3, 0, 2, "#ff0000"),
            ("noisy_water", 8, 7, 0, 2, "#00ffff"),
            ("surf", 4, 4, 0, 4, "#ddfeff"),
            ("coast", 5, 5, 0, 5, "#ffe48b"),
            ("cloud", 6, 6, 1, 6, "#a9a9a9"),
            ("noise", 7, 6, 1, 6, "#e16be1"),
            ("bare_terrain", 10, 8, 0, 8, "#0000ff"),
            ("built", 11, 9, 0, 9, "#ff0000"),
        ],
        columns=["code", "intcode", "simplintcode", "cloud", "unified_water", "color"],
    )

    removes = ["shift"]
    training_data = pd.read_csv(training_data).query("code not in @removes")

    training_data = training_data.join(codes.set_index("code"), on="code")
    training_columns = [
        "blue",
        "blue_mad",
        "count",
        "green",
        "green_mad",
        "nir08",
        "nir08_mad",
        "red",
        "red_mad",
        "swir16",
        "swir16_mad",
        "swir22",
        "swir22_mad",
        "twndwi",
        "twndwi_mad",
        "twndwi_stdev",
    ]
    training_data = training_data.dropna(subset=training_columns)
    X = training_data.loc[:, training_columns].to_numpy()
    response_column = "cloud"
    y = training_data[response_column].to_numpy()

    m = RandomForestClassifier()
    full_model = m.fit(X, y)
    return SavedModel(
        model=full_model,
        training_data=training_data,
        predictor_columns=training_columns,
        response_column=response_column,
        codes=codes,
    )


if __name__ == "__main__":
    output = train(TRAINING_DATA_FILE)
    dump(output, CLOUD_MODEL_FILE)
