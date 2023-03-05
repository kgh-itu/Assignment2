from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score)

from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import TimeSeriesSplit
import numpy as np
import json
import mlflow
import shutil
from datetime import datetime

from get_data import get_data, get_forecast_data
from transformers import DropNA, SelectSpeedDirectionCols, DirectionToVector, NormalizeSpeed
import mlflow


def main():
    X, y = get_data()

    reg_models = {"Linear Regression": LinearRegression(),
                  "Gradient Boosting Regression": GradientBoostingRegressor(),
                  "K-nearest Neighbors Regression": KNeighborsRegressor()}

    for reg_name, reg_model in reg_models.items():

        metrics = {
            "MAE": (mean_absolute_error, []),
            "MSE": (mean_squared_error, []),
            "R2-Score": (r2_score, [])}

        with mlflow.start_run(run_name=f"{reg_name}"):

            pipeline = _create_pipeline(reg_model)

            for train, test in TimeSeriesSplit(4).split(X, y):
                pipeline.fit(X.iloc[train], y.iloc[train])
                preds, true = pipeline.predict(X.iloc[test]), y.iloc[test]

                for metric_name, metric_tuple in metrics.items():
                    metric_func, score_list = metric_tuple
                    score_list.append(metric_func(true, preds))

            pipeline.fit(X, y)
            mlflow.sklearn.log_model(pipeline, artifact_path=f"{reg_name}_model")

            for metric_name, val in metrics.items():
                _, scores = val
                mlflow.log_metric(f"Mean {metric_name}", float(np.mean(scores)))

            final_score = np.mean(metrics["MSE"][1])

            try:
                best_model = _read_best_model_from_disk()
                if final_score < best_model["score"]:
                    shutil.rmtree('best_model')  # deleting current best_model directory
                    print(f"Saving new best model {reg_model} to disk")
                    _save_model_to_disk(pipeline)
                    best_model = _create_new_model_dict(reg_name, final_score)
                    _save_model_metadata_to_disk(best_model)

                else:
                    print(f"Model {reg_model} not better. Not saving")

            except FileNotFoundError:
                print(f"No model saved to disk yet. Saving {reg_model}")
                _save_model_to_disk(pipeline)
                best_model = _create_new_model_dict(reg_name, final_score)
                _save_model_metadata_to_disk(best_model)


def _create_new_model_dict(model_name, final_score):
    model_dict = {
        "timestamp": datetime.now().timestamp(),
        "date": datetime.today().strftime("%m/%d/%Y"),
        "model_name": model_name,
        "score": final_score,
    }

    return model_dict


def _save_model_to_disk(pipeline):
    mlflow.sklearn.save_model(
        pipeline,
        path="best_model")


def _read_best_model_from_disk():
    with open("best_model.json", "r") as f:
        best_model = json.loads(f.read())
        return best_model


def _save_model_metadata_to_disk(best_model):
    with open("best_model.json", "w") as f:
        json.dump(best_model, f)


def _create_pipeline(reg_model):
    pipeline = Pipeline(

        [("drop_na", DropNA()),
         ("column_filter", SelectSpeedDirectionCols()),
         ("normalize_speed", NormalizeSpeed()),
         ("direction_to_vector", DirectionToVector()),
         ("poly_features", PolynomialFeatures(degree=2)),
         ("reg_model", reg_model)
         ]

    )

    return pipeline


if __name__ == '__main__':
    main()
