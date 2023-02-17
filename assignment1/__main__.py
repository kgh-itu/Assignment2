from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor

from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
import numpy as np
import pickle

from assignment1.get_data import get_data, get_forecast_data
from assignment1.transformers import DropNA, SelectSpeedDirectionCols, DirectionToVector, NormalizeSpeed


def main():
    X, y = get_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.1)

    reg_models = {"Linear Regression": LinearRegression(),
                  "Gradient Boosting Regression": GradientBoostingRegressor(),
                  "K-nearest Neighbors Regression": KNeighborsRegressor()}

    pipeline_performance = {}

    for name, model in reg_models.items():
        pipeline = _create_pipeline(model)
        pipeline.fit(X_train, y_train)
        preds, true = pipeline.predict(X_test), y_test
        r2_test_score = r2_score(true, preds)
        pipeline_performance[name] = r2_test_score

    best_model = max(pipeline_performance, key=pipeline_performance.get)
    best_pipeline = _create_pipeline(reg_models[best_model])
    best_pipeline.fit(X_train, y_train)

    curr_best_model_r2 = pipeline_performance[best_model]
    print("Curr Best Model R2", curr_best_model_r2)

    try:
        stored_best_pipeline = pickle.load(open("best_pipeline.sav", 'rb'))
        stored_best_pipeline_preds = stored_best_pipeline.predict(X_test)
        stored_r2 = r2_score(y_test, stored_best_pipeline_preds)
        print("Stored Best Model R2", stored_r2)

        if curr_best_model_r2 > stored_r2:
            print(f"Saving NEW Best Pipeline {best_pipeline}")
            pickle.dump(best_pipeline, open('best_pipeline.sav', 'wb'))
        else:
            print("Stored model outperforms current best model")

    except FileNotFoundError:
        print(f"Saving INITIAL best pipeline {best_pipeline}")
        pickle.dump(best_pipeline, open('best_pipeline.sav', 'wb'))

    return best_pipeline


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
    pipeline_ = main()
