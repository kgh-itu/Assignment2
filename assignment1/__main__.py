from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from assignment1.get_data import get_data
from assignment1.transformers import DropNA, SelectRelevantColumns, DirectionToVector, NormalizeSpeed


def main():
    X_train, y_train = get_data()
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, shuffle=False, test_size=0.1)

    pipeline = Pipeline(

        [("column_filter", SelectRelevantColumns()),
         ("drop_na", DropNA()),
         ("direction_to_degrees", DirectionToVector()),
         ("normalize_speed", NormalizeSpeed()),
         ("lin_reg", LinearRegression())
         ]

    )

    pipeline.fit(X_train, y_train)
    print(pipeline.score(X_test, y_test))


if __name__ == '__main__':
    main()
