import pandas as pd


def get_data():
    X = pd.read_csv("X_train.csv").drop(columns=["Unnamed: 0"])
    Y = pd.read_csv("Y_train.csv").drop(columns=["Unnamed: 0"])["Total"]

    return X, Y


def get_df(results):
    values = results.raw['series'][0]['values']
    columns = results.raw['series'][0]['columns']
    df = pd.DataFrame(values, columns=columns).set_index("time")
    df.index = pd.to_datetime(df.index)

    return df
