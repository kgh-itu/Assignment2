import pandas as pd



def get_data():
    # client = InfluxDBClient(host='influxus.itu.dk', port='8086', username='lsda', password='icanonlyread')
    # client.switch_database('orkney')
    # generation = client.query(
    #     "SELECT * FROM Generation where time > now()-90d"
    # )
    # wind = client.query(
    #     "SELECT * FROM MetForecasts where time > now()-90d and time <= now() and Lead_hours='1'"
    # )
    #
    # gen_df = get_df(generation)
    # wind_df = get_df(wind)
    #
    # merged = pd.merge_asof(gen_df, wind_df, direction="nearest", on="time")
    #
    # Y = merged["Total"]
    # X = merged.drop(columns=["Total"])
    #
    # X.to_csv("X_train.csv")
    # Y.to_csv("Y_train.csv")

    X = pd.read_csv("X_train.csv").drop(columns=["Unnamed: 0"])
    Y = pd.read_csv("Y_train.csv").drop(columns=["Unnamed: 0"])["Total"]


    return X, Y


# def get_forecast_data():
#     client = InfluxDBClient(host='influxus.itu.dk', port='8086', username='lsda', password='icanonlyread')
#     client.switch_database('orkney')
#     forecasts = client.query("SELECT * FROM MetForecasts where time > now()")
#     for_df = get_df(forecasts)
#     newest_source_time = for_df["Source_time"].max()
#     newest_forecasts = for_df.loc[for_df["Source_time"] == newest_source_time].copy()
#
#     return newest_forecasts


def get_df(results):
    values = results.raw['series'][0]['values']
    columns = results.raw['series'][0]['columns']
    df = pd.DataFrame(values, columns=columns).set_index("time")
    df.index = pd.to_datetime(df.index)

    return df