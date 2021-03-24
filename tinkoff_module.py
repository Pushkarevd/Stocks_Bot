import pandas as pd
import numpy as np
import tinvest as ti
from datetime import datetime, timedelta
import plotly.graph_objects as go
from os import walk
from config import token


client = ti.SyncClient(token)


# Function for creating DataFrame of portfolio
def get_portfolio() -> pd.DataFrame:
    data_raw = client.get_portfolio().payload
    data_clear = {p.name: {"currency": p.average_position_price.currency.value,
                           "price": float(p.average_position_price.value),
                           "amount": float(p.balance),
                           "figi": p.figi} for p in data_raw.positions}
    dataframe = pd.DataFrame.from_dict(data_clear, orient='index')

    return dataframe


# Stocks history
def get_figi_data(figi: str, days=1, time_now=datetime.now()) -> pd.DataFrame:
    interval = ti.CandleResolution.hour if days == 1 else ti.CandleResolution.day
    data = client.get_market_candles(
        figi=figi,
        from_=time_now - timedelta(days=days),
        to=time_now,
        interval=interval,
    ).payload # Api doesn't count playdays
    dataframe = pd.DataFrame([{"close" : float(step.c),
                               "high": float(step.h),
                               "low": float(step.l),
                               "open": float(step.o),
                               "time": step.time} for step in data.candles])
    return dataframe


# Create new data of stocks with specific figi
def create_data(figi: str, days = 50):
    data = [get_figi_data(figi, 1, datetime.now() - timedelta(days=i)) for i in range(days)]
    pd.concat(data, ignore_index=True).to_csv("./stocks_data/{}.csv".format(figi))


# Function that find already existing data
def find_data(figi: str):
    _, _, filenames = next(walk("./stocks_data"))
    if figi + ".csv" in filenames:
        return pd.read_csv("./stocks_data/{}.csv".format(figi))
    else:
        create_data(figi)
        return pd.read_csv("./stocks_data/{}.csv".format(figi))


if __name__ == "__main__":
    print(get_portfolio())
    print(find_data("BBG000BNGBW9"))
    print(find_data("BBG0013HGFT4"))
    data = find_data("BBG000BNGBW9").iloc[:500]
    fig = go.Figure(data=[go.Candlestick(x=data['time'],
                    open=data['open'],
                    high=data['high'],
                    low=data['low'],
                    close=data['close'])])
    fig.show()
