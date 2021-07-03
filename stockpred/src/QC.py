import pandas as pd
import matplotlib.pyplot as plt
import requests
import math
from termcolor import colored as cl
import numpy as np

plt.style.use('fivethirtyeight')
plt.rcParams['figure.figsize'] = (20, 10)


def get_historic_data(symbol):
    ticker = symbol
    iex_api_key = 'Tsk_30a2677082d54c7b8697675d84baf94b'
    api_url = f'https://sandbox.iexapis.com/stable/stock/{ticker}/chart/max?token={iex_api_key}'
    df = requests.get(api_url).json()

    date = []
    open = []
    high = []
    low = []
    close = []

    for i in range(len(df)):
        date.append(df[i]['date'])
        open.append(df[i]['open'])
        high.append(df[i]['high'])
        low.append(df[i]['low'])
        close.append(df[i]['close'])

    date_df = pd.DataFrame(date).rename(columns={0: 'date'})
    open_df = pd.DataFrame(open).rename(columns={0: 'open'})
    high_df = pd.DataFrame(high).rename(columns={0: 'high'})
    low_df = pd.DataFrame(low).rename(columns={0: 'low'})
    close_df = pd.DataFrame(close).rename(columns={0: 'close'})
    frames = [date_df, open_df, high_df, low_df, close_df]
    df = pd.concat(frames, axis=1, join='inner')
    return df


tsla = get_historic_data('TSLA')
tsla = tsla.set_index('date')
tsla = tsla[tsla.index >= '2020-01-01']
tsla.to_csv('tsla.csv')

tsla = pd.read_csv('tsla.csv').set_index('date')
tsla.index = pd.to_datetime(tsla.index)
tsla.tail()