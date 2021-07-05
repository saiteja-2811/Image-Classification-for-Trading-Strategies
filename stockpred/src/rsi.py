import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.dates import num2date, date2num
warnings.simplefilter("ignore")

import yfinance as yf
ticker='^GSPC'
tickerData = yf.Ticker(ticker)
df = tickerData.history(period='1d', start='1983-01-01', end='2021-06-18')

df.head()


def rsi(df, window=14):
    close = df['close']

    # Price increase or decrease over previous day
    dif = close.diff()
    dif = dif[1:]

    # pos_m identifies stock price going up
    # neg_m udentifies stock price going down
    pos_m, neg_m = dif.copy(), dif.copy()
    pos_m[pos_m < 0] = 0
    neg_m[neg_m > 0] = 0

    # Positive Rolling Mean Exponential
    prme = pos_m.ewm(span=window).mean()
    # Negative Rolling Mean Exponential
    nrme = neg_m.abs().ewm(span=window).mean()

    # Ratio of magnitude of up move to down move
    RSE = prme / nrme
    RSIE = 100.0 - (100.0 / (1.0 + RSE))
    df['rsie'] = RSIE

    # Positive Rolling Mean Simple
    prms = pos_m.rolling(window).mean()
    # Negative Rolling Mean Simple
    nrms = neg_m.abs().rolling(window).mean()

    RSS = prms / nrms
    RSIS = 100.0 - (100.0 / (1.0 + RSS))

    df['rsis'] = RSIS
    return df

df = rsi(df)