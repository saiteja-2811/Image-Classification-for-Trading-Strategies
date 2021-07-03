# Import Libraries
import pandas as pd
import matplotlib.pyplot as plt
from pandas_datareader import data as web
import math
from termcolor import colored as cl
import numpy as np

# Set the plot style
plt.style.use('fivethirtyeight')
plt.rcParams['figure.figsize'] = (20, 10)

# Pull data from yahoo with a timeline from 2010 1st Jan to - 2021 18th June
ticker = '^GSPC'
df = web.DataReader(ticker, data_source='yahoo', start='1978-01-01', end='2021-06-18')
df.reset_index(inplace=True)
df = df[['Date','Open','High','Low','Close']]
df = df.rename({'Date':'date','High':'high','Low':'low','Open':'open','Close':'close'},axis=1)
df = df.set_index('date')
df.index = pd.to_datetime(df.index)
df.tail()

# Bollinger Bands calculation

# Moving Averages Calculation
def sma(data, window):
    sma = data.rolling(window = window).mean()
    return sma

df['sma_20'] = sma(df['close'], 20)
df.tail()

# Bands Calculation
def bb(data, sma, window):
    std = data.rolling(window = window).std()
    upper_bb = sma + std * 2
    lower_bb = sma - std * 2
    return upper_bb, lower_bb

df['upper_bb'], df['lower_bb'] = bb(df['close'], df['sma_20'], 20)
df.tail()

df.to_csv('df_v3.csv',index=False)


# plot BB
# df['close'].plot(label = 'CLOSE PRICES', color = 'skyblue')
# df['upper_bb'].plot(label = 'UPPER BB 20', linestyle = '--', linewidth = 1, color = 'black')
# df['sma_20'].plot(label = 'MIDDLE BB 20', linestyle = '--', linewidth = 1.2, color = 'grey')
# df['lower_bb'].plot(label = 'LOWER BB 20', linestyle = '--', linewidth = 1, color = 'black')
# plt.legend(loc = 'upper left')
# plt.title('df BOLLINGER BANDS')
# plt.show()

# Function for strategy
def implement_bb_strategy(data, lower_bb, upper_bb):
    buy_price = []
    sell_price = []
    bb_signal = []
    signal = 0

    for i in range(len(data)):
        # Two consecutive rows comparison for trend detection
        # Scenario 1 Both Below Lower Band - Buy
        if data[i - 1] < lower_bb[i - 1] and data[i] < lower_bb[i]:
            if signal != 1:
                buy_price.append(data[i])
                sell_price.append(np.nan)
                signal = 1
                bb_signal.append(signal)
            else:
                buy_price.append(np.nan)
                sell_price.append(np.nan)
                bb_signal.append(0)
        # Scenario 2 Middle to Upper - Sell
        elif data[i - 1] < upper_bb[i - 1] and data[i] > upper_bb[i]:
            if signal != -1:
                buy_price.append(np.nan)
                sell_price.append(data[i])
                signal = -1
                bb_signal.append(signal)
            else:
                buy_price.append(np.nan)
                sell_price.append(np.nan)
                bb_signal.append(0)
        # Scenario 3 Both Upper - Sell
        elif data[i - 1] > upper_bb[i - 1] and data[i] > upper_bb[i]:
            if signal != -1:
                buy_price.append(np.nan)
                sell_price.append(data[i])
                signal = -1
                bb_signal.append(signal)
            else:
                buy_price.append(np.nan)
                sell_price.append(np.nan)
                bb_signal.append(0)
        # Scenario 4 Middle to Lower - Buy
        elif data[i - 1] > lower_bb[i - 1] and data[i] < lower_bb[i]:
            if signal != 1:
                buy_price.append(data[i])
                sell_price.append(np.nan)
                signal = 1
                bb_signal.append(signal)
            else:
                buy_price.append(np.nan)
                sell_price.append(np.nan)
                bb_signal.append(0)

        else:
            buy_price.append(np.nan)
            sell_price.append(np.nan)
            bb_signal.append(0)

    return buy_price, sell_price, bb_signal


buy_price, sell_price, bb_signal = implement_bb_strategy(df['close'], df['lower_bb'], df['upper_bb'])