# !pip install mplfinance

# https://github.com/cderinbogaz/inpredo

import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")
from numpy import genfromtxt
import matplotlib.pyplot as plt
import mpl_finance
import numpy as np
import uuid
from datetime import datetime

# Daily data from Monday to Thursday for 43 years 1978 - 2020
# Input your csv file here with historical data
ad = genfromtxt("ML Project/stockpred/financial_data/samp_snp.csv", delimiter=','
                , dtype=str,skip_header=1)
pd = np.flipud(ad)

# Set the directories
buy_dir = "ML Project/stockpred/data/train/buy/"
sell_dir = "ML Project/stockpred/data/train/sell/"

# simple moving average for a mentioned period
def convolve_sma(array, period):
    return np.convolve(array, np.ones((period,)) / period, mode='valid')  # "valid" : We only get a result when the two signals overlap completely

def graphwerk(start, finish):
    date = []
    open = []
    high = []
    low = []
    close = []
    for x in range(finish - start):
        # Below filtering is valid for eurusd.csv file. Other financial data files have different orders so you need to find out
        # what means open, high and close in their respective order.
        date.append(float(datetime.strptime(df[start][0], '%d-%m-%y').timestamp()))
        open.append(float(df[start][1]))
        high.append(float(df[start][2]))
        low.append(float(df[start][3]))
        close.append(float(df[start][4]))
        start = start + 1

    fig = plt.figure(num=1, figsize=(3, 3), dpi=50, facecolor='w', edgecolor='k')
    dx = fig.add_subplot(111)
    mpl_finance.candlestick2_ochl(dx, open, close, high, low, width=1.5, colorup='g', colordown='r', alpha=0.5)
    plt.autoscale()
    plt.axis('off')

    if bb_signal[iter] == -1:
        plt.savefig(sell_dir + str(uuid.uuid4()) + '.jpg', bbox_inches='tight')
    else:
        plt.savefig(buy_dir + str(uuid.uuid4()) + '.jpg', bbox_inches='tight')

    open.clear()
    high.clear()
    low.clear()
    close.clear()
    plt.cla()
    plt.clf()

iter = 0

for x in range(len(pd) - 4):
    graphwerk(iter, iter + 12)
    iter = iter + 2

for i in range(len(bb_signal)):
    if bb_signal[i] != 0:
        iter = i
        graphwerk(iter-20, iter)
