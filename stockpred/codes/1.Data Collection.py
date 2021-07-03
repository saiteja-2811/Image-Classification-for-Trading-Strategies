# Import Libraries
import pandas as pd
import matplotlib.pyplot as plt
import mpl_finance
import fix_yahoo_finance as yf

# Pull data from yahoo with a timeline from 1983 1st Jan to - 2021 18th June
ticker = '^GSPC'
df = yf.download(ticker, start='1983-01-01', end='2021-06-18')
df.reset_index(inplace=True)
root_dir = "C:/Users/saite/PycharmProjects/py38"

df.to_csv(root_dir + "/ML Project/stockpred/data/stock_data.csv",index=False)

# Import the scraped data
df = pd.read_csv(root_dir + "/ML Project/stockpred/data/stock_data.csv")

df = df[['Date','Open','High','Low','Close']]
df = df.rename({'Date':'date','High':'high','Low':'low','Open':'open','Close':'close'},axis=1)
df = df.set_index('date')
df.index = pd.to_datetime(df.index)

# Bollinger Bands calculation

# Moving Averages Calculation
def sma(data, window):
    sma = data.rolling(window = window).mean()
    return sma
df['sma_20'] = sma(df['close'], 20)

# Bands Calculation
def bb(data, sma, window):
    std = data.rolling(window = window).std()
    upper_bb = sma + std * 2
    lower_bb = sma - std * 2
    return upper_bb, lower_bb
df['upper_bb'], df['lower_bb'] = bb(df['close'], df['sma_20'], 20)

# Function for strategy
def implement_bb_strategy(data, lower_bb, upper_bb):
    bb_signal = []
    for i in range(len(data)):
        # Two consecutive rows comparison for trend detection
        # Scenario 1 Both Below Lower Band - Buy
        if data[i - 1] < lower_bb[i - 1] and data[i] < lower_bb[i]:
            signal = 1
            bb_signal.append(signal)
        # Scenario 2 Middle to Upper - Sell
        elif data[i - 1] < upper_bb[i - 1] and data[i] > upper_bb[i]:
            signal = -1
            bb_signal.append(signal)
        # Scenario 3 Both Upper - Sell
        elif data[i - 1] > upper_bb[i - 1] and data[i] > upper_bb[i]:
            signal = -1
            bb_signal.append(signal)
        # Scenario 4 Middle to Lower - Buy
        elif data[i - 1] > lower_bb[i - 1] and data[i] < lower_bb[i]:
            signal = 1
            bb_signal.append(signal)
        else:
            bb_signal.append(0)

    return bb_signal
bb_signal = implement_bb_strategy(df['close'], df['lower_bb'], df['upper_bb'])
bb_signal = pd.DataFrame(bb_signal).rename(columns = {0:'bb_signal'}).set_index(df.index)
df = df.join(bb_signal, how = 'inner')
df.reset_index(inplace=True)
df.bb_signal.value_counts(dropna=False)
# Creating Images
methods = ['BPS','BHPS','BPHS']
str_dir_1 = str()
str_dir_2 = str()

def plotgraph(start, finish,method,str_dir_1,str_dir_2):
    open = []
    high = []
    low = []
    close = []
    for x in range(finish - start):
        open.append(float(df.iloc[start][1]))
        high.append(float(df.iloc[start][2]))
        low.append(float(df.iloc[start][3]))
        close.append(float(df.iloc[start][4]))
        start = start + 1

    fig = plt.figure(num=1, figsize=(3, 3), dpi=50, facecolor='w', edgecolor='k')
    dx = fig.add_subplot(111)
    mpl_finance.candlestick2_ochl(dx, open, close, high, low, width=1.5, colorup='g', colordown='r', alpha=0.5)
    plt.autoscale()
    plt.axis('off')

    # Save to separate directories
    if method == "BPS":
    # Buy & Sell
        if df.bb_signal[finish] == -1:
            plt.savefig(str_dir_1 + str(finish) + '.jpg', bbox_inches='tight') #buy
        else:
            plt.savefig(str_dir_2 + str(finish) + '.jpg', bbox_inches='tight') #sell
    # Buy , [Hold + Sell]
    elif method == "BHPS":
        if df.bb_signal[finish] == 1:
            plt.savefig(str_dir_1 + str(finish) + '.jpg', bbox_inches='tight') #buy
        else:
            plt.savefig(str_dir_2 + str(finish) + '.jpg', bbox_inches='tight') #sell + hold

    #[Buy + Hold], Sell
    else:
        if df.bb_signal[finish] == -1:
            plt.savefig(str_dir_2 + str(finish) + '.jpg', bbox_inches='tight') #sell
        else:
            plt.savefig(str_dir_1 + str(finish) + '.jpg', bbox_inches='tight') #buy + hold
        ""
    open.clear()
    high.clear()
    low.clear()
    close.clear()
    plt.cla()
    plt.clf()

# Set the directories
for i in methods:
    # Buy & Sell
    if i == "BPS":
        str_dir_1 = root_dir + "/ML Project/stockpred/str1/train/buy/"
        str_dir_2 = root_dir + "/ML Project/stockpred/str1/train/sell/"
        for j in range(21,len(df.bb_signal)):
            iter = 0
            if df.bb_signal[j] != 0:
                plotgraph(j - 20, j,i, str_dir_1, str_dir_2)
    # Buy and [Hold + Sell]
    elif i == "BHPS":
        str_dir_1 = root_dir + "/ML Project/stockpred/str2/train/buy/"
        str_dir_2 = root_dir + "/ML Project/stockpred/str2/train/hold_sell/"
        for j in range(21, len(df.bb_signal)):
            plotgraph(j - 20, j, i, str_dir_1, str_dir_2)
    # [Buy + Hold], Sell
    else:
        str_dir_1 = root_dir + "/ML Project/stockpred/str3/train/buy_hold/"
        str_dir_2 = root_dir + "/ML Project/stockpred/str3/train/sell/"
        for j in range(21, len(df.bb_signal)):
            plotgraph(j - 20, j, i, str_dir_1, str_dir_2)








