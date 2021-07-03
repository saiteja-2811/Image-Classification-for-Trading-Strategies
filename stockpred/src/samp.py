from pandas_datareader import data as web
import plotly.graph_objects as go

stock = '^GSPC'
df = web.DataReader(stock, data_source='yahoo', start='06-01-2021',end='06-18-2021')

trace1 = {
    'x': df.index,
    'open': df.Open,
    'close': df.Close,
    'high': df.High,
    'low': df.Low,
    'type': 'candlestick',
    'name': '^GSPC',
    'showlegend': True
}

# Set the chart layout
data1 = [trace1]
# Config graph layout
layout = go.Layout({
    'title': {
        'text': 'S&P 500 Moving Averages',
        'font': {
            'size': 15
        }
    }
})

# Calculate and define moving average of 30 periods
avg_30 = df.Close.rolling(window=30, min_periods=1).mean()

# Calculate and define moving average of 50 periods
avg_50 = df.Close.rolling(window=50, min_periods=1).mean()

trace2 = {
    'x': df.index,
    'y': avg_30,
    'type': 'scatter',
    'mode': 'lines',
    'line': {
        'width': 1,
        'color': 'blue'
            },
    'name': 'Moving Average of 30 periods'
}

trace3 = {
    'x': df.index,
    'y': avg_50,
    'type': 'scatter',
    'mode': 'lines',
    'line': {
        'width': 1,
        'color': 'red'
    },
    'name': 'Moving Average of 50 periods'
}
data2 = [trace2]
data3 = [trace3]

fig1 = go.Figure(data=data1, layout=layout)
fig1.show()
fig2 = go.Figure(data=data2, layout=layout)
fig2.show()
fig3 = go.Figure(data=data3, layout=layout)
fig3.show()

# 2The Deep Neural Net uses 32x32x32 structure, while the Convolutional Neural Net
# (CNN) uses three layers of 32 3x3 filters with ReLU activations and Max Pooling of
# 2x2 in between the layers. The last layer incorporates Sigmoid activation. The CCN
# model is compiled with Adam optimizer, binary-cross entropy loss function and run
# with a batch size of 16 samples for 50 iterations