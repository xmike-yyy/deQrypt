import requests
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split


def get_data(ticker):
    df = yf.Tickers(ticker).tickers[ticker].history(period="5y").iloc[:,:4][:-1]
    return df


def feature_today_trend(opening_price, closing_price):
    p_o = opening_price
    p_c = closing_price

    return 1 if p_c - p_o >= 0 else 0


# Desc: Calculates the avg of a selected range of prices, usually closing prices
# by the # of periods in that range
# Formula: (P_1 + P_2  + P_3  + ... + P_n)/n
def feature_sma(prices, n = 14) -> float:
    sum = 0.0
    for price in prices:
        sum+=price
    average = sum/n
    return average

# RSI (Relative Strength Index) - Michael Tesfaye 🐐
# Desc: measure of momentum of a stock to identify whether it is overbought or oversold
# Ranges between 0-100, inclusive
# Formula: 100 - [100/(1 + rs))]
def average(prices) -> float:
    sum = 0.0
    for price in prices:
        sum += price
    average = sum/len(prices)
    return average


def feature_rsi(closing_prices, period = 14) -> float:
    # deltas = stock_data["Close"].diff()
    deltas = []
    for i, price in enumerate(closing_prices):
        if i == 0:
            deltas += [0]
        else:
            deltas += [price - closing_prices[i - 1]]

    gains = [delta if delta > 0 else 0 for delta in deltas]
    losses = [-delta if delta < 0 else 0 for delta in deltas]

    avg_gain = np.mean(gains[-period:])
    avg_loss = np.mean(losses[-period:])

    rs = avg_gain/avg_loss
    rsi = 100 - (100/(1 + rs))
    return rsi


# Computes (in a similar manner to RSI) an indication of momentum of a stock for
# seeing whether it is being overbought or oversold, but uses "support" and "resistance" levels. Default is n = 14 trading days
def feature_oscillator(closing_prices, low_prices, high_prices, n = 14) -> float:
    p_c = closing_prices[-1]
    lowest_low = min(low_prices[-n:])
    highest_high = max(high_prices[-n:])
    k = 100 * (p_c - lowest_low) / (highest_high - lowest_low)
    return k



def generate_features(ticker):
    stock_data, test = get_data(ticker)
    stock_data = stock_data.sort_values('Date', ascending=False)

    closing_prices = stock_data['Close'].tolist()
    opening_prices = stock_data['Open'].tolist()
    low_prices = stock_data['Low'].tolist()
    high_prices = stock_data['High'].tolist()

    required_days = 14  # Maximum look-back period required by any feature

    if len(stock_data) < required_days:
        raise ValueError(f"Insufficient data. At least {required_days} days of data are required.")
    nrsi = [feature_rsi(closing_prices[i:i+15]) for i in range(len(closing_prices) - 14)]
    nsma = [feature_sma(closing_prices[i:i+15]) for i in range(len(closing_prices) - 14)]
    nocsi = [feature_oscillator(closing_prices[i:i+15], low_prices[i:i+15], high_prices[i:i+15]) for i in range(len(closing_prices) - 14)]
    trends = [feature_today_trend(opening_prices[i], closing_prices[i]) for i in range(len(opening_prices))][-len(nsma):]


    features = pd.DataFrame({
        "tdtrend": trends[:-1],
        "rsi": nrsi[:-1],
        "sma": nsma[:-1],
        "oscillator": nocsi[:-1],
    })
    test = pd.DataFrame({
        "rsi": [nrsi[-1]],
        "sma": [nsma[-1]],
        "oscillator": [nocsi[-1]],
    })
    return features, test




def scaleData(features):
    columns_to_scale = ['rsi', 'sma', 'oscillator']

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(features[columns_to_scale])
    scaled_df = pd.DataFrame(scaled_data, columns=['rsi', 'sma', 'oscillator'])
    df_scaled = features.assign(rsi=scaled_df['rsi'], sma=scaled_df['sma'], oscillator = scaled_df['oscillator'])
    return df_scaled

def train_and_predict(data,tp):

    data = scaleData(data)
    X = data.drop('tdtrend', axis=1)
    Y = data['tdtrend']
    X_train, X_test, Y_train,  Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)



    from tensorflow.keras.layers import BatchNormalization

    model = Sequential()
    model.add(Dense(512, input_shape=(X_train.shape[1],), activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(256, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(128, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(32, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

    early_stopping = EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)

    history = model.fit(X_train, Y_train, validation_split=0.2, epochs=100, batch_size=32, callbacks=[early_stopping])

    test_loss, test_accuracy = model.evaluate(X_test, Y_test)
    
    res = model.predict(tp)

    return res
