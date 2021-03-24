from silence_tensorflow import silence_tensorflow
silence_tensorflow()
import os, sys
import math
import joblib
import pandas as pd
import numpy as np
from tensorflow import keras
from models import LSTM_Model
import pandas_datareader as web
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler

# TODO fix checking exists of stock

# Create model for specific stock
def create_model(stock: str):
    # Consts
    days_before = 60
    start = '2010-01-01'

    # Get data from start to today
    try:
        df = web.DataReader(stock, data_source='yahoo', start=start, end=datetime.now())
    except:
        return None

    # Preprocessing data
    data = df.filter(['Close'])
    dataset = data.values

    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(dataset)

    # Percentage of training data
    training_data_len = math.ceil(len(dataset) * 0.8)

    # Create train & test partitions of data(number of days before prediction)
    train_data = scaled_data[0:training_data_len, :]

    x_train = np.array([train_data[i-days_before:i, 0] for i in range(days_before, len(train_data))])
    y_train = np.array([train_data[i, 0] for i in range(days_before, len(train_data))])

    test_data = scaled_data[training_data_len-days_before: , :]
    x_test = np.array([test_data[i-days_before:i, 0] for i in range(days_before, len(test_data))])
    y_test = dataset[training_data_len:, :]

    # Reshape data for models
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    # Init model
    model = LSTM_Model(x_train.shape[1])
    model.fit(x_train, y_train)

    # Test model
    predictions = scaler.inverse_transform(model.predict(x_test))

    # Save models
    os.mkdir('./models/{}'.format(stock))
    model.save(stock)
    joblib.dump(scaler, "./models/{}/scaler.save".format(stock))

    model._rmse = np.sqrt(np.mean(predictions - y_test)**2)
# Find existing model
def find_model(stock: str) -> str:
    filename = "lstm_model.h5".format(stock)

    # Check if models for this stock exists
    if not(os.path.isdir("./models/{}".format(stock))):
        flag = create_model(stock)

    return "./models/{}/{}".format(stock, filename)


# Function that predict stock price for next day
def predict_stock_price(stock: str) -> float:
    model_path = find_model(stock)
    # Load existing models
    model = keras.models.load_model(model_path)
    scaler = joblib.load("./models/{}/scaler.save".format(stock))

    # Get data for previos 60 days

    stock_quote = web.DataReader(stock, data_source='yahoo', start=datetime.now() - timedelta(days=60), end=datetime.now()).filter(["Close"])

    # Preprocess data for model
    last_60_days = scaler.transform(stock_quote.values)
    X = np.array([last_60_days])
    last_60_days = np.reshape(X, (X.shape[0], X.shape[1], 1))

    # Get prediction
    pred_price= scaler.inverse_transform(model.predict(X))[0][0]

    return "Tomorrow the {} will cost: {:.4f}".format(stock, pred_price)


def main():
    print(predict_stock_price(input("Type stock symbols: ")))

if __name__ == "__main__":
    main()
