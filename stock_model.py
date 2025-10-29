import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping

MODEL_PATH = "model.h5"
SCALER_PATH = "scaler.npy"

def build_model(input_shape):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train_model(symbol, epochs=10):
    # Fetch 1 year of historical data
    df = yf.download(symbol, period="1y")
    if df is None or df.empty:
        raise ValueError("Couldn't download data for symbol: " + symbol)
    close = df['Close'].values.reshape(-1,1)

    scaler = MinMaxScaler(feature_range=(0,1))
    scaled = scaler.fit_transform(close)

    X, y = [], []
    seq_len = 60
    for i in range(seq_len, len(scaled)):
        X.append(scaled[i-seq_len:i, 0])
        y.append(scaled[i, 0])
    X, y = np.array(X), np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    model = build_model((X.shape[1], 1))
    early = EarlyStopping(monitor='loss', patience=3, restore_best_weights=True)
    model.fit(X, y, epochs=epochs, batch_size=32, callbacks=[early], verbose=1)

    # save model and scaler
    model.save(MODEL_PATH)
    # save scaler params (min_ and scale_) using numpy
    np.save(SCALER_PATH, np.array([scaler.min_.tolist(), scaler.scale_.tolist()], dtype=object), allow_pickle=True)
    return model, scaler

def load_scaler():
    if not os.path.exists(SCALER_PATH):
        return None
    arr = np.load(SCALER_PATH, allow_pickle=True)
    mins, scales = arr[0], arr[1]
    scaler = MinMaxScaler(feature_range=(0,1))
    # hack: set attributes directly
    scaler.min_ = np.array(mins, dtype=float)
    scaler.scale_ = np.array(scales, dtype=float)
    scaler.data_min_ = np.zeros_like(scaler.min_)
    scaler.data_max_ = scaler.data_min_ + 1.0
    scaler.data_range_ = scaler.data_max_ - scaler.data_min_
    return scaler

def predict_future_prices(symbol, days_ahead=1):
    # Validate days
    if days_ahead < 1 or days_ahead > 7:
        raise ValueError("days_ahead must be between 1 and 7")

    # Download recent data
    df = yf.download(symbol, period="1y")
    if df is None or df.empty:
        raise ValueError("Couldn't download data for symbol: " + symbol)

    close = df['Close'].values.reshape(-1,1)

    # Load scaler and model if present; otherwise train
    scaler = load_scaler()
    model = None
    if os.path.exists(MODEL_PATH) and scaler is not None:
        try:
            model = load_model(MODEL_PATH)
        except Exception:
            model = None

    if model is None or scaler is None:
        # Train a fresh model (takes time)
        model, scaler = train_model(symbol, epochs=10)

    scaled = scaler.transform(close)

    # Use last 60 points as input sequence
    seq_len = 60
    if len(scaled) < seq_len:
        raise ValueError("Not enough data to make prediction (need at least 60 days).")

    input_seq = scaled[-seq_len:].tolist()
    preds = []
    for _ in range(days_ahead):
        x_input = np.array(input_seq[-seq_len:]).reshape(1, seq_len, 1)
        pred_scaled = model.predict(x_input, verbose=0)[0][0]
        preds.append(pred_scaled)
        # append predicted scaled value for rolling forecast
        input_seq.append([pred_scaled])

    preds = np.array(preds).reshape(-1,1)
    inv = scaler.inverse_transform(preds)
    inv_list = [float(x[0]) for x in inv]
    return inv_list