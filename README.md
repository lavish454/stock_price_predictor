# Stock Price Predictor (Local Flask App)

## What it does
- Simple Flask web app that accepts a stock symbol and number of days (1-7).
- Uses an LSTM model to predict future closing prices.
- If a trained model is not present, the app will train one using 1 year of historical data (this may take time).

## Setup (local)
1. Create a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate    # on Windows: venv\Scripts\activate
   ```
2. Install requirements:
   ```
   pip install -r requirements.txt
   ```
3. Run the app:
   ```
   python app.py
   ```
4. Open `http://127.0.0.1:5000` in your browser.

## Notes
- Training may take several minutes depending on your machine and internet connection.
- The app saves `model.h5` and `scaler.npy` in the project root after training.
- For production or faster usage, pretrain the model and place `model.h5` in the project folder.