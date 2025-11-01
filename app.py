from flask import Flask, render_template, request
from stock_model import predict_future_prices
import traceback

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    stock = request.form.get('stock', '').strip().upper()
    days = int(request.form.get('days', 1))
    if days < 1 or days > 7:
        return render_template('index.html', error='Please enter days between 1 and 7.')
    if not stock:
        return render_template('index.html', error='Please enter a stock symbol.')

    try:
        preds = predict_future_prices(stock, days)
        # preds is a list of floats
        return render_template('index.html', stock=stock, days=days, predictions=preds)
    except Exception as e:
        tb = traceback.format_exc()
        return render_template('index.html', error=f'Error: {str(e)}\n{tb}')

import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
