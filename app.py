from flask import Flask, render_template, request
import os, random, traceback

app = Flask(__name__)

def predict_future_prices(stock, days):
    prices = []
    base_price = random.uniform(100, 500)
    for i in range(days):
        change = random.uniform(-5, 5)
        base_price += change
        prices.append(round(base_price, 2))
    return prices

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        stock = request.form.get('stock', '').strip().upper()
        days = request.form.get('days', '').strip()

        if not stock:
            return render_template('index.html', error="Please enter a stock symbol.")
        if not days.isdigit():
            return render_template('index.html', error="Days must be a number between 1 and 7.")

        days = int(days)
        if days < 1 or days > 7:
            return render_template('index.html', error="Please enter days between 1 and 7.")

        predictions = predict_future_prices(stock, days)
        return render_template('index.html', stock=stock, days=days, predictions=predictions)

    except Exception as e:
        tb = traceback.format_exc()
        return render_template('index.html', error=f"Error: {str(e)}\n{tb}")

import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)

