import yfinance as yf
import numpy as np
import ML
from flask import Flask, render_template, request, redirect, url_for, jsonify
app = Flask(__name__)



tickers = yf.Tickers('aapl goog meta nflx amzn')
aapl_df = tickers.tickers['AAPL'].history(period="5y").iloc[:,:6]
goog_df = tickers.tickers['GOOG'].history(period="5y").iloc[:,:6]
meta_df = tickers.tickers['META'].history(period="5y").iloc[:,:6]
nflx_df = tickers.tickers['NFLX'].history(period="5y").iloc[:,:6]
amzn_df = tickers.tickers['AMZN'].history(period="5y").iloc[:,:6]


@app.route("/", methods=['GET', 'POST'])
def index():
    return render_template('index.html')

@app.route('/get_data', methods=['GET'])
def get_data():
  answer = ""
  selected_value = request.args.get('selected_value')
  ticker = str.capitalize(selected_value)
  data, tp = generate_features(ticker)
  res = train_and_predict(data, tp)
  if res >= 0.5:
    answer = "Uptrend"
  else:
    answer = "Downtrend"
  final = f'Your selected stock will be in the {answer} !'
  response_data = {'selected_value': final, 'message': 'Received your request!'}
  return jsonify(response_data)
  
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
