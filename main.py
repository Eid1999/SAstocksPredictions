from flask import Flask, request, send_file, send_from_directory, Response
from flask_cors import CORS
import datetime
from Regression import StockPriceForecaster

app = Flask(__name__, static_url_path='/static', static_folder='static')
CORS(app)

@app.route('/chart')
def chart():
    stock = request.args.get('stock', 'apple')
    start_year = int(request.args.get('startYear'))
    start_month = int(request.args.get('startMonth'))
    end_year = int(request.args.get('endYear'))
    end_month = int(request.args.get('endMonth'))

    # Construct date range
    start_date = datetime.date(start_year, start_month, 1)
    end_date = datetime.date(end_year, end_month, 28)  # fallback day

    # Forecast
    forecaster = StockPriceForecaster(company=stock, start_date=start_date, end_date=end_date)
    forecaster.prepare_data()
    forecaster.create_tft_dataset()
    forecaster.build_model()
    forecaster.train_model(max_epochs=5)  # lower for speed
    image_buf = forecaster.plot_prediction()

    return send_file(image_buf, mimetype='image/png')


@app.route('/') 
def send_js(path):
    return send_from_directory('static', "grafico.html")


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080, debug=True)
