from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from Regression import StockPriceForecaster
import matplotlib.pyplot as plt
import numpy as np
import io
import base64
import pdb
import seaborn as sns

app = Flask(__name__, static_url_path='/static', static_folder='static')
CORS(app)
def return_stock_symbol(company_name):
    if company_name == "apple":
        return "AAPL"
    elif company_name == "microsoft":
        return "MSFT"
    elif company_name == "google":
        return "GOOGL"
    elif company_name == "amazon":
        return "AMZN"
    elif company_name == "tesla":
        return "TSLA"
    else:
        raise ValueError("Unknown company name")


@app.route('/chart')
def chart():
    company = request.args.get('stock')
    target_date = request.args.get('date', '2023-10-01')
    if not company:
        return jsonify({'error': 'Missing stock parameter'}), 400

    try:
        # Use your real forecasting class
        forecaster = StockPriceForecaster(return_stock_symbol(company))
        forecaster.build_model()
        forecaster.load_model()
        predicted_price, last_sequence = forecaster.predict_for_given_day(target_date)

        # Plot the first feature (e.g., closing price)
        values = [row[0] for row in last_sequence]
        steps = np.arange(len(values))

        # Set style
        sns.set_style("whitegrid")

        plt.figure(figsize=(12, 6))

        # Scatter plot for historical data
        sns.scatterplot(x=steps, y=values, color='blue', label='Historical Data Points', zorder=2)

        # Line connecting points
        sns.lineplot(x=steps, y=values, color='lightgray', linestyle='-', linewidth=1, zorder=1)

        # Red line from last known point to prediction
        last_step = len(values) - 1
        last_value = values[-1]
        predicted_step = last_step + 1

        # Draw red dashed trend line to predicted value
        plt.plot([last_step, predicted_step], [last_value, predicted_price],
                color='red', linestyle='--', linewidth=2, label='Prediction Trend', zorder=3)

        # Mark predicted point
        sns.scatterplot(x=[predicted_step], y=[predicted_price],
                        color='red', s=100, label='Predicted Value', zorder=3)

        # Labels and title
        plt.xlabel("Sequence Step", fontsize=12)
        plt.ylabel("Value", fontsize=12)
        plt.title(f"Stock Sequence for {company.upper()} - Prediction for {target_date}", fontsize=14)
        plt.legend()
        plt.tight_layout()

        img_io = io.BytesIO()
        plt.savefig(img_io, format='png', bbox_inches='tight')
        plt.close()
        img_io.seek(0)

        # Convert to base64 for frontend
        base64_img = base64.b64encode(img_io.read()).decode('utf-8')
        image_uri = f"data:image/png;base64,{base64_img}"

        return jsonify({
            'image': image_uri,
            'predicted_price': round(predicted_price, 2),
            'company': company.upper(),
            'date': target_date
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/') 
def send_js():
    return send_from_directory('./', "grafico.html")


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080, debug=True)
