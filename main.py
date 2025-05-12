from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.utils
import json

# Assuming Regression.py uses company name or maps internally
from Regression import StockPriceForecaster

app = Flask(__name__, static_url_path="/static", static_folder="static")
CORS(app, resources={r"/*": {"origins": "http://localhost:8000"}})


@app.route("/")
def serve_graphic():
    return send_file("grafico.html")


def generate_trading_dates(start_date, n_days):
    """Generate past trading days (skip weekends)"""
    dates = []
    current = start_date
    while len(dates) < n_days:
        if current.weekday() < 5:  # Mon-Fri
            dates.append(current)
        current -= timedelta(days=1)
    return dates[::-1]  # Oldest first


@app.route("/chart")
def chart():
    company_name = request.args.get("stock")  # "amazon", "apple", etc.
    target_date_str = request.args.get("date")

    if not company_name or not target_date_str:
        return jsonify({"error": "Missing parameters"}), 400

    try:
        target_date = datetime.strptime(target_date_str, "%Y-%m-%d")

        # Map company name to symbol if needed
        company_map = {"amazon": "AMZN", "apple": "AAPL", "google": "GOOGL"}
        ticker = company_map.get(company_name.lower(), company_name.upper())

        forecaster = StockPriceForecaster(ticker)
        forecaster.build_model()
        forecaster.load_model()
        predicted_price, last_sequence = forecaster.predict_for_given_day(
            target_date_str
        )

        values = [row[0] for row in last_sequence]
        sequence_length = len(values)

        dates = generate_trading_dates(target_date - timedelta(days=1), sequence_length)
        pred_date = target_date
        pred_date = target_date + timedelta(days=0.1)
        # Create Plotly figure
        fig = go.Figure()
        last_value = values[-1]
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=values,
                mode="lines+markers",
                name="Historical Value",
                line=dict(color="blue"),
                marker=dict(size=6),
                hovertemplate="<b>Date:</b> %{x}<br><b>Value:</b> $%{y:.2f}<extra></extra>",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=[dates[-1], pred_date],
                y=[last_value, predicted_price],
                mode="lines",
                name="Prediction Trend",
                line=dict(
                    color="red",
                    dash="10px, 40px"  # Custom dash pattern: short dash, long gap
                ),
                hoverinfo="skip",  # Skip this trace in hover tooltips
                showlegend=False,  # Hide from legend
            )
        )

        fig.add_trace(
            go.Scatter(
                x=[pred_date],
                y=[predicted_price],
                mode="markers",
                name="Predicted Value",
                marker=dict(color="red", size=10),
                hovertemplate="<b>Date:</b> %{x}<br><b>Predicted Price:</b> $%{y:.2f}<extra></extra>",
            )
        )

        fig.update_layout(
            title=f"{company_name.capitalize()} - Prediction for {pred_date.strftime('%Y-%m-%d')}",
            xaxis_title="Date",
            yaxis_title="Stock Price",
            hovermode="closest",
            xaxis=dict(
                showspikes=True,
                spikedash="dot",  # Dashed line style
                spikemode="across",  # Extend the spike across the entire plot
                spikecolor="gray",  # Line color
                gridcolor="lightgray",  # Grid line color
                showgrid=True,  # Show grid lines
                zeroline=False,  # Hide the zero line
            ),
            # yaxis=dict(showspikes=True),
            template="plotly_white",
            hoverlabel=dict(  # 👈 Customize hover tooltip
                bgcolor="white", bordercolor="black", font_size=14, font_color="black"
            ),
        )
        # fig.update_layout(hovermode="closest")

        graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

        return jsonify(
            {
                "graph": graphJSON,
                "predicted_price": round(predicted_price, 2),
                "company": company_name.capitalize(),
                "date": pred_date.strftime("%Y-%m-%d"),
            }
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500
if __name__ == "__main__":
    app.run(debug=True)
