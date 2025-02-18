from __future__ import annotations
import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import joblib
import os

# Load Models
model_old = joblib.load(os.path.join(os.getcwd(), "model", "a1_car_price.pkl"))
model_new = joblib.load(os.path.join(os.getcwd(), "model", "a2_car_price.pkl"))
scaler_model = joblib.load(os.path.join(os.getcwd(), "model", "scaler.dump"))

# Dash App Setup
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.LUX])
app.title = "Car Price Predictor"

# App Layout
app.layout = dbc.Container([
    html.H1("Car Price Prediction", className="text-center mt-4 text-primary fw-bold"),
    
    # Notice Message at the Top
    dbc.Alert([
        html.H5("⚡ New Model Update!", className="fw-bold"),
        html.P("We now have two models for predicting car prices."),
        html.Ul([
            html.Li("The 🔵 **Old Model** is based on basic regression."),
            html.Li("The 🟢 **New Model** uses an improved machine learning approach with better accuracy."),
            html.Li("For better results, we recommend using the **New Model**."),
        ], className="mb-0"),
    ], color="info", className="shadow-sm text-dark"),
    
    html.P("Enter car details to predict the price using two models.", className="text-center text-muted mb-4"),

    # Card Layout with More Spacious Styling
    dbc.Card([
        dbc.CardBody([
            # Input Fields with More Padding and Clear Labels
            dbc.Row([
                dbc.Col([
                    dbc.Label("Year of Manufacture", className="fw-semibold text-muted"),
                    dbc.Input(id="input-year", type="number", placeholder="e.g., 2015", min=1900, max=2025, step=1, className="form-control"),
                ], width=6, className="mb-3"),
                dbc.Col([
                    dbc.Label("Mileage (kmpl)", className="fw-semibold text-muted"),
                    dbc.Input(id="input-mileage", type="number", placeholder="Mileage in kmpl", min=0, className="form-control"),
                ], width=6, className="mb-3"),
            ]),
            
            dbc.Row([
                dbc.Col([
                    dbc.Label("Max Power (bhp)", className="fw-semibold text-muted"),
                    dbc.Input(id="input-max-power", type="number", placeholder="Max Power in bhp", min=0, className="form-control"),
                ], width=6, className="mb-3"),
                dbc.Col([
                    dbc.Label("Engine (cc)", className="fw-semibold text-muted"),
                    dbc.Input(id="input-engine", type="number", placeholder="Engine size in cc", min=0, className="form-control"),
                ], width=6, className="mb-3"),
            ]),
            
            # Prediction Buttons with Better Layout
            dbc.Row([
                dbc.Col([
                    dbc.Button("Predict (Old Model)", id="predict-button-old", color="primary", className="w-100 py-3 fw-bold"),
                ], width=6, className="mb-3"),
                dbc.Col([
                    dbc.Button("Predict (New Model)", id="predict-button-new", color="success", className="w-100 py-3 fw-bold"),
                ], width=6, className="mb-3"),
            ]),

        ])
    ], className="shadow-lg p-4 rounded-3 border-0 bg-white mb-5"),
    
    # Prediction Output with More Breathing Room
    dbc.Row(
        dbc.Col([
            html.H4("Predicted Price:", className="mt-4 text-primary fw-semibold"),
            dbc.Spinner(html.Div(id="output-prediction", className="alert alert-info mt-3"), color="primary"),
        ], width=12, className="text-center"),
    ),
], fluid=True, className="d-flex flex-column align-items-center justify-content-start py-1")

# Callback for Prediction
@app.callback(
    Output("output-prediction", "children"),
    [Input("predict-button-old", "n_clicks"), Input("predict-button-new", "n_clicks")],
    [State("input-year", "value"), State("input-mileage", "value"),
     State("input-max-power", "value"), State("input-engine", "value")]
)
def predict_price(n_clicks_old, n_clicks_new, year, mileage, max_power, engine):
    ctx = dash.callback_context
    if not ctx.triggered:
        return "Click a button to predict price."
    if None in [year, mileage, max_power, engine]:
        return "Please provide all input values."
    
    # Determine which button was clicked
    button_id = ctx.triggered[0]["prop_id"].split(".")[0]
    
    # Choose Model Based on Button Click
    model = model_old if button_id == "predict-button-old" else model_new
    
    # Prepare input data
    input_data = pd.DataFrame({
        'engine': [engine],
        'max_power': [max_power],
        'mileage': [mileage],
        'year': [year],
    })
    
    try:
        scaled_data = scaler_model.transform(input_data)
        pred_log = model.predict(scaled_data)
        pred_price = np.exp(pred_log[0])
        return f"Predicted Price by {'New Model' if model == model_new else 'Old Model'}: {pred_price:,.2f} Baht"

    except Exception as e:
        return f"Error in prediction: {e}"

if __name__ == "__main__":
    app.run_server(debug=True)
