from __future__ import annotations
import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
from itertools import product
import pandas as pd
import numpy as np
import joblib
import os

# Load Models
model_old = joblib.load(os.path.join(os.getcwd(), "model", "a1_car_price.pkl"))
model_new = joblib.load(os.path.join(os.getcwd(), "model", "a2_car_price.pkl"))
scaler_model = joblib.load(os.path.join(os.getcwd(), "model", "scaler.dump"))
# print("Model Old:", model_old)
# print("New Model", model_new)
print('helo yeta ougey')

# try:
#     model_old = joblib.load(model_path_old)
#     model_new = joblib.load(model_path_new)
#     scaler_model = joblib.load(scaler_path)
#     print("Models loaded successfully!")
# except Exception as e:
#     print(f"Error loading models: {e}")
#     exit()

# Dash App Setup
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Car Price Predictor"

# Card Layout for the Form
app.layout = dbc.Container([
    html.H1("Car Price Prediction", className="text-center my-4"),
    html.P("Enter car details below to predict its price using two models.", className="text-center"),
    
    # Card for the Form
    dbc.Card([
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    dbc.Label("Year of Manufacture"),
                    dbc.Input(id="input-year", type="number", placeholder="Enter the year", min=1900, max=2025, step=1),
                ], width=6),
                dbc.Col([
                    dbc.Label("Mileage (kmpl)"),
                    dbc.Input(id="input-mileage", type="number", placeholder="Enter the mileage in kmpl", min=0),
                ], width=6),
            ], className="mb-3"),
            
            dbc.Row([
                dbc.Col([
                    dbc.Label("Max Power (bhp)"),
                    dbc.Input(id="input-max-power", type="number", placeholder="Enter max power in bhp", min=0),
                ], width=6),
                dbc.Col([
                    dbc.Label("Engine (cc)"),
                    dbc.Input(id="input-engine", type="number", placeholder="Enter engine size in cc", min=0),
                ], width=6),
            ], className="mb-3"),
            
            dbc.Row([
                dbc.Col([
                    dbc.Button("Predict (Old Model)", id="predict-button-old", color="primary", className="mt-3 w-100"),
                ], width=6),
                dbc.Col([
                    dbc.Button("Predict (New Model)", id="predict-button-new", color="success", className="mt-3 w-100"),
                ], width=6),
            ], className="mb-4"),
        ])
    ], className="shadow p-3 mb-5 bg-white rounded mx-auto"), 
    
    # Prediction Output
    dbc.Row(
        dbc.Col([
            html.H4("Predicted Price:", className="mt-4"),
            dbc.Spinner(html.Div(id="output-prediction", className="alert alert-info"), color="primary"),
        ], width=12),
    ),
], fluid=True)

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
        try:
            scaled_data = scaler_model.transform(input_data)
        except Exception as e:
            return f"Error in scaling data: {e}"
        
        print("Scaled passed")
        pred_log = model.predict(scaled_data)
        pred_price = np.exp(pred_log[0])
        return f"Predicted Price: {pred_price:,.2f} Baht"
    except Exception as e:
        return f"Error in prediction: {e}"

# Run the app
if __name__ == "__main__":
    app.run_server(debug=True)