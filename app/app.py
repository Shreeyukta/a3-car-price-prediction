import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import joblib
import os

model_path = "./model/a1_car_price.pkl"
model = joblib.load(model_path)
print("Model loaded successfully!")
scaler_model = joblib.load("./model/scaler.dump")

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Car Price Predictor"

app.layout = dbc.Container([
    html.H1("Car Price Prediction", className="text-center my-4"),
    html.H2("Instructions", className="text-center"),
    html.P(
        "To predict the car price, enter maximum power, engine, mileage, and year. ",
        className="text-center"
    ),
    dbc.Row(
        dbc.Col(
            [
                dbc.Row([
                    dbc.Col([
                        dbc.Label("Year of Manufacture"),
                        dbc.Input(id="input-year", type="number", placeholder="Enter the year", min=1900, max=2025, step=1),
                            dbc.Tooltip("Please enter a year between 1900 and 2025.", target="input-year", placement="right", id="year-tooltip", is_open=False)
                    ], width=6),

                    dbc.Col([
                        dbc.Label("Mileage (kmpl)"),
                        dbc.Input(id="input-mileage", type="number", placeholder="Enter the mileage in kmpl", min=0),
                    ], width=6),
                ], className="mb-3"),

                dbc.Row([
                    dbc.Col([
                        dbc.Label("Max Power (bhp)"),
                        dbc.Input(id="input-max-power", type="number", placeholder="Enter the max power in bhp", min=0),
                    ], width=6),

                    dbc.Col([
                        dbc.Label("Engine (cc)"),
                        dbc.Input(id="input-engine", type="number", placeholder="Enter the engine size in cc", min=0),
                    ], width=6),
                ], className="mb-3"),

                dbc.Row([
                    dbc.Col([
                        dbc.Button("Predict", id="predict-button", color="primary", className="mt-3"),
                    ], width=12, className="text-center"),
                ]),

                dbc.Row([
                    dbc.Col([
                        html.H4("Predicted Price:", className="mt-4"),
                        html.Div(id="output-prediction", className="alert alert-info"),
                    ], width=12),
                ]),
            ],
            width=8,  
            className="shadow p-4 bg-light rounded mx-auto"  
        ),
        justify="center", 
        className="my-4"  
    )
], fluid=True)



@app.callback(
    Output("output-prediction", "children"),
    Input("predict-button", "n_clicks"),
    State("input-year", "value"),
    State("input-mileage", "value"),
    State("input-max-power", "value"),
    State("input-engine", "value"),
)
def predict_price(n_clicks, year, mileage, max_power, engine):
    if n_clicks is None or None in [year, mileage, max_power, engine]:
        return "Please provide all input values."

    input_data = {
        'engine': [engine],
        'max_power': [max_power],
        'mileage': [mileage],
        'year': [year],
    }

    df = pd.DataFrame(input_data)

    try:
        scaled_data = scaler_model.transform(df)

        
        pred_log = model.predict(scaled_data)
        print("Predicted log price:", pred_log)

        pred_price = np.exp(pred_log[0])

        return f"Predicted Price: {pred_price:,.2f} Baht"
    except Exception as e:
        return f"Error in prediction: {e}"

if __name__ == "__main__":
    app.run_server(debug=True)