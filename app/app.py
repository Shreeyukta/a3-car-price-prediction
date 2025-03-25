import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import joblib
import os

# Load Models
# model_new = joblib.load(os.path.join(os.getcwd(), "model", "a3_car_price.pkl"))
model_new = joblib.load("./model/a3_car_price.pkl")
scaler_model = joblib.load(os.path.join(os.getcwd(), "model", "scaler.dump"))

# Dash App Setup
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.LUX])
app.title = "Car Price Predictor"

# App Layout
app.layout = dbc.Container([
    html.H1("Car Price Category Prediction Model", className="text-center mt-4 text-primary fw-bold"),
    
    dbc.Alert([
        html.H5("", className="fw-bold text-center"),
        html.P("This model predicts the price category of a car based on features like year, mileage, max power, and engine size."),
        html.Ul([
            html.Li([
                "Category ", html.Strong("0"), ": Budget cars (lowest price range)"
            ]),
            html.Li([
                "Category ", html.Strong("1"), ": Affordable cars (slightly higher range)"
            ]),
            html.Li([
                "Category ", html.Strong("2"), ": Mid-range cars (moderately priced)"
            ]),
            html.Li([
                "Category ", html.Strong("3"), ": Premium cars (highest price range)"
            ]),
        ])
    ], color="info", className="shadow-sm text-dark"),

    
    html.P("Enter car details to predict the price using two models.", className="text-center text-muted mb-4"),

    dbc.Card([
        dbc.CardBody([
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
            
            dbc.Row([
                dbc.Col([
                    dbc.Button("Predict", id="predict-button-new", color="primary", className="px-5 py-3 fw-bold")
                ], width="auto")
            ], justify="center", className="mb-3"),

        ])
    ], className="shadow-lg p-4 rounded-3 border-0 bg-white mb-5"),
    
    dbc.Row(
        dbc.Col([
            html.H4("Predicted Price Category:", className="mt-4 text-primary fw-semibold"),
            dbc.Spinner(html.Div(id="output-prediction", className="alert alert-info mt-3"), color="primary"),
        ], width=12, className="text-center"),
    ),
], fluid=True, className="d-flex flex-column align-items-center justify-content-start py-1")

@app.callback(
    Output("output-prediction", "children"),
    Input("predict-button-new", "n_clicks"),
    [State("input-year", "value"), State("input-mileage", "value"),
     State("input-max-power", "value"), State("input-engine", "value")]
)
def predict_price(n_clicks_new, year, mileage, max_power, engine):
    import dash
    import pandas as pd
    import numpy as np
    import traceback

    ctx = dash.ctx if hasattr(dash, 'ctx') else dash.callback_context
    if not ctx.triggered:
        return "Click a button to predict price."
    if None in [year, mileage, max_power, engine]:
        return "Please provide all input values."

    button_id = ctx.triggered[0]["prop_id"].split(".")[0]
    model =  model_new
    
    input_data = pd.DataFrame({
        'engine': [engine],
        'max_power': [max_power],
        'mileage': [mileage],
        'year': [year],
    })

    try:
        scaled_data = scaler_model.transform(input_data)
        pred = model.predict(scaled_data)
        pred_price = (pred[0])
        return f"Category : {pred_price}"

    except Exception as e:
        return f"Error in prediction: {e}"

if __name__ == "__main__":
    app.run(debug=True)
