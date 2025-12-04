# Importamos las librerias necesarias
import dash
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from dash import html, dcc
from dash.dependencies import Input, Output
from dash.exceptions import PreventUpdate
from plotly.subplots import make_subplots
from src.grafica1 import registrar_callback_radar
from src.grafica2_3 import registrar_callbacks_mercado

from src.modelos.modelo import predecir_entreuvas
from pathlib import Path
import logging

DATA_SP = Path("src/data/wines_SPA.csv")
df_spain_raw = pd.read_csv(DATA_SP)
df_spain = df_spain_raw.copy()


# Inicializamos la app con un tema limpio
app = dash.Dash(__name__)


# Drop columnas no usadas
for col in ["acidity", "body"]:
    if col in df_spain.columns:
        df_spain = df_spain.drop(columns=[col])

# Escala de rating 4.2-4.9 -> 3-9 + ruido
if "rating" in df_spain.columns:
    scale = (9 - 3) / (4.9 - 4.2)
    df_spain["rating_scaled"] = 3 + (df_spain["rating"] - 4.2) * scale
    df_spain["rating_scaled"] += np.random.uniform(-0.5, 0.5, size=len(df_spain))
    df_spain["rating_scaled"] = df_spain["rating_scaled"].clip(3, 9)
else:
    df_spain["rating_scaled"] = pd.Series(dtype=float)

# Color simplificado
df_spain["color"] = (
    df_spain.get("color")
    if "color" in df_spain.columns
    else df_spain.get("type", pd.Series(dtype=str)).str.lower().fillna("").apply(
        lambda x: "red" if "red" in x else ("white" if "white" in x or "blanco" in x else "other")
    )
)

PRICE_MIN = float(df_spain["price"].min()) if not df_spain.empty else 3.0
PRICE_MAX = float(1000.0)

app.layout = html.Div(
    children=[
        # ================================
        #  BANNER NUEVO ENTREUVAS
        # ================================
        html.Div(
            className="banner-wrapper",
            children=[
                html.Div(
                    className="banner",
                    style={"background-image": "url('/assets/vinedo2.png')"},
                    children=[
                        html.Img(src="/assets/logo_uvas0.png", className="banner-logo"),
                        html.Div(
                            className="banner-text-container",
                            children=[
                                html.H1("EntreUvas", className="banner-title"),
                                html.P(
                                    "La plataforma que acompana a tu bodega en cada eleccion.",
                                    className="banner-sub",
                                ),
                                html.P(
                                    "Analiza tu vino y descubre su potencial en el mercado.",
                                    className="banner-sub",
                                ),
                                html.Div(
                                    className="banner-buttons",
                                    children=[
                                        html.A(
                                            children=[
                                                html.Img(src="/assets/logo_uvas1.png"),
                                                "Predice la calidad de tu vino",
                                            ],
                                            href="#predictor",
                                            className="banner-button banner-btn-green",
                                        ),
                                        html.A(
                                            children=[
                                                html.Img(src="/assets/logo_graficas0.png"),
                                                "Compara con el mercado",
                                            ],
                                            href="#market",
                                            className="banner-button banner-btn-red",
                                        ),
                                    ],
                                ),
                            ],
                        ),
                    ],
                ),
            ],
        ),

        # ================================
        #  SECCION: PREDICCION DE CALIDAD
        # ================================
        html.Div(
            id="predictor",
            className="section",
            children=[
                html.Img(src="/assets/logo_uvas2.png", className="pred-logo"),
                
                html.Div(
                    children=[
                        html.H2("Analiza tu vino y descubre su potencial:", className="pred-title"),
                        html.H4("Introduce los siguientes parámetros:", className="pred-sub"),
                    ]
                ),
            ],
        ),

        # ======= FILA PRINCIPAL (IZQ + DER) =======
        html.Div(
            className="section",
            children=[
        
                # -------- IZQUIERDA --------
                html.Div(
                    className="section-left",
                    children=[
                        
                        html.Div(
                            className="card form-card",
                            children=[
                                html.Div(
                                    [
                                        html.Label("Alcohol", className="slider-label"),
                                        dcc.Slider(
                                            id="sl-alcohol",
                                            min=5,
                                            max=20,
                                            step=0.1,
                                            value=10,
                                            marks={5: "5", 20: "20"},
                                        ),
                                        html.Span(id="val-alcohol", className="slider-value"),
                                    ],
                                    className="slider-block",
                                ),
                                html.Div(
                                    [
                                        html.Label("Volatile Acidity", className="slider-label"),
                                        dcc.Slider(
                                            id="sl-vol-acid",
                                            min=0.0,
                                            max=2.0,
                                            step=0.01,
                                            value=0.5,
                                            marks={0.1: "0.1", 1.999999: "2.0"},
                                        ),
                                        html.Span(id="val-acid", className="slider-value"),
                                    ],
                                    className="slider-block",
                                ),
                                html.Div(
                                    [
                                        html.Label("Free SO2", className="slider-label"),
                                        dcc.Slider(
                                            id="sl-free-sulfur",
                                            min=0,
                                            max=300,
                                            step=1.0,
                                            value=50,
                                            marks={0: "0", 300: "300"},
                                        ),
                                        html.Span(id="val-free-sulfur", className="slider-value"),
                                    ],
                                    className="slider-block",
                                ),
                                html.Div(
                                    [
                                        html.Label("Sulphates", className="slider-label"),
                                        dcc.Slider(
                                            id="sl-sulphates",
                                            min=0.0,
                                            max=5.0,
                                            step=0.01,
                                            value=0.5,
                                            marks={0: "0", 5: "5"},
                                        ),
                                        html.Span(id="val-sulphates", className="slider-value"),
                                    ],
                                    className="slider-block",
                                ),
                                html.Div(
                                    [
                                        html.Label("Residual Sugar", className="slider-label"),
                                        dcc.Slider(
                                            id="sl-res-sugar",
                                            min=0,
                                            max=80,
                                            step=0.1,
                                            value=5,
                                            marks={0: "0", 80: "80"},
                                        ),
                                        html.Span(id="val-res-sugar", className="slider-value"),
                                    ],
                                    className="slider-block",
                                ),
                                html.Button("Calcular calidad", id="btn-calcular", className="btn-primary"),
                                html.Div(
                                    children=[
                                        html.Div(id="resultado-calidad", className="result-box"),
                                        html.P(id="estado-prediccion", className="pred-status"),
                                    ]
                                ),
                            ],
                        ),
                    ],
                ),

                # -------- DERECHA --------
                html.Div(
                    className="section-right",
                    children=[
                        html.Div(
                            className="card graph-card",
                            children=[
                                html.H3("Grafico de calidad proyectada", className="graph-title"),
                                html.Div(
                                    [
                                        html.Label("Referencia de mercado", className="slider-label"),
                                        dcc.Dropdown(
                                            id="dd-mercado",
                                            options=[
                                                {"label": "Combinado (todo)", "value": "combinado"},
                                                {"label": "Tinto", "value": "red"},
                                                {"label": "Blanco", "value": "white"},
                                            ],
                                            value="combinado",
                                            clearable=False,
                                            className="dropdown-mercado",
                                        ),
                                    ],
                                    className="dropdown-block",
                                ),
                                dcc.Graph(
                                    id="graph-calidad",
                                    config={"displayModeBar": False},
                                    style={"height": "340px"},
                                ),
                            ],
                        ),
                        html.Div(
                            className="card image-card",
                            children=html.Img(src="/assets/uvas0.png", className="graph-image"),
                        ),
                    ],
                ),
            ],
        ),
    



        # ================================
        #  SECCION: MERCADO (placeholder)
        # ================================
        html.Div(
            id="market",
            className="market-section",
            children=[
                html.Div(
                    className="market-hero card",
                    children=[
                        html.Div(
                            className="market-hero-header",
                            children=[
                                html.Img(src="/assets/logo_uvas2.png", className="pred-logo"),
                                html.Div(
                                    children=[
                                        html.H2("Compara tu vino con el mercado:", className="pred-title"),
                                        html.P("Explora su posición según calidad, precio y color.", className="pred-sub"),
                                    ],
                                ),
                            ],
                        ),
                        html.Img(src="/assets/Botellas.png", className="market-hero-img"),
                    ],
                ),

                html.Div(
                    className="market-controls card",
                    children=[
                            html.Div(className="market-control-block", children=[
                                html.Label("Rango de precio (€)", className="slider-label"),
                                dcc.RangeSlider(
                                    id="rs-precio",
                                    min=PRICE_MIN, max=PRICE_MAX, step=1, value=[max(PRICE_MIN, 10), min(PRICE_MAX, 80)],
                                    marks={int(PRICE_MIN): f"{int(PRICE_MIN)}", int(PRICE_MAX): f"{int(PRICE_MAX)}"},
                                    tooltip={"placement": "bottom", "always_visible": False},
                                    allowCross=False,
                                ),
                            ]),
                        html.Div(className="market-control-row", children=[
                            html.Div(className="market-input", children=[
                                html.Label("Color", className="slider-label"),
                                dcc.Dropdown(
                                    id="dd-market-color",
                                    options=[
                                        {"label": "Todos", "value": "todos"},
                                        {"label": "Tinto", "value": "red"},
                                        {"label": "Blanco", "value": "white"},
                                    ],
                                    value="todos",
                                    clearable=False,
                                ),
                            ]),
                            html.Div(className="market-input", children=[
                                html.Label("Calidad estimada (3-9)", className="slider-label"),
                                dcc.Input(
                                    id="input-my-rating",
                                    type="number",
                                    min=3, max=9, step=0.1,
                                    value=7.0,
                                    className="market-number",
                                ),
                            ]),
                            html.Div(className="market-input", children=[
                                html.Label("Precio estimado (€)", className="slider-label"),
                                dcc.Input(
                                    id="input-my-price",
                                    type="number",
                                    min=3, max=300, step=0.5,
                                    value=15.0,
                                    className="market-number",
                                ),
                            ]),
                        ]),
                    ],
                ),

                html.Div(
                    className="market-grid",
                    children=[
                        html.Div(
                            className="card market-card",
                            children=[
                                html.H3("Precio vs Calidad (mercado vs tu vino)", className="market-card-title"),
                                dcc.Graph(
                                    id="graph-mercado-scatter",
                                    config={"displayModeBar": False},
                                    style={"height": "240px"},
                                ),
                            ],
                        ),
                        html.Div(
                            className="card market-card",
                            children=[
                                html.H3("Vinos más similares a tu vino", className="market-card-title"),
                                html.Div(id="table-vecinos", className="market-table"),
                            ],
                        ),
                    ],
                ),
            ],
        ),
    ]
)

registrar_callback_radar(app)
registrar_callbacks_mercado(app)



@app.callback(
    [
        Output("val-alcohol", "children"),
        Output("val-acid", "children"),
        Output("val-free-sulfur", "children"),
        Output("val-sulphates", "children"),
        Output("val-res-sugar", "children"),
    ],
    [
        Input("sl-alcohol", "value"),
        Input("sl-vol-acid", "value"),
        Input("sl-free-sulfur", "value"),
        Input("sl-sulphates", "value"),
        Input("sl-res-sugar", "value"),
    ],
    )
def actualizar_valores(alcohol, acid, free_s, sulph, sugar):
    return (
        f"Tu valor seleccionado es: {alcohol:.2f}",
        f"Tu valor seleccionado es: {acid:.2f}",
        f"Tu valor seleccionado es: {free_s:.0f}",
        f"Tu valor seleccionado es: {sulph:.2f}",
        f"Tu valor seleccionado es: {sugar:.1f}",
    )


@app.callback(
    [Output("resultado-calidad", "children"), Output("estado-prediccion", "children")],
    [
        Input("btn-calcular", "n_clicks"),
        Input("sl-alcohol", "value"),
        Input("sl-vol-acid", "value"),
        Input("sl-free-sulfur", "value"),
        Input("sl-sulphates", "value"),
        Input("sl-res-sugar", "value"),
    ],
    )
def actualizar_prediccion(n, alcohol, vol_acid, free_sulfur, sulphates, res_sugar):
    ctx = dash.callback_context

    if ctx.triggered[0]["prop_id"] != "btn-calcular.n_clicks":
        return "", "A la espera de calcular..."

    if n is None:
        raise PreventUpdate

    pred, conf = predecir_entreuvas(alcohol, vol_acid, free_sulfur, sulphates, res_sugar)

    resultado = html.Div(
        [
            html.Div(f"Calidad estimada: {pred:.1f}/10"),
            html.Div(f"Confianza del modelo: {conf:.0f}%"),
        ]
    )

    return resultado, ""


if __name__ == "__main__":
    app.run(debug=True)
