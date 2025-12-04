import plotly.graph_objects as go
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import pandas as pd
from pathlib import Path

BASE = Path(__file__).resolve().parent
DATA = BASE / "data" / "winequality-combined.csv"

df = pd.read_csv(DATA)

def _medias(sub):
    return {
        "alcohol": sub["alcohol"].mean(),
        "vol_acid": sub["volatile_acidity"].mean(),
        "free_sulfur": sub["free_sulfur_dioxide"].mean(),
        "sulphates": sub["sulphates"].mean(),
        "res_sugar": sub["residual_sugar"].mean(),
    }

REFS = {
    "combinado": _medias(df),
    "red": _medias(df[df["type"] == "red"]),
    "white": _medias(df[df["type"] == "white"]),
}

def _features(tipo):
    ref = REFS.get(tipo, REFS["combinado"])
    return [
        ("Alcohol", "alcohol", 5, 20, ref["alcohol"]),
        ("Volatile acidity", "vol_acid", 0.1, 2.0, ref["vol_acid"]),
        ("Free SO₂", "free_sulfur", 0, 500, ref["free_sulfur"]),
        ("Sulphates", "sulphates", 0, 5, ref["sulphates"]),
        ("Residual sugar", "res_sugar", 0, 100, ref["res_sugar"]),
    ]

def build_radar_figure(valores: dict, tipo: str) -> go.Figure:
    feats = _features(tipo)
    etiquetas = [cfg[0] for cfg in feats]
    mercado, cliente = [], []
    for _, key, low, high, ref in feats:
        rango = max(high - low, 1e-6)
        cliente.append((valores[key] - low) / rango)
        mercado.append((ref - low) / rango)

    etiquetas_c = etiquetas + [etiquetas[0]]
    cliente_c = cliente + [cliente[0]]
    mercado_c = mercado + [mercado[0]]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=cliente_c, theta=etiquetas_c, fill="toself", name="Tu vino",
        line=dict(color="#6B7558", width=3), fillcolor="rgba(107, 117, 88, 0.18)",
    ))
    fig.add_trace(go.Scatterpolar(
        r=mercado_c, theta=etiquetas_c, fill="toself", name="Media mercado",
        line=dict(color="#cc999d", width=2, dash="dash"), fillcolor="rgba(204, 153, 157, 0.18)",
    ))
    fig.update_layout(
        height=340,
        margin=dict(l=20, r=20, t=30, b=20),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        polar=dict(
            radialaxis=dict(visible=False),
            angularaxis=dict(
                tickfont=dict(color="#3F452F", family="Georgia, serif", size=12),
                linecolor="rgba(0,0,0,0.1)",
            ),
        ),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
    )
    return fig

def registrar_callback_radar(app):
    @app.callback(
        Output("graph-calidad", "figure"),
        [
            Input("dd-mercado", "value"),
            Input("sl-alcohol", "value"),
            Input("sl-vol-acid", "value"),
            Input("sl-free-sulfur", "value"),
            Input("sl-sulphates", "value"),
            Input("sl-res-sugar", "value"),
        ],
    )
    def actualizar_grafica(mercado, alcohol, vol_acid, free_sulfur, sulphates, res_sugar):
        valores = {
            "alcohol": alcohol,
            "vol_acid": vol_acid,
            "free_sulfur": free_sulfur,
            "sulphates": sulphates,
            "res_sugar": res_sugar,
        }
        return build_radar_figure(valores, mercado or "combinado")
