import pandas as pd
import numpy as np
from pathlib import Path
import plotly.graph_objects as go
from dash import html, dcc
from dash.dependencies import Input, Output

# =========================================================
#   CARGA Y LIMPIEZA DEL DATASET UNA SOLA VEZ
# =========================================================
DATA_SP = Path("src/data/wines_SPA.csv")
df_spain_raw = pd.read_csv(DATA_SP)
df_spain = df_spain_raw.copy()
df_spain = df_spain.drop_duplicates(subset=["winery", "wine"], keep="first")


# Eliminar columnas no usadas si existen
for col in ["acidity", "body"]:
    if col in df_spain.columns:
        df_spain = df_spain.drop(columns=[col])

# =========================================================
#   NORMALIZAR RATING A ESCALA 3–9
# =========================================================
if "rating" in df_spain.columns:
    scale = (9 - 3) / (4.9 - 4.2)
    df_spain["rating_scaled"] = 3 + (df_spain["rating"] - 4.2) * scale
    df_spain["rating_scaled"] += np.random.uniform(-0.5, 0.5, size=len(df_spain))
    df_spain["rating_scaled"] = df_spain["rating_scaled"].clip(3, 9)
else:
    df_spain["rating_scaled"] = pd.Series(dtype=float)

# =========================================================
#   NORMALIZAR COLOR
# =========================================================
def normalizar_color(x):
    """Devuelve red / white / other aunque haya valores raros."""
    if pd.isna(x):
        return "other"
    x = str(x).lower()
    if "red" in x:
        return "red"
    if "white" in x or "blanco" in x:
        return "white"
    return "other"

if "color" in df_spain.columns:
    df_spain["color"] = df_spain["color"].apply(normalizar_color)
else:
    df_spain["color"] = df_spain["type"].astype(str).apply(normalizar_color)

# =========================================================
#   RANGOS DE PRECIO
# =========================================================
PRICE_MIN = float(df_spain["price"].min())
PRICE_MAX = float(df_spain["price"].max())


# =========================================================
#   REGISTRAR LOS 2 CALLBACKS
# =========================================================
def registrar_callbacks_mercado(app):

    # ---------------------------------------------------------
    #   CALLBACK 1 → SCATTER PRECIO vs CALIDAD
    # ---------------------------------------------------------
    @app.callback(
        Output("graph-mercado-scatter", "figure"),
        [
            Input("dd-market-color", "value"),
            Input("rs-precio", "value"),
            Input("input-my-rating", "value"),
            Input("input-my-price", "value"),
        ],
    )
    def actualizar_scatter(color, rango_precio, my_rating, my_price):
        df = df_spain.copy()
        y_min, y_max = PRICE_MIN, PRICE_MAX

        # Filtrar color
        if color != "todos":
            df = df[df["color"] == color]

        # Filtrar precio
        if rango_precio:
            pmin, pmax = rango_precio
            df = df[(df["price"] >= pmin) & (df["price"] <= pmax)]
            y_min, y_max = pmin, pmax

        fig = go.Figure()

        # Mercado
        if not df.empty:
            fig.add_trace(
                go.Scatter(
                    x=df["rating_scaled"],
                    y=df["price"],
                    mode="markers",
                    name="Mercado",
                    marker=dict(color="rgba(107,117,88,0.35)", size=8),
                    # 👇 AQUI LE PASAMOS winery Y wine
                    customdata=np.stack((df["winery"], df["wine"]), axis=-1),
                    hovertemplate="Vino: %{customdata[1]}<br>Calidad: %{x:.1f}<br>Precio: €%{y}<extra></extra>",
                )
            )

        # Tu vino
        if my_rating is not None and my_price is not None:
            fig.add_trace(
                go.Scatter(
                    x=[my_rating],
                    y=[my_price],
                    mode="markers",
                    name="Tu vino",
                    marker=dict(color="#cc6666", size=14, line=dict(color="#3F452F", width=1.5)),
                    hovertemplate="Tu vino<br>Calidad: %{x:.1f}<br>Precio: €%{y}<extra></extra>",
                )
            )

        fig.update_layout(
            height=240,
            margin=dict(l=20, r=20, t=10, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(title="Calidad (3–9)", gridcolor="rgba(0,0,0,0.08)"),
            yaxis=dict(title="Precio (€)", gridcolor="rgba(0,0,0,0.08)", range=[y_min, y_max]),
            showlegend=False,
        )

        return fig

    # ---------------------------------------------------------
    #   CALLBACK 2 → TABLA DE VECINOS (FILTRAR SOLO POR COLOR)
    # ---------------------------------------------------------
    @app.callback(
        Output("table-vecinos", "children"),
        [
            Input("dd-market-color", "value"),
            Input("input-my-rating", "value"),
            Input("input-my-price", "value"),
        ],
    )
    def actualizar_vecinos(color, my_rating, my_price):

        df = df_spain.copy()

        # Filtrar por color según dropdown principal
        if color != "todos":
            df = df[df["color"] == color]

        # No filtrar por precio

        # Validación
        if df.empty or my_rating is None or my_price is None:
            return html.Div("Introduce calidad y precio", className="table-empty")

        # Distancia euclídea
        df["dist"] = np.sqrt(
            (df["rating_scaled"] - my_rating) ** 2 +
            (df["price"] - my_price) ** 2
        )

        # Top 4 más cercanos
        top = df.nsmallest(4, "dist")

        filas = [
            html.Tr([
                html.Th("Bodega"),
                html.Th("Vino"),
                html.Th("Calidad"),
                html.Th("Precio (€)"),
            ])
        ]

        # Tu vino arriba
        filas.append(
            html.Tr([
                html.Td("Tu bodega"),
                html.Td("Tu vino"),
                html.Td(f"{my_rating:.2f}"),
                html.Td(f"€{my_price:.2f}"),
            ], className="table-myrow")
        )

        # Vecinos
        for _, row in top.iterrows():
            filas.append(
                html.Tr([
                    html.Td(row.get("winery", "")),
                    html.Td(row.get("wine", "")),
                    html.Td(f"{row['rating_scaled']:.2f}"),
                    html.Td(f"€{row['price']:.2f}"),
                ])
            )

        return html.Table(filas, className="market-table-inner")
