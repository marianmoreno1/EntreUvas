import numpy as np
import pandas as pd
import joblib
import json
import os

# Cargar modelo y features
BASE = os.path.dirname(__file__)  # carpeta actual del archivo modelo.py

model = joblib.load(os.path.join(BASE, "modelo_rf_entreuvas.pkl"))



with open(os.path.join(BASE, "features_rf_entreuvas.json")) as f:
    FEATURES = json.load(f)

def predecir_entreuvas(alcohol, vol_acid, free_sulfur, sulphates, res_sugar):
    """Devuelve predicción y confianza del modelo."""
    
    valores = [[
        alcohol,
        vol_acid,
        free_sulfur,
        sulphates,
        res_sugar
    ]]
    
    df = pd.DataFrame(valores, columns=FEATURES)

    # Predicción principal
    pred = model.predict(df)[0]

    # Confianza usando desviación estándar de árboles
    preds_arboles = np.array([
        tree.predict(df.values)[0] for tree in model.estimators_
    ])
    std_pred = preds_arboles.std()
    confianza = max(0, 1 - std_pred / (abs(pred) + 1e-6)) * 100

    return float(pred), float(confianza)
