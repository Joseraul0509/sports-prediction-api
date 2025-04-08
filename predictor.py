import xgboost as xgb
import pandas as pd
from datetime import datetime
import os
from dotenv import load_dotenv

# Cargar las variables de entorno
load_dotenv()

# Cargar el modelo
model_path = "modelo_entrenado.json"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Modelo no encontrado en {model_path}")

modelo = xgb.XGBClassifier()
modelo.load_model(model_path)

# Datos simulados para probar (luego esto vendrá de tu base de datos)
datos_partido = pd.DataFrame([{
    "equipo_local": "Real Madrid",
    "equipo_visitante": "Barcelona",
    "goles_local": 1,
    "goles_visitante": 2,
    "posesion_local": 55,
    "posesion_visitante": 45,
    # añade aquí todas las variables que espera tu modelo
}])

# Realiza predicción (ajusta si tu modelo da probabilidades)
pred = modelo.predict(datos_partido)[0]
confianza = modelo.predict_proba(datos_partido).max()

# Preparar datos
payload = {
    "deporte": "futbol",
    "liga": "La Liga",
    "partido": "Real Madrid vs Barcelona",
    "hora": datetime.now().strftime("%Y-%m-%d %H:%M"),
    "pronostico_1": f"Resultado predicho: {pred}",
    "confianza_1": float(round(confianza, 2)),
    "pronostico_2": "Ambos anotan",
    "confianza_2": 0.75,
    "pronostico_3": "Más de 2.5 goles",
    "confianza_3": 0.80
}

# Enviar a la API
import requests
res = requests.post("http://localhost:8000/guardar_prediccion", json=payload)

print("Estado:", res.status_code)
print("Respuesta:", res.json())
