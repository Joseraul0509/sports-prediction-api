import xgboost as xgb
import pandas as pd
import requests
from datetime import datetime

# Carga el modelo (ajusta el path si es necesario)
modelo = xgb.XGBClassifier()
modelo.load_model("modelo_entrenado.json")

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
res = requests.post("http://localhost:8000/guardar_prediccion", json=payload)

print("Estado:", res.status_code)
print("Respuesta:", res.json())
