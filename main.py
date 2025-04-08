import pandas as pd
import xgboost as xgb
from datetime import datetime
import requests
import os
from supabase import create_client
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()
url = os.getenv("SUPABASE_URL")
key = os.getenv("SUPABASE_KEY")
supabase = create_client(url, key)

# Cargar modelo
model_path = "modelo_entrenado.json"
modelo = xgb.XGBClassifier()
modelo.load_model(model_path)

# Paso 1: Obtener datos reales desde Supabase
def obtener_datos_supabase():
    response = supabase.table("estadisticas").select("*").execute()
    return response.data if response.data else []

# Paso 2: Preparar DataFrame para predicción
def preparar_datos(datos_raw):
    df = pd.DataFrame(datos_raw)

    columnas_necesarias = [
        "equipo_local", "equipo_visitante", "goles_local", "goles_visitante",
        "posesion_local", "posesion_visitante"
        # Agrega aquí todas las variables que espera tu modelo
    ]

    df = df[columnas_necesarias]
    return df

# Paso 3: Ejecutar predicción para cada fila
def procesar_partidos():
    partidos = obtener_datos_supabase()
    if not partidos:
        print("No se encontraron datos en Supabase.")
        return

    df = preparar_datos(partidos)

    for i, fila in df.iterrows():
        pred = modelo.predict(pd.DataFrame([fila]))[0]
        confianza = modelo.predict_proba(pd.DataFrame([fila])).max()

        partido = partidos[i]
        payload = {
            "deporte": partido.get("deporte", "futbol"),
            "liga": partido.get("liga", "Desconocida"),
            "partido": f"{fila['equipo_local']} vs {fila['equipo_visitante']}",
            "hora": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "pronostico_1": f"Resultado predicho: {pred}",
            "confianza_1": float(round(confianza, 2)),
            "pronostico_2": "Ambos anotan",  # Personaliza este pronóstico
            "confianza_2": 0.75,
            "pronostico_3": "Más de 2.5 goles",
            "confianza_3": 0.80
        }

        response = requests.post("http://localhost:8000/guardar_prediccion", json=payload)
        print("Predicción enviada:", payload["partido"], "| Estado:", response.status_code)

# Ejecutar el proceso
if __name__ == "__main__":
    procesar_partidos()
