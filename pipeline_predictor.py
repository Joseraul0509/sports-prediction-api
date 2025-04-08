import pandas as pd
import xgboost as xgb
from datetime import datetime
import requests
import os
from supabase import create_client
from dotenv import load_dotenv

# Cargar variables de entorno y conectar a Supabase
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
if not SUPABASE_URL or not SUPABASE_KEY:
    raise Exception("Falta configurar SUPABASE_URL o SUPABASE_KEY en el archivo .env")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# Cargar modelo
MODEL_PATH = "modelo_entrenado.json"
modelo = xgb.XGBClassifier()
modelo.load_model(MODEL_PATH)

# Paso 1: Obtener datos reales desde la tabla "estadisticas"
def obtener_datos_supabase():
    response = supabase.table("estadisticas").select("*").execute()
    return response.data if response.data else []

# Paso 2: Preparar el DataFrame con las columnas necesarias
def preparar_datos(datos_raw):
    df = pd.DataFrame(datos_raw)
    columnas_necesarias = [
        "equipo_local", "equipo_visitante", "goles_local", "goles_visitante",
        "posesion_local", "posesion_visitante"
        # Agrega aquí todas las variables que tu modelo requiera
    ]
    df = df[columnas_necesarias]
    return df

# Paso 3: Procesar cada partido y enviar la predicción a la API
def procesar_partidos():
    partidos = obtener_datos_supabase()
    if not partidos:
        print("No se encontraron datos en Supabase.")
        return

    df = preparar_datos(partidos)

    for i, fila in df.iterrows():
        pred = modelo.predict(pd.DataFrame([fila]))[0]
        confianza = modelo.predict_proba(pd.DataFrame([fila])).max()

        partido_info = partidos[i]
        payload = {
            "deporte": partido_info.get("deporte", "futbol"),
            "liga": partido_info.get("liga", "Desconocida"),
            "partido": f"{fila['equipo_local']} vs {fila['equipo_visitante']}",
            "hora": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "pronostico_1": f"Resultado predicho: {pred}",
            "confianza_1": float(round(confianza, 2)),
            "pronostico_2": "Ambos anotan",  # Personaliza según tus necesidades
            "confianza_2": 0.75,
            "pronostico_3": "Más de 2.5 goles",
            "confianza_3": 0.80
        }

        # Envía la predicción a la API; asegúrate de usar el puerto correcto
        response = requests.post("http://localhost:10000/guardar_prediccion", json=payload)
        print("Predicción enviada:", payload["partido"], "| Estado:", response.status_code)

# Permitir ejecutar este archivo en modo standalone si es necesario
if __name__ == "__main__":
    procesar_partidos()
