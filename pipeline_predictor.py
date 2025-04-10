import os
from datetime import datetime
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import requests
from supabase import create_client

# Conexión a Supabase
url = os.getenv("SUPABASE_URL")
key = os.getenv("SUPABASE_KEY")
supabase = create_client(url, key)

# CONFIGURACIÓN DE APIs DEPORTIVAS

# API OpenLigaDB (puedes llamar a varios endpoints, en este ejemplo usamos uno)
OPENLIGADB_ENDPOINTS = [
    "https://api.openligadb.de/getmatchdata/dfb/2024/5",
    "https://api.openligadb.de/getmatchdata/bl2/2024/28",
    "https://api.openligadb.de/getmatchdata/bl3/2024/31",
    "https://api.openligadb.de/getmatchdata/ucl24/2024/12",
    "https://api.openligadb.de/getmatchdata/bl1/2024/28",
    "https://api.openligadb.de/getmatchdata/ucl2024/2024/4",
    "https://api.openligadb.de/getmatchdata/uel24/2024/12"
]

# API Football-data.org (requiere header con token)
FOOTBALL_DATA_ENDPOINT = "https://api.football-data.org/v2/matches"
FOOTBALL_DATA_TOKEN = os.getenv("FOOTBALL_DATA_TOKEN")  # 707789134d794688bdafc8c0c013811b

def obtener_datos_openligadb():
    datos = []
    for url_endpoint in OPENLIGADB_ENDPOINTS:
        try:
            resp = requests.get(url_endpoint, timeout=10)
            if resp.status_code == 200:
                datos.extend(resp.json())
        except Exception as e:
            print("Error en OpenLigaDB:", e)
    return datos

def obtener_datos_football_data():
    headers = {"X-Auth-Token": FOOTBALL_DATA_TOKEN}
    try:
        resp = requests.get(FOOTBALL_DATA_ENDPOINT, headers=headers, timeout=10)
        if resp.status_code == 200:
            return resp.json().get("matches", [])
    except Exception as e:
        print("Error en Football Data API:", e)
    return []

def obtener_datos_actualizados():
    """
    Combina datos de varias APIs para obtener información real de partidos.
    Esta función debe mapear los datos recibidos a la estructura que necesita el sistema.
    """
    datos = []
    # Obtener datos de OpenLigaDB:
    datos_openliga = obtener_datos_openligadb()
    for partido in datos_openliga:
        # Mapea los datos de OpenLigaDB (ejemplo; ajusta según la respuesta real)
        datos.append({
            "nombre_partido": partido.get("MatchName", "Partido Desconocido"),
            "liga": partido.get("League", "Desconocida"),
            "deporte": "futbol",
            "goles_local_prom": partido.get("MatchResults", [{}])[0].get("PointsTeam1", 1.5),
            "goles_visita_prom": partido.get("MatchResults", [{}])[0].get("PointsTeam2", 1.2),
            "racha_local": 3,  # Como ejemplo (aquí se puede conectar otra API para rachas)
            "racha_visita": 2,
            "clima": 1,
            "importancia_partido": 3,
            "hora": partido.get("MatchDateTime", datetime.utcnow().isoformat())
        })

    # Obtener datos de Football-data.org:
    datos_ft = obtener_datos_football_data()
    for partido in datos_ft:
        datos.append({
            "nombre_partido": partido["homeTeam"]["name"] + " vs " + partido["awayTeam"]["name"],
            "liga": partido.get("competition", {}).get("name", "Desconocida"),
            "deporte": "futbol",
            "goles_local_prom": partido.get("score", {}).get("fullTime", {}).get("homeTeam", 1.5),
            "goles_visita_prom": partido.get("score", {}).get("fullTime", {}).get("awayTeam", 1.2),
            "racha_local": 3,
            "racha_visita": 2,
            "clima": 1,
            "importancia_partido": 3,
            "hora": partido.get("utcDate", datetime.utcnow().isoformat())
        })

    # Si no se obtuvieron datos, se puede simular uno
    if not datos:
        datos = [{
            "nombre_partido": "Arsenal vs Chelsea",
            "liga": "Premier League",
            "deporte": "futbol",
            "goles_local_prom": 1.8,
            "goles_visita_prom": 1.2,
            "racha_local": 4,
            "racha_visita": 2,
            "clima": 1,
            "importancia_partido": 3,
            "hora": datetime.utcnow().isoformat()
        }]
    return datos

# Función para obtener datos de entrenamiento históricos o simulados
def obtener_datos_entrenamiento():
    # Simulación de datos: 0 = Gana Local, 1 = Empate, 2 = Gana Visitante
    return pd.DataFrame([
        {"goles_local_prom": 1.5, "goles_visita_prom": 1.0, "racha_local": 3, "racha_visita": 2, "clima": 1, "importancia_partido": 2, "resultado": 0},
        {"goles_local_prom": 1.2, "goles_visita_prom": 1.2, "racha_local": 2, "racha_visita": 2, "clima": 1, "importancia_partido": 2, "resultado": 1},
        {"goles_local_prom": 0.9, "goles_visita_prom": 1.7, "racha_local": 1, "racha_visita": 4, "clima": 2, "importancia_partido": 3, "resultado": 2},
        {"goles_local_prom": 2.1, "goles_visita_prom": 0.8, "racha_local": 5, "racha_visita": 1, "clima": 1, "importancia_partido": 1, "resultado": 0},
        {"goles_local_prom": 1.0, "goles_visita_prom": 1.0, "racha_local": 2, "racha_visita": 2, "clima": 1, "importancia_partido": 2, "resultado": 1},
        {"goles_local_prom": 1.3, "goles_visita_prom": 1.6, "racha_local": 2, "racha_visita": 3, "clima": 2, "importancia_partido": 2, "resultado": 2}
    ])

# Entrenar el modelo XGBoost con datos históricos
def entrenar_modelo():
    datos = obtener_datos_entrenamiento()
    X = datos.drop("resultado", axis=1)
    y = datos["resultado"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', num_class=3)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    precision = accuracy_score(y_test, y_pred)
    print(f"Precisión del modelo: {precision:.2f}")

    return model

# Predecir el resultado para un partido usando el modelo entrenado
def predecir_resultado(modelo, datos_partido):
    df = pd.DataFrame([{
        "goles_local_prom": datos_partido["goles_local_prom"],
        "goles_visita_prom": datos_partido["goles_visita_prom"],
        "racha_local": datos_partido["racha_local"],
        "racha_visita": datos_partido["racha_visita"],
        "clima": datos_partido["clima"],
        "importancia_partido": datos_partido["importancia_partido"]
    }])
    pred = model.predict(df)[0]  # ATENCIÓN: cuidado con el nombre usado aquí.

    opciones = {
        0: ("Gana Local", 0.70),
        1: ("Empate", 0.65),
        2: ("Gana Visitante", 0.68)
    }
    return opciones.get(pred, ("Indefinido", 0.0))

def predecir_resultado(modelo, datos_partido):
    # Corregido para usar el modelo entrenado
    df = pd.DataFrame([{
        "goles_local_prom": datos_partido["goles_local_prom"],
        "goles_visita_prom": datos_partido["goles_visita_prom"],
        "racha_local": datos_partido["racha_local"],
        "racha_visita": datos_partido["racha_visita"],
        "clima": datos_partido["clima"],
        "importancia_partido": datos_partido["importancia_partido"]
    }])
    pred = modelo.predict(df)[0]
    opciones = {
        0: ("Gana Local", 0.70),
        1: ("Empate", 0.65),
        2: ("Gana Visitante", 0.68)
    }
    return opciones.get(pred, ("Indefinido", 0.0))

# Actualiza los datos de partidos obtenidos de las APIs
def actualizar_datos_partidos():
    nuevos_datos = obtener_datos_actualizados()
    for partido in nuevos_datos:
        supabase.table("partidos").upsert(partido, on_conflict=["nombre_partido", "hora"]).execute()
    return {"status": "Datos actualizados correctamente"}

# Procesa los partidos: entrena el modelo y genera predicciones para cada partido
def procesar_partidos():
    modelo = entrenar_modelo()
    partidos = obtener_datos_actualizados()  # En producción, se extraen de Supabase o de las APIs directamente

    for partido in partidos:
        resultado, confianza = predecir_resultado(modelo, partido)
        prediccion = {
            "deporte": partido["deporte"],
            "liga": partido["liga"],
            "partido": partido["nombre_partido"],
            "hora": partido["hora"],
            "pronostico_1": resultado,
            "confianza_1": confianza,
            "pronostico_2": "Menos de 2.5 goles",  # Este es un ejemplo; personaliza según tu lógica de negocio
            "confianza_2": 0.70,
            "pronostico_3": "Ambos no anotan",
            "confianza_3": 0.65
        }
        supabase.table("predicciones").upsert(prediccion, on_conflict=["partido", "hora"]).execute()
        print("Predicción guardada para:", prediccion["partido"])

    return {"message": "Predicciones generadas correctamente"}
