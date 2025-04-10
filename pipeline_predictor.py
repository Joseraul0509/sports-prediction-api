import os
from datetime import datetime
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import requests
from supabase import create_client

# Conexión a Supabase (usando variables de entorno definidas en .env)
url = os.getenv("SUPABASE_URL")
key = os.getenv("SUPABASE_KEY")
supabase = create_client(url, key)

# CONFIGURACIÓN DE APIs DEPORTIVAS

# Fútbol: OpenLigaDB y Football-data.org
OPENLIGADB_ENDPOINTS = [
    "https://api.openligadb.de/getmatchdata/dfb/2024/5",
    "https://api.openligadb.de/getmatchdata/bl2/2024/28",
    "https://api.openligadb.de/getmatchdata/bl3/2024/31",
    "https://api.openligadb.de/getmatchdata/ucl24/2024/12",
    "https://api.openligadb.de/getmatchdata/bl1/2024/28",
    "https://api.openligadb.de/getmatchdata/ucl2024/2024/4",
    "https://api.openligadb.de/getmatchdata/uel24/2024/12"
]
FOOTBALL_DATA_ENDPOINT = "https://api.football-data.org/v2/matches"
# Asegúrate de tener la variable FOOTBALL_DATA_TOKEN en tu .env

# APIs para NBA, MLB y NHL usando balldontlie (para NBA es real, para los otros se simulan)
BALLDONTLIE_BASE_URL = "https://www.balldontlie.io/api/v1"

# Función para obtener datos de fútbol mediante OpenLigaDB y Football-data.org
def obtener_datos_futbol():
    datos = []
    # Obtener datos de OpenLigaDB:
    for endpoint in OPENLIGADB_ENDPOINTS:
        try:
            resp = requests.get(endpoint, timeout=10)
            if resp.status_code == 200:
                for partido in resp.json():
                    datos.append({
                        "nombre_partido": partido.get("MatchName", "Partido Desconocido"),
                        "liga": partido.get("League", "Desconocida"),
                        "deporte": "futbol",
                        "goles_local_prom": partido.get("MatchResults", [{}])[0].get("PointsTeam1", 1.5),
                        "goles_visita_prom": partido.get("MatchResults", [{}])[0].get("PointsTeam2", 1.2),
                        "racha_local": 3,
                        "racha_visita": 2,
                        "clima": 1,
                        "importancia_partido": 3,
                        "hora": partido.get("MatchDateTime", datetime.utcnow().isoformat())
                    })
        except Exception as e:
            print("Error en OpenLigaDB:", e)
    # Obtener datos de Football-data.org:
    headers = {"X-Auth-Token": os.getenv("FOOTBALL_DATA_TOKEN")}
    try:
        resp = requests.get(FOOTBALL_DATA_ENDPOINT, headers=headers, timeout=10)
        if resp.status_code == 200:
            for partido in resp.json().get("matches", []):
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
    except Exception as e:
        print("Error en Football Data API:", e)
    return datos

# Función para obtener datos de NBA, MLB y NHL usando balldontlie
def obtener_datos_balldontlie(deporte):
    datos = []
    if deporte.upper() == "NBA":
        try:
            url = f"{BALLDONTLIE_BASE_URL}/games"
            params = {"per_page": 20}  # Obtenemos algunos partidos recientes
            resp = requests.get(url, params=params, timeout=10)
            if resp.status_code == 200:
                for game in resp.json().get("data", []):
                    datos.append({
                        "nombre_partido": game["home_team"]["full_name"] + " vs " + game["visitor_team"]["full_name"],
                        "liga": "NBA",
                        "deporte": "NBA",
                        "goles_local_prom": float(game.get("home_team_score", 100))/10,  # Simulación
                        "goles_visita_prom": float(game.get("visitor_team_score", 100))/10,
                        "racha_local": 3,
                        "racha_visita": 2,
                        "clima": None,
                        "importancia_partido": 3,
                        "hora": game.get("date", datetime.utcnow().isoformat())
                    })
        except Exception as e:
            print("Error en balldontlie (NBA):", e)
    else:
        # Para MLB y NHL, simulamos datos usando la misma clave (ya que no hay un endpoint real en balldontlie para estos deportes)
        datos.append({
            "nombre_partido": f"Simulado {deporte} - Equipo A vs Equipo B",
            "liga": deporte,
            "deporte": deporte,
            "goles_local_prom": 2.0 if deporte.upper() == "MLB" else 3.0,
            "goles_visita_prom": 1.5 if deporte.upper() == "MLB" else 3.0,
            "racha_local": 3,
            "racha_visita": 2,
            "clima": None,
            "importancia_partido": 3,
            "hora": datetime.utcnow().isoformat()
        })
    return datos

# Función para obtener datos adicionales con TheSportsDB (ejemplo básico)
def obtener_datos_thesportsdb():
    datos = []
    # Ejemplo: listar todas las ligas
    url = "https://www.thesportsdb.com/api/v1/json/3/all_leagues.php"
    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            datos = resp.json().get("leagues", [])
    except Exception as e:
        print("Error en TheSportsDB:", e)
    return datos

# Función que combina datos de todas las fuentes
def obtener_datos_actualizados():
    datos = []
    # Datos para fútbol
    datos.extend(obtener_datos_futbol())
    # Datos para NBA, MLB y NHL
    for deporte in ["NBA", "MLB", "NHL"]:
        datos.extend(obtener_datos_balldontlie(deporte))
    # Se pueden integrar datos de TheSportsDB si se desean detalles adicionales (opcional)
    return datos if datos else [{
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

# Función para obtener datos de entrenamiento históricos o simulados
def obtener_datos_entrenamiento():
    # Simulación con tres clases: 0 = Gana Local, 1 = Empate, 2 = Gana Visitante
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
    pred = modelo.predict(df)[0]
    opciones = {
        0: ("Gana Local", 0.70),
        1: ("Empate", 0.65),
        2: ("Gana Visitante", 0.68)
    }
    return opciones.get(pred, ("Indefinido", 0.0))

# Actualiza los datos de partidos obtenidos de las APIs y guarda en Supabase.
def actualizar_datos_partidos():
    nuevos_datos = obtener_datos_actualizados()
    for partido in nuevos_datos:
        # Upsert en la tabla "partidos" usando on_conflict sobre ["nombre_partido", "hora"]
        supabase.table("partidos").upsert(partido, on_conflict=["nombre_partido", "hora"]).execute()
    return {"status": "Datos actualizados correctamente"}

# Procesa los partidos: entrena el modelo y genera predicciones para cada partido,
# insertándolas en la tabla "predicciones" con upsert usando on_conflict sobre ["partido", "hora"]
def procesar_predicciones():
    model = entrenar_modelo()
    partidos = obtener_datos_actualizados()  # En producción, se pueden extraer de Supabase o las APIs directamente

    for partido in partidos:
        resultado, confianza = predecir_resultado(model, partido)
        prediccion = {
            "deporte": partido["deporte"],
            "liga": partido["liga"],
            "partido": partido["nombre_partido"],
            "hora": partido["hora"],
            "pronostico_1": resultado,
            "confianza_1": confianza,
            "pronostico_2": "Menos de 2.5 goles",  # Ejemplo: personalizar según tus reglas
            "confianza_2": 0.70,
            "pronostico_3": "Ambos no anotan",
            "confianza_3": 0.65
        }
        # Upsert en la tabla "predicciones" usando on_conflict ["partido", "hora"]
        supabase.table("predicciones").upsert(prediccion, on_conflict=["partido", "hora"]).execute()
        print("Predicción guardada para:", prediccion["partido"])

    return {"message": "Predicciones generadas correctamente"}
