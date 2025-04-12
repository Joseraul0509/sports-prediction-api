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

# ===============================
# NUEVAS FUNCIONES DE INSERCIÓN
# ===============================

# Función para insertar en la tabla leagues (usar el nombre "leagues" para ser consistente)
def insertar_league(nombre_liga: str, deporte: str):
    try:
        existe = supabase.table("leagues").select("*").match({
            "nombre": nombre_liga,
            "deporte": deporte
        }).execute()
        if not existe.data:
            supabase.table("leagues").insert({
                "nombre": nombre_liga,
                "deporte": deporte
            }).execute()
            print(f"Liga insertada: {nombre_liga}")
    except Exception as e:
        print(f"Error al insertar liga {nombre_liga}: {str(e)}")

# Función para insertar en la tabla partidos (se insertan campos mínimos: nombre_partido, liga y hora)
def insertar_partido(nombre_partido: str, liga: str, hora: str):
    try:
        existe = supabase.table("partidos").select("*").match({
            "nombre_partido": nombre_partido,
            "liga": liga,
            "hora": hora
        }).execute()
        if not existe.data:
            supabase.table("partidos").insert({
                "nombre_partido": nombre_partido,
                "liga": liga,
                "hora": hora
            }).execute()
            print(f"Partido insertado: {nombre_partido}")
    except Exception as e:
        print(f"Error al insertar partido {nombre_partido}: {str(e)}")

# ===============================
# FIN DE NUEVAS FUNCIONES
# ===============================

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
                        "clima": partido.get("clima", 1),  # Si existe un campo, sino se asigna 1
                        "importancia_partido": partido.get("importancia_partido", 3),
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

def obtener_datos_balldontlie(deporte):
    datos = []
    if deporte.upper() == "NBA":
        try:
            url = f"{BALLDONTLIE_BASE_URL}/games"
            params = {"per_page": 20}
            resp = requests.get(url, params=params, timeout=10)
            if resp.status_code == 200:
                for game in resp.json().get("data", []):
                    datos.append({
                        "nombre_partido": game["home_team"]["full_name"] + " vs " + game["visitor_team"]["full_name"],
                        "liga": "NBA",
                        "deporte": "NBA",
                        "goles_local_prom": float(game.get("home_team_score", 100)) / 10,
                        "goles_visita_prom": float(game.get("visitor_team_score", 100)) / 10,
                        "racha_local": 3,
                        "racha_visita": 2,
                        "clima": 1,  # Asignamos un valor por defecto
                        "importancia_partido": 3,
                        "hora": game.get("date", datetime.utcnow().isoformat())
                    })
        except Exception as e:
            print("Error en balldontlie (NBA):", e)
    else:
        datos.append({
            "nombre_partido": f"Simulado {deporte} - Equipo A vs Equipo B",
            "liga": deporte,
            "deporte": deporte,
            "goles_local_prom": 2.0 if deporte.upper() == "MLB" else 3.0,
            "goles_visita_prom": 1.5 if deporte.upper() == "MLB" else 3.0,
            "racha_local": 3,
            "racha_visita": 2,
            "clima": 1,
            "importancia_partido": 3,
            "hora": datetime.utcnow().isoformat()
        })
    return datos

def obtener_datos_actualizados():
    datos = []
    datos.extend(obtener_datos_futbol())
    for deporte in ["NBA", "MLB", "NHL"]:
        datos.extend(obtener_datos_balldontlie(deporte))
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

def obtener_datos_entrenamiento():
    return pd.DataFrame([
        {"goles_local_prom": 1.8, "goles_visita_prom": 1.2, "racha_local": 3, "racha_visita": 2, "clima": 1, "importancia_partido": 3, "resultado": 1},
        {"goles_local_prom": 1.0, "goles_visita_prom": 1.5, "racha_local": 1, "racha_visita": 3, "clima": 2, "importancia_partido": 2, "resultado": 0},
        {"goles_local_prom": 2.2, "goles_visita_prom": 0.9, "racha_local": 4, "racha_visita": 1, "clima": 1, "importancia_partido": 3, "resultado": 1},
        {"goles_local_prom": 1.3, "goles_visita_prom": 1.4, "racha_local": 2, "racha_visita": 3, "clima": 3, "importancia_partido": 2, "resultado": 0}
    ])

def entrenar_modelo():
    df = obtener_datos_entrenamiento()
    # Convertir explícitamente las columnas a tipos numéricos
    df = df.astype({
        "goles_local_prom": float,
        "goles_visita_prom": float,
        "racha_local": int,
        "racha_visita": int,
        "clima": int,
        "importancia_partido": int,
        "resultado": int
    })
    X = df.drop("resultado", axis=1)
    y = df["resultado"]
    # Si hay menos de 2 clases, entrenar con todo el dataset para evitar error de clases
    if len(y.unique()) < 2:
        print("Datos insuficientes para diferenciar clases; usando todo el dataset para entrenamiento.")
        model = xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss")
        model.fit(X, y)
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
        model = xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss")
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        print(f"Precisión del modelo: {acc:.2f}")
    return model

def predecir_resultado(modelo, datos_partido):
    df = pd.DataFrame([{
        "goles_local_prom": datos_partido["goles_local_prom"],
        "goles_visita_prom": datos_partido["goles_visita_prom"],
        "racha_local": datos_partido["racha_local"],
        "racha_visita": datos_partido["racha_visita"],
        "clima": int(datos_partido["clima"]),  # Aseguramos que sea entero
        "importancia_partido": datos_partido["importancia_partido"]
    }])
    pred = modelo.predict(df)[0]
    opciones = {
        0: ("Gana Local", 0.70),
        1: ("Empate", 0.65),
        2: ("Gana Visitante", 0.68)
    }
    return opciones.get(pred, ("Indefinido", 0.0))

def manual_upsert(table_name: str, data: dict, conflict_columns: list):
    query = supabase.table(table_name).select("*")
    for col in conflict_columns:
        query = query.eq(col, data[col])
    result = query.execute()
    if result.data and len(result.data) > 0:
        update_query = supabase.table(table_name).update(data)
        for col in conflict_columns:
            update_query = update_query.eq(col, data[col])
        update_query.execute()
        return "updated"
    else:
        supabase.table(table_name).insert(data).execute()
        return "inserted"

def actualizar_datos_partidos():
    nuevos_datos = obtener_datos_actualizados()
    for partido in nuevos_datos:
        hora_valor = partido["hora"]
        if isinstance(hora_valor, str):
            try:
                hora_valor = datetime.fromisoformat(hora_valor)
            except Exception as ex:
                print(f"Error al convertir hora: {ex}")
                hora_valor = datetime.utcnow()
        partido["hora"] = hora_valor.strftime("%Y-%m-%d %H:%M:%S")
        
        # --- NUEVAS LLAMADAS PARA INSERTAR EN ORDEN ---
        insertar_league(partido["liga"], partido["deporte"])
        insertar_partido(partido["nombre_partido"], partido["liga"], partido["hora"])
        # -------------------------------------------------

        result = manual_upsert("partidos", partido, ["nombre_partido", "hora"])
        print(f"Registro en partidos {partido['nombre_partido']}: {result}")
    return {"status": "Datos actualizados correctamente"}

def procesar_predicciones():
    model = entrenar_modelo()
    partidos = obtener_datos_actualizados()
    for partido in partidos:
        hora_valor = partido["hora"]
        if isinstance(hora_valor, str):
            try:
                hora_dt = datetime.fromisoformat(hora_valor)
            except Exception as ex:
                print(f"Error al convertir hora en predicción: {ex}")
                hora_dt = datetime.utcnow()
        else:
            hora_dt = hora_valor
        partido["hora"] = hora_dt.strftime("%Y-%m-%d %H:%M:%S")
        resultado, confianza = predecir_resultado(model, partido)
        prediccion = {
            "deporte": partido["deporte"],
            "liga": partido["liga"],
            "partido": partido["nombre_partido"],
            "hora": partido["hora"],
            "pronostico_1": resultado,
            "confianza_1": confianza,
            "pronostico_2": "Menos de 2.5 goles",
            "confianza_2": 0.70,
            "pronostico_3": "Ambos no anotan",
            "confianza_3": 0.65
        }
        result = manual_upsert("predicciones", prediccion, ["partido", "hora"])
        print(f"Registro en predicciones {prediccion['partido']}: {result}")
    return {"message": "Predicciones generadas correctamente"}
