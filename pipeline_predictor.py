import os
import uuid
import time
from datetime import datetime, timezone
import random

import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from supabase import create_client, Client
from dotenv import load_dotenv
import requests

# ------------------------------
# Carga de configuración
# ------------------------------
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
API_SPORTS_KEY = os.getenv("API_SPORTS_KEY")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

BALDONTLIE_BASE = "https://api.balldontlie.io/v1"

# ------------------------------
# Helpers
# ------------------------------
def get_today() -> str:
    """Fecha actual en UTC AAAA-MM-DD."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")

def safe_numeric(value) -> float:
    """Intenta convertir a float, si falla retorna 0.0."""
    try:
        return float(value)
    except Exception:
        print(f"[safe_numeric] recibido {type(value)}, asignando 0.0")
        return 0.0

# ------------------------------
# UPSERT de Leagues y Partidos
# ------------------------------
def upsert_league(name: str) -> str:
    """Inserta o actualiza liga y devuelve su UUID."""
    try:
        supabase.table("leagues").upsert(
            {"id": str(uuid.uuid4()), "name": name, "country": None, "flag": None},
            on_conflict=["name"]
        ).execute()
        resp = supabase.table("leagues").select("id").eq("name", name).single().execute()
        league_id = resp.data["id"]
        print(f"[upsert_league] '{name}' -> {league_id}")
        return league_id
    except Exception as e:
        print(f"[upsert_league] ERROR '{name}': {e}")
        raise

def upsert_partido(record: dict):
    """Inserta o actualiza partido con 3 reintentos."""
    for attempt in range(1, 4):
        try:
            supabase.table("partidos").upsert(
                record,
                on_conflict=["league_id", "nombre_partido", "hora"]
            ).execute()
            print(f"[upsert_partido] OK {record['nombre_partido']} @ {record['hora']}")
            return
        except Exception as e:
            print(f"[upsert_partido] intento {attempt} falló: {e}")
            time.sleep(1)
    print(f"[upsert_partido] Falló definitivo en partido {record['nombre_partido']}")

# ------------------------------
# Extracción de Datos
# ------------------------------
def obtener_datos_futbol() -> list:
    datos = []
    fecha = get_today()
    print(f"[Fútbol] Obteniendo fixtures para {fecha}...")
    try:
        url = f"https://v3.football.api-sports.io/fixtures?date={fecha}"
        r = requests.get(url, headers={"x-apisports-key": API_SPORTS_KEY}, timeout=10)
        fixtures = r.json().get("response", [])
        print(f"[Fútbol] Obtenidos {len(fixtures)} fixtures")
        for f in fixtures:
            home = f["teams"]["home"]["name"]
            away = f["teams"]["away"]["name"]
            datos.append({
                "nombre_partido": f"{home} vs {away}",
                "liga": f["league"]["name"] or "Desconocida",
                "deporte": "futbol",
                "goles_local_prom": safe_numeric(f["goals"]["home"]),
                "goles_visita_prom": safe_numeric(f["goals"]["away"]),
                "racha_local": 0,
                "racha_visita": 0,
                "clima": 1,
                "importancia_partido": 1,
                "hora": f["fixture"]["date"]
            })
    except Exception as e:
        print(f"[Fútbol] ERROR al obtener fixtures: {e}")
    return datos

def obtener_datos_nba() -> list:
    datos = []
    fecha = get_today()
    print(f"[NBA] Obteniendo fixtures para {fecha} (v2)...")
    try:
        url = f"https://v2.nba.api-sports.io/games?date={fecha}"
        r = requests.get(url, headers={"x-apisports-key": API_SPORTS_KEY}, timeout=10)
        games = r.json().get("response", [])
        print(f"[NBA v2] Obtenidos {len(games)} juegos")
        for g in games:
            home = g["teams"]["home"]["name"]
            away = g["teams"]["away"]["name"]
            datos.append({
                "nombre_partido": f"{home} vs {away}",
                "liga": g["league"]["name"] or "NBA",
                "deporte": "nba",
                "goles_local_prom": safe_numeric(g["scores"]["home"]),
                "goles_visita_prom": safe_numeric(g["scores"]["away"]),
                "racha_local": 0,
                "racha_visita": 0,
                "clima": 1,
                "importancia_partido": 1,
                "hora": g["date"]
            })
    except Exception as e:
        print(f"[NBA v2] ERROR: {e}")
    # Aquí podrías añadir fallback a v1 y balldontlie si lo necesitas
    return datos

def obtener_datos_deporte_api(deporte: str) -> list:
    """MLB y NHL via API-Sports v1."""
    mapping = {
        "MLB": "https://v1.baseball.api-sports.io/games",
        "NHL": "https://v1.hockey.api-sports.io/games"
    }
    datos = []
    fecha = get_today()
    url = mapping.get(deporte.upper())
    if not url:
        return datos
    print(f"[{deporte}] Obteniendo fixtures para {fecha}...")
    try:
        r = requests.get(url, headers={"x-apisports-key": API_SPORTS_KEY}, params={"date": fecha}, timeout=10)
        games = r.json().get("response", [])
        print(f"[{deporte}] Obtenidos {len(games)} juegos")
        for g in games:
            home = g["teams"]["home"]["name"]
            away = g["teams"]["away"]["name"]
            datos.append({
                "nombre_partido": f"{home} vs {away}",
                "liga": g["league"]["name"] or deporte.upper(),
                "deporte": deporte.lower(),
                "goles_local_prom": safe_numeric(g["scores"]["home"]),
                "goles_visita_prom": safe_numeric(g["scores"]["away"]),
                "racha_local": 0,
                "racha_visita": 0,
                "clima": 1,
                "importancia_partido": 1,
                "hora": g["date"]
            })
    except Exception as e:
        print(f"[{deporte}] ERROR: {e}")
    return datos

def obtener_datos_actualizados() -> list:
    all_data = []
    all_data += obtener_datos_futbol()
    all_data += obtener_datos_nba()
    for dep in ["MLB", "NHL"]:
        all_data += obtener_datos_deporte_api(dep)
    print(f"[Pipeline] Total fixtures recopilados: {len(all_data)}")
    return all_data

# ------------------------------
# Entrenamiento y Predicción
# ------------------------------
def obtener_datos_entrenamiento() -> pd.DataFrame:
    raw = obtener_datos_actualizados()
    # simular mayor cantidad de datos
    sample = raw * 5
    for d in sample:
        d["resultado"] = random.choice([0, 1, 2])  # 0=local,1=empate,2=visitante
    rows = []
    for d in sample:
        try:
            rows.append({
                "goles_local_prom": d["goles_local_prom"],
                "goles_visita_prom": d["goles_visita_prom"],
                "racha_local": int(d["racha_local"]),
                "racha_visita": int(d["racha_visita"]),
                "clima": int(d["clima"]),
                "importancia_partido": int(d["importancia_partido"]),
                "resultado": int(d["resultado"])
            })
        except Exception as e:
            print(f"[Train] ERROR fila: {e}")
    df = pd.DataFrame(rows)
    print(f"[Train] DataFrame con {len(df)} filas, distribución: {df['resultado'].value_counts().to_dict()}")
    return df

def entrenar_modelo() -> xgb.XGBClassifier:
    df = obtener_datos_entrenamiento()
    if df.empty:
        raise RuntimeError("DataFrame de entrenamiento vacío")
    X = df.drop("resultado", axis=1)
    y = df["resultado"]
    model = xgb.XGBClassifier(eval_metric="mlogloss", use_label_encoder=False)
    model.fit(X, y)
    print(f"[Model] Entrenado con {len(y)} muestras")
    return model

def predecir_resultado(modelo, partido: dict):
    feat = pd.DataFrame([{
        "goles_local_prom": partido["goles_local_prom"],
        "goles_visita_prom": partido["goles_visita_prom"],
        "racha_local": partido["racha_local"],
        "racha_visita": partido["racha_visita"],
        "clima": partido["clima"],
        "importancia_partido": partido["importancia_partido"]
    }])
    pred = int(modelo.predict(feat)[0])
    mapping = {0: "Gana Local", 1: "Empate", 2: "Gana Visitante"}
    return mapping.get(pred, "Indefinido"), 0.75

# ------------------------------
# Pipeline Principal
# ------------------------------
def actualizar_datos_partidos():
    fixtures = obtener_datos_actualizados()
    for f in fixtures:
        league_id = upsert_league(f["liga"])
        record = {
            "league_id": league_id,
            "nombre_partido": f["nombre_partido"],
            "hora": f["hora"],
            "deporte": f["deporte"],
            "goles_local_prom": f["goles_local_prom"],
            "goles_visita_prom": f["goles_visita_prom"],
            "racha_local": f["racha_local"],
            "racha_visita": f["racha_visita"],
            "clima": f["clima"],
            "importancia_partido": f["importancia_partido"]
        }
        upsert_partido(record)
    return {"status": "partidos actualizados"}

def procesar_predicciones():
    model = entrenar_modelo()
    fixtures = obtener_datos_actualizados()
    for f in fixtures:
        league_id = upsert_league(f["liga"])
        hora = f["hora"]
        res, conf = predecir_resultado(model, f)
        pred = {
            "league_id": league_id,
            "partido": f["nombre_partido"],
            "hora": hora,
            "pronostico_1": res,
            "confianza_1": conf
        }
        try:
            supabase.table("predicciones").upsert(
                pred,
                on_conflict=["league_id", "partido", "hora"]
            ).execute()
            print(f"[Predicción] OK {pred['partido']} @ {hora}")
        except Exception as e:
            print(f"[Predicción] ERROR {pred['partido']}: {e}")
    return {"message": "Predicciones generadas"}

def ejecutar_pipeline():
    actualizar_datos_partidos()
    procesar_predicciones()

if __name__ == "__main__":
    ejecutar_pipeline()
