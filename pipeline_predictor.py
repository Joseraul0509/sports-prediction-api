import os
import uuid
import time
import random
from datetime import datetime, timezone
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import requests
from supabase import create_client, Client
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
API_SPORTS_KEY = os.getenv("API_SPORTS_KEY")  # tu clave para API‑Sports

# Cliente Supabase
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Helper numérico seguro
def safe_numeric(v, default=0.0):
    if isinstance(v, (int, float)):
        return float(v)
    try:
        return float(v)
    except Exception:
        print(f"safe_numeric: Se recibió {type(v)}; asignando {default}")
        return default

# Fecha hoy en UTC (ISO date)
def get_today():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")

# Inserción/Upsert genérico con reintentos
def upsert_with_retries(table, data, conflict_cols, max_retries=3):
    for i in range(1, max_retries+1):
        try:
            res = supabase.table(table).upsert(data, on_conflict=conflict_cols).execute()
            return res
        except Exception as e:
            print(f"Error en upsert {table}, intento {i}: {e}")
            time.sleep(1)
    print(f"Falló upsert definitivo en {table}: {data.get('nombre_partido', data)}")
    return None

# ============================
# Funciones de extracción
# ============================

def obtener_datos_futbol():
    today = get_today()
    url = f"https://v3.football.api-sports.io/fixtures?date={today}"
    headers = {"x-apisports-key": API_SPORTS_KEY}
    datos = []
    try:
        r = requests.get(url, headers=headers, timeout=10)
        fixtures = r.json().get("response", [])
        print(f"Obtenidos {len(fixtures)} fixtures de API‑Sports fútbol")
        for f in fixtures:
            fix = f.get("fixture", {})
            league = f.get("league", {}) or {}
            teams = f.get("teams", {}) or {}
            home = teams.get("home") or {}
            away = teams.get("away") or {}
            if not all(isinstance(x, dict) for x in (home, away)):
                continue
            datos.append({
                "id": str(uuid.uuid4()),
                "nombre_partido": f"{home.get('name','')} vs {away.get('name','')}",
                "liga": league.get("name", "Desconocida"),
                "deporte": "futbol",
                "goles_local_prom": safe_numeric(f.get("goals", {}).get("home")),
                "goles_visita_prom": safe_numeric(f.get("goals", {}).get("away")),
                "racha_local": 0,
                "racha_visita": 0,
                "clima": 1,
                "importancia_partido": 1,
                "hora": fix.get("date", datetime.now(timezone.utc).isoformat())
            })
    except Exception as e:
        print("Error en API‑Sports fútbol:", e)
    return datos

def obtener_datos_nba():
    today = get_today()
    datos = []
    # Intentamos v2
    for endpoint in [f"https://v2.nba.api-sports.io/games?date={today}",
                     f"https://v1.basketball.api-sports.io/games?date={today}"]:
        try:
            r = requests.get(endpoint, headers={"x-apisports-key": API_SPORTS_KEY}, timeout=10)
            fixtures = r.json().get("response", [])
            print(f"Obtenidos {len(fixtures)} juegos de API‑Sports NBA ({endpoint.split('//')[1].split('/')[0]})")
            for g in fixtures:
                teams = g.get("teams") or {}
                home = teams.get("home") or {}
                away = teams.get("away") or {}
                if not all(isinstance(x, dict) for x in (home, away)):
                    continue
                datos.append({
                    "id": str(uuid.uuid4()),
                    "nombre_partido": f"{home.get('name','')} vs {away.get('name','')}",
                    "liga": g.get("league", {}).get("name", "NBA"),
                    "deporte": "NBA",
                    "goles_local_prom": safe_numeric(g.get("scores", {}).get("home")),
                    "goles_visita_prom": safe_numeric(g.get("scores", {}).get("away")),
                    "racha_local": 0,
                    "racha_visita": 0,
                    "clima": 1,
                    "importancia_partido": 1,
                    "hora": g.get("date", datetime.now(timezone.utc).isoformat())
                })
            if datos:
                break
        except Exception as e:
            print(f"Excepción en API‑Sports NBA ({endpoint}):", e)
    return datos

def obtener_datos_deporte_api(deporte):
    today = get_today()
    base = {
        "MLB": "https://v1.baseball.api-sports.io/games",
        "NHL": "https://v1.hockey.api-sports.io/games"
    }.get(deporte.upper())
    datos = []
    if not base:
        return datos
    try:
        r = requests.get(base, headers={"x-apisports-key": API_SPORTS_KEY},
                         params={"date": today}, timeout=10)
        fixtures = r.json().get("response", [])
        print(f"{deporte}: obtenidos {len(fixtures)} juegos de API‑Sports")
        for g in fixtures:
            teams = g.get("teams") or {}
            home = teams.get("home") or {}
            away = teams.get("away") or {}
            if not all(isinstance(x, dict) for x in (home, away)):
                continue
            datos.append({
                "id": str(uuid.uuid4()),
                "nombre_partido": f"{home.get('name','')} vs {away.get('name','')}",
                "liga": g.get("league", {}).get("name", deporte),
                "deporte": deporte.lower(),
                "goles_local_prom": safe_numeric(g.get("scores", {}).get("home")),
                "goles_visita_prom": safe_numeric(g.get("scores", {}).get("away")),
                "racha_local": 0,
                "racha_visita": 0,
                "clima": 1,
                "importancia_partido": 1,
                "hora": g.get("date", datetime.now(timezone.utc).isoformat())
            })
    except Exception as e:
        print(f"Error en API‑Sports {deporte}:", e)
    return datos

def obtener_datos_actualizados():
    datos = []
    datos.extend(obtener_datos_futbol())
    datos.extend(obtener_datos_nba())
    for dep in ["MLB", "NHL"]:
        datos.extend(obtener_datos_deporte_api(dep))
    print(f"Total registros obtenidos de todas las fuentes: {len(datos)}")
    return datos

# ============================
# Entrenamiento & Predicción
# ============================

def obtener_datos_entrenamiento():
    partidos = obtener_datos_actualizados()
    print(f"Número de registros obtenidos para entrenamiento: {len(partidos)}")
    rows = []
    for p in partidos:
        try:
            rows.append({
                "goles_local_prom": p["goles_local_prom"],
                "goles_visita_prom": p["goles_visita_prom"],
                "racha_local": p["racha_local"],
                "racha_visita": p["racha_visita"],
                "clima": p["clima"],
                "importancia_partido": p["importancia_partido"],
                "resultado": random.choice([0,1,2])
            })
        except Exception as e:
            print("Error construyendo fila entrenamiento:", e)
    df = pd.DataFrame(rows)
    if df.empty or df["resultado"].nunique()<2:
        print("DataFrame de entrenamiento inválido.")
    else:
        print("DataFrame de entrenamiento listo. Distribución:", df["resultado"].value_counts().to_dict())
    return df

def entrenar_modelo():
    df = obtener_datos_entrenamiento()
    if df.empty or df["resultado"].nunique()<2:
        return None
    X = df.drop("resultado", axis=1)
    y = df["resultado"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    model = xgb.XGBClassifier(eval_metric="mlogloss")
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    print("Precisión del modelo:", accuracy_score(y_test, preds))
    return model

def predecir_resultado(modelo, p):
    df = pd.DataFrame([{
        "goles_local_prom": p["goles_local_prom"],
        "goles_visita_prom": p["goles_visita_prom"],
        "racha_local": p["racha_local"],
        "racha_visita": p["racha_visita"],
        "clima": p["clima"],
        "importancia_partido": p["importancia_partido"]
    }])
    try:
        pred = modelo.predict(df)[0]
    except Exception:
        return "Indefinido", 0.0
    opciones = {
        0: ("Gana Local", 0.75),
        1: ("Empate", 0.65),
        2: ("Gana Visitante", 0.75)
    }
    return opciones.get(pred, ("Indefinido", 0.0))

# ============================
# Pipeline: carga y predicción
# ============================

def actualizar_datos_partidos():
    for p in obtener_datos_actualizados():
        upsert_with_retries("leagues", {
            "id": p["id"],
            "name": p["liga"],
            "country": None,
            "flag": None
        }, ["id"])
        upsert_with_retries("partidos", p, ["id"])
    print("Partidos y ligas actualizados.")

def procesar_predicciones():
    model = entrenar_modelo()
    if model is None:
        print("No hay modelo entrenado; se omiten predicciones.")
        return
    for p in obtener_datos_actualizados():
        res, conf = predecir_resultado(model, p)
        pred = {
            "id": str(uuid.uuid4()),
            "deporte": p["deporte"],
            "liga": p["liga"],
            "partido": p["nombre_partido"],
            "hora": p["hora"],
            "pronostico_1": res,
            "confianza_1": conf
        }
        upsert_with_retries("predicciones", pred, ["id"])
    print("Predicciones generadas.")

def ejecutar_pipeline():
    actualizar_datos_partidos()
    procesar_predicciones()

if __name__ == "__main__":
    ejecutar_pipeline()
