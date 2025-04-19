import os
import uuid
import time
from datetime import datetime
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import requests
from supabase import create_client, Client
import random
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

# Conexión a Supabase
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Claves API
api_sports_key = os.getenv("API_SPORTS_KEY")
balldontlie_base_url = "https://api.balldontlie.io/v1"
balldontlie_api_key = "5293380d-9c4f-4e89-be15-bf19b1042182"

# ===============================
# Funciones de inserción y upsert
# ===============================

def insertar_league(nombre_liga: str, deporte: str):
    try:
        if nombre_liga.lower() == "desconocida":
            print("Liga no insertada: valor 'Desconocida'")
            return
        existe = supabase.table("leagues").select("*").match({"name": nombre_liga}).execute()
        if not existe.data:
            supabase.table("leagues").insert({
                "id": str(uuid.uuid4()),
                "name": nombre_liga,
                "country": "Unknown",
                "flag": None,
            }).execute()
            print(f"Liga insertada: {nombre_liga}")
    except Exception as e:
        print(f"Error al insertar liga {nombre_liga}: {e}")

def upsert_partido(data_game: dict):
    """
    Inserta o actualiza un partido usando ON CONFLICT (nombre_partido, hora).
    """
    for intento in range(1, 4):
        try:
            supabase.table("partidos") \
                .upsert(data_game, on_conflict=["nombre_partido", "hora"]) \
                .execute()
            print(f"Upsert OK en partidos: {data_game['nombre_partido']}")
            return
        except Exception as e:
            print(f"Error en upsert partidos, intento {intento}: {e}")
            time.sleep(1)
    print(f"Falló upsert definitivo en partidos: {data_game['nombre_partido']}")

# ===============================
# Helpers
# ===============================

def get_today():
    return datetime.utcnow().strftime("%Y-%m-%d")

def safe_numeric(val):
    """
    Intenta convertir a float; si es dict o None o falla, retorna 0.0.
    """
    if isinstance(val, dict):
        return float(val.get("points", 0) if "points" in val else 0)
    try:
        return float(val)
    except:
        return 0.0

# ===============================
# Configuración Endpoints
# ===============================

OPENLIGADB_ENDPOINTS = [
    "https://api.openligadb.de/getmatchdata/dfb/2024/5",
    # ... demás endpoints ...
]

# ===============================
# Obtención de datos por deporte
# ===============================

def obtener_datos_futbol():
    datos = []
    today = get_today()
    print(f"Obteniendo fútbol para {today} desde API-Sports v3...")
    try:
        url_api = f"https://v3.football.api-sports.io/fixtures?date={today}"
        headers = {"x-apisports-key": api_sports_key}
        resp = requests.get(url_api, headers=headers, timeout=10)
        if resp.status_code == 200:
            fixtures = resp.json().get("response", [])
            print(f"Obtenidos {len(fixtures)} fixtures de API-Sports fútbol")
            for f in fixtures:
                fx = f.get("fixture", {})
                teams = f.get("teams", {})
                stats = f.get("goals", {})
                datos.append({
                    "nombre_partido": f"{teams.get('home', {}).get('name','')} vs {teams.get('away',{}).get('name','')}",
                    "liga": f.get("league", {}).get("name", "Desconocida"),
                    "deporte": "futbol",
                    "goles_local_prom": safe_numeric(stats.get("home")),
                    "goles_visita_prom": safe_numeric(stats.get("away")),
                    "racha_local": 3,
                    "racha_visita": 2,
                    "clima": 1,
                    "importancia_partido": 3,
                    "hora": fx.get("date", datetime.utcnow().isoformat())
                })
        else:
            print("Error en API-Sports fútbol:", resp.status_code, resp.text)
    except Exception as e:
        print("Excepción en API-Sports fútbol:", e)

    # Complemento OpenLigaDB...
    total_ol = 0
    for ep in OPENLIGADB_ENDPOINTS:
        try:
            r = requests.get(ep, timeout=10)
            if r.status_code == 200:
                fixtures = r.json()
                total_ol += len(fixtures)
                for p in fixtures:
                    datos.append({
                        "nombre_partido": p.get("MatchName", "Partido Desconocido"),
                        "liga": p.get("League", "Desconocida"),
                        "deporte": "futbol",
                        "goles_local_prom": p.get("MatchResults", [{}])[0].get("PointsTeam1",1.5),
                        "goles_visita_prom": p.get("MatchResults", [{}])[0].get("PointsTeam2",1.2),
                        "racha_local": 3,
                        "racha_visita": 2,
                        "clima": 1,
                        "importancia_partido": 3,
                        "hora": p.get("MatchDateTime", datetime.utcnow().isoformat())
                    })
        except Exception as e:
            print("Error en OpenLigaDB:", e)
    print(f"Obtenidos {total_ol} fixtures de OpenLigaDB")
    return datos

def obtener_datos_nba():
    datos = []
    today = get_today()
    total = 0
    # API-Sports v2
    try:
        url_v2 = f"https://v2.nba.api-sports.io/games?date={today}"
        headers = {"x-apisports-key": api_sports_key}
        r2 = requests.get(url_v2, headers=headers, timeout=10)
        if r2.status_code == 200:
            resp = r2.json().get("response", [])
            print(f"Obtenidos {len(resp)} juegos de API-Sports NBA (v2)")
            for g in resp:
                teams = g.get("teams", {})
                score = g.get("scores", {})
                datos.append({
                    "nombre_partido": f"{teams.get('home',{}).get('name','Desconocido')} vs {teams.get('away',{}).get('name','')}",
                    "liga": g.get("league",{}).get("name","NBA"),
                    "deporte": "NBA",
                    "goles_local_prom": safe_numeric(score.get("home"))/10,
                    "goles_visita_prom": safe_numeric(score.get("away"))/10,
                    "racha_local": 3,
                    "racha_visita": 2,
                    "clima": 1,
                    "importancia_partido": 3,
                    "hora": g.get("date", datetime.utcnow().isoformat())
                })
            total += len(resp)
        else:
            print("Error en API-Sports NBA (v2):", r2.status_code)
    except Exception as e:
        print("Excepción en API-Sports NBA (v2):", e)
    # ...Aquí podrías intentar v1 y balldontlie de forma similar...
    print(f"Total NBA fixtures recopilados: {total}")
    return datos

def obtener_datos_deporte_api(deporte):
    datos = []
    today = get_today()
    if deporte.upper() == "MLB":
        endpoint = "https://v1.baseball.api-sports.io/games"
    elif deporte.upper() == "NHL":
        endpoint = "https://v1.hockey.api-sports.io/games"
    else:
        return datos
    try:
        r = requests.get(endpoint, headers={"x-apisports-key": api_sports_key}, params={"date": today}, timeout=10)
        resp = r.json().get("response", [])
        print(f"{deporte}: obtenidos {len(resp)} juegos de API-Sports")
        for g in resp:
            teams = g.get("teams", {})
            scores = g.get("scores", {})
            datos.append({
                "nombre_partido": f"{teams.get('home',{}).get('name','')} vs {teams.get('away',{}).get('name','')}",
                "liga": g.get("league",{}).get("name","Desconocida"),
                "deporte": deporte,
                "goles_local_prom": safe_numeric(scores.get("home")),
                "goles_visita_prom": safe_numeric(scores.get("away")),
                "racha_local": 0,
                "racha_visita": 0,
                "clima": 1,
                "importancia_partido": 3,
                "hora": g.get("date", datetime.utcnow().isoformat())
            })
    except Exception as e:
        print(f"Error en API-Sports {deporte}:", e)
    return datos

def obtener_datos_actualizados():
    datos = []
    datos.extend(obtener_datos_futbol())
    datos.extend(obtener_datos_nba())
    for dep in ["MLB", "NHL"]:
        datos.extend(obtener_datos_deporte_api(dep))
    print(f"Total de registros obtenidos de todas las fuentes: {len(datos)}")
    return datos

# ===============================
# Entrenamiento y predicción
# ===============================

def obtener_datos_entrenamiento():
    datos = obtener_datos_actualizados()
    if not datos:
        print("El DataFrame de entrenamiento está vacío.")
        return pd.DataFrame()
    print(f"Número de registros obtenidos para entrenamiento: {len(datos)}")
    # Duplicar, simular resultados, etc.
    sim = datos * 10
    rows = []
    for d in sim:
        try:
            rows.append({
                "goles_local_prom": float(d.get("goles_local_prom", 0.0)),
                "goles_visita_prom": float(d.get("goles_visita_prom", 0.0)),
                "racha_local": int(d.get("racha_local", 0)),
                "racha_visita": int(d.get("racha_visita", 0)),
                "clima": int(d.get("clima", 1)),
                "importancia_partido": int(d.get("importancia_partido", 3)),
                "resultado": random.choice([0,1,2])
            })
        except Exception as e:
            print("Error construyendo datos de entrenamiento:", e)
    df = pd.DataFrame(rows)
    if df.empty or df["resultado"].nunique() < 2:
        print("Datos insuficientes o una sola clase en 'resultado'.")
    else:
        print("Datos de entrenamiento listos. Distribución de clases:", df["resultado"].value_counts().to_dict())
    return df

def entrenar_modelo():
    df = obtener_datos_entrenamiento()
    if df.empty or df["resultado"].nunique() < 2:
        print("No hay datos suficientes para entrenar un modelo válido.")
        return None
    X = df.drop("resultado", axis=1)
    y = df["resultado"]
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.3,
                                                        random_state=42,
                                                        stratify=y)
    model = xgb.XGBClassifier(eval_metric="logloss", use_label_encoder=False)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    print(f"Precisión del modelo: {accuracy_score(y_test, preds):.2f}")
    return model

def predecir_resultado(modelo, partido):
    df = pd.DataFrame([{
        "goles_local_prom": partido["goles_local_prom"],
        "goles_visita_prom": partido["goles_visita_prom"],
        "racha_local": partido["racha_local"],
        "racha_visita": partido["racha_visita"],
        "clima": partido["clima"],
        "importancia_partido": partido["importancia_partido"]
    }])
    try:
        pred = modelo.predict(df)[0]
    except Exception as e:
        print("Error prediciendo:", e)
        return ("Indefinido", 0.0)
    opciones = {
        0: ("Gana Local", 0.70),
        1: ("Empate",     0.65),
        2: ("Gana Visit", 0.68),
    }
    return opciones.get(pred, ("Indefinido", 0.0))

# ===============================
# Pipeline: actualizar + predecir
# ===============================

def actualizar_datos_partidos():
    nuevos = obtener_datos_actualizados()
    for p in nuevos:
        # normalizar hora
        h = p["hora"]
        if isinstance(h, str):
            try:
                h = datetime.fromisoformat(h)
            except:
                h = datetime.utcnow()
        p["hora"] = h.strftime("%Y-%m-%d %H:%M:%S")
        insertar_league(p["liga"], p["deporte"])
        upsert_partido(p)
    return {"status": "Datos actualizados correctamente"}

def procesar_predicciones():
    try:
        model = entrenar_modelo()
        if model is None:
            raise ValueError("Modelo no entrenado.")
        partidos = obtener_datos_actualizados()
        for p in partidos:
            h = p["hora"]
            if isinstance(h, str):
                try:
                    h = datetime.fromisoformat(h)
                except:
                    h = datetime.utcnow()
            p["hora"] = h.strftime("%Y-%m-%d %H:%M:%S")
            res, conf = predecir_resultado(model, p)
            pred = {
                "deporte": p["deporte"],
                "liga": p["liga"],
                "partido": p["nombre_partido"],
                "hora": p["hora"],
                "pronostico_1": res,
                "confianza_1": conf,
                "pronostico_2": "Menos de 2.5 goles",
                "confianza_2": 0.70,
                "pronostico_3": "Ambos no anotan",
                "confianza_3": 0.65
            }
            supabase.table("predicciones") \
                .upsert(pred, on_conflict=["partido","hora"]) \
                .execute()
            print(f"Predicción guardada: {pred['partido']}")
        return {"message": "Predicciones generadas correctamente"}
    except Exception as e:
        print("Error global en el procesamiento de predicciones:", e)
        return {"message": "Error en predicciones", "error": str(e)}

def ejecutar_pipeline():
    actualizar_datos_partidos()
    procesar_predicciones()

if __name__ == "__main__":
    ejecutar_pipeline()
