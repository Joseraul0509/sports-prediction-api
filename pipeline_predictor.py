import os
import uuid
import time
from datetime import datetime
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from supabase import create_client, Client
import requests
import random
from dotenv import load_dotenv

load_dotenv()

# --- Supabase client ---
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# --- API Keys & Endpoints ---
api_sports_key = os.getenv("API_SPORTS_KEY")
balldontlie_base_url = "https://api.balldontlie.io/v1"

# --- Helpers ---
def get_today():
    return datetime.utcnow().strftime("%Y-%m-%d")

def safe_numeric(value):
    """Convierte a float o devuelve 0.0 en caso de dict/None."""
    try:
        return float(value)
    except Exception:
        print(f"safe_numeric: Se recibió un {type(value).__name__}; se asigna 0.0")
        return 0.0

def upsert_partido(data_game: dict):
    """Hace upsert en 'partidos' con retries y logs."""
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

# --- Inserción de ligas/partidos simples ---
def insertar_league(nombre_liga: str, deporte: str):
    if nombre_liga.lower() == "desconocida":
        return
    try:
        existe = supabase.table("leagues").select("*").eq("name", nombre_liga).execute()
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

# --- Datos de Fútbol ---
OPENLIGADB_ENDPOINTS = [
    "https://api.openligadb.de/getmatchdata/dfb/2024/5",
    # ... otras ligas ...
]

def obtener_datos_futbol():
    datos = []
    today = get_today()
    # API-Sports fútbol
    try:
        r = requests.get(
            f"https://v3.football.api-sports.io/fixtures?date={today}",
            headers={"x-apisports-key": api_sports_key}, timeout=10
        )
        fixtures = r.json().get("response", []) if r.status_code == 200 else []
        print(f"Obtenidos {len(fixtures)} fixtures de API-Sports fútbol")
        for f in fixtures:
            teams = f.get("teams", {})
            datos.append({
                "nombre_partido": f"{teams.get('home',{}).get('name','Desconocido')} vs {teams.get('away',{}).get('name','Desconocido')}",
                "liga": f.get("league",{}).get("name","Desconocida"),
                "deporte": "futbol",
                "goles_local_prom": safe_numeric(f.get("goals",{}).get("home")),
                "goles_visita_prom": safe_numeric(f.get("goals",{}).get("away")),
                "racha_local": 3,
                "racha_visita": 2,
                "clima": 1,
                "importancia_partido": 3,
                "hora": f.get("fixture",{}).get("date", datetime.utcnow().isoformat())
            })
    except Exception as ex:
        print("Error API-Sports fútbol:", ex)

    # Complemento OpenLigaDB
    for ep in OPENLIGADB_ENDPOINTS:
        try:
            r2 = requests.get(ep, timeout=10)
            fixes = r2.json() if r2.status_code==200 else []
            print(f"Obtenidos {len(fixes)} fixtures de OpenLigaDB")
            for p in fixes:
                datos.append({
                    "nombre_partido": p.get("MatchName","Desconocido"),
                    "liga": p.get("League","Desconocida"),
                    "deporte": "futbol",
                    "goles_local_prom": safe_numeric(p.get("MatchResults",[{}])[0].get("PointsTeam1")),
                    "goles_visita_prom": safe_numeric(p.get("MatchResults",[{}])[0].get("PointsTeam2")),
                    "racha_local": 3,
                    "racha_visita": 2,
                    "clima": 1,
                    "importancia_partido": 3,
                    "hora": p.get("MatchDateTime", datetime.utcnow().isoformat())
                })
        except Exception as ex:
            print("Error OpenLigaDB:", ex)

    print(f"Total fútbol registros: {len(datos)}")
    return datos

# --- Datos de NBA ---
def obtener_datos_nba():
    datos = []
    today = get_today()
    # API-Sports NBA v2
    try:
        r2 = requests.get(
            f"https://v2.nba.api-sports.io/games?date={today}",
            headers={"x-apisports-key": api_sports_key}, timeout=10
        )
        resp = r2.json().get("response", []) if r2.status_code==200 else []
        print(f"Obtenidos {len(resp)} juegos de API-Sports NBA (v2)")
        for g in resp:
            teams = g.get("teams")
            if not isinstance(teams, dict):
                print("NBA v2: formato inesperado de teams, omitiendo", teams)
                continue
            scores = g.get("scores", {})
            datos.append({
                "nombre_partido": f"{teams['home']['name']} vs {teams['away']['name']}",
                "liga": g.get("league",{}).get("name","NBA"),
                "deporte": "NBA",
                "goles_local_prom": safe_numeric(scores.get("home"))/10,
                "goles_visita_prom": safe_numeric(scores.get("away"))/10,
                "racha_local": 3,
                "racha_visita": 2,
                "clima": 1,
                "importancia_partido": 3,
                "hora": g.get("date", datetime.utcnow().isoformat())
            })
    except Exception as ex:
        print("Excepción en API-Sports NBA (v2):", ex)

    print(f"Total NBA fixtures recopilados: {len(datos)}")
    return datos

# --- Datos MLB & NHL via API-Sports ---
def obtener_datos_deporte_api(deporte: str):
    datos = []
    today = get_today()
    if deporte.upper() == "MLB":
        url = "https://v1.baseball.api-sports.io/games"
    else:
        url = "https://v1.hockey.api-sports.io/games"
    try:
        r = requests.get(url, headers={"x-apisports-key": api_sports_key}, params={"date": today}, timeout=10)
        resp = r.json().get("response", []) if r.status_code==200 else []
        print(f"{deporte}: obtenidos {len(resp)} juegos de API-Sports")
        for g in resp:
            teams = g.get("teams", {})
            scores = g.get("scores", {})
            datos.append({
                "nombre_partido": f"{teams.get('home',{}).get('name','Local')} vs {teams.get('away',{}).get('name','Visitante')}",
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
    except Exception as ex:
        print(f"Excepción en API-Sports {deporte}:", ex)
    return datos

# --- Consolidación ---
def obtener_datos_actualizados():
    datos = []
    datos.extend(obtener_datos_futbol())
    datos.extend(obtener_datos_nba())
    for dep in ["MLB","NHL"]:
        datos.extend(obtener_datos_deporte_api(dep))
    print(f"Total de registros obtenidos de todas las fuentes: {len(datos)}")
    return datos

# --- Entrenamiento ---
def obtener_datos_entrenamiento():
    actual = obtener_datos_actualizados()
    sim = actual * 5
    for d in sim:
        d["resultado"] = random.choice([0,1,2])  # 0 local,1 empate,2 visita
    rows = []
    for d in sim:
        try:
            rows.append({
                "goles_local_prom": safe_numeric(d["goles_local_prom"]),
                "goles_visita_prom": safe_numeric(d["goles_visita_prom"]),
                "racha_local": int(d["racha_local"]),
                "racha_visita": int(d["racha_visita"]),
                "clima": int(d["clima"]),
                "importancia_partido": int(d["importancia_partido"]),
                "resultado": int(d["resultado"]),
            })
        except Exception as e:
            print("Error construyendo datos de entrenamiento:", e)
    df = pd.DataFrame(rows)
    print(f"Número de registros de entrenamiento: {len(df)}")
    print("Distribución de clases:", df["resultado"].value_counts().to_dict())
    return df

def entrenar_modelo():
    df = obtener_datos_entrenamiento()
    if df.empty or df["resultado"].nunique()<2:
        print("Datos insuficientes, entrenamiento abortado")
        return None
    X = df.drop("resultado", axis=1)
    y = df["resultado"]
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=42,stratify=y)
    model = xgb.XGBClassifier(eval_metric="mlogloss")
    model.fit(X_train, y_train)
    acc = (model.predict(X_test)==y_test).mean()
    print(f"Precisión del modelo: {acc:.2f}")
    return model

# --- Predicción ---
def predecir_resultado(modelo, d):
    features = pd.DataFrame([{
        "goles_local_prom": safe_numeric(d["goles_local_prom"]),
        "goles_visita_prom": safe_numeric(d["goles_visita_prom"]),
        "racha_local": int(d["racha_local"]),
        "racha_visita": int(d["racha_visita"]),
        "clima": int(d["clima"]),
        "importancia_partido": int(d["importancia_partido"]),
    }])
    pred = modelo.predict(features)[0]
    mapping = {0:("Gana Local",0.70),1:("Empate",0.50),2:("Gana Visitante",0.70)}
    return mapping.get(pred,("Indefinido",0.0))

def procesar_predicciones():
    model = entrenar_modelo()
    if model is None:
        print("No se generaron predicciones por falta de modelo")
        return
    partidos = obtener_datos_actualizados()
    for p in partidos:
        p["hora"] = datetime.fromisoformat(p["hora"][:19]).strftime("%Y-%m-%d %H:%M:%S")
        resultado, confianza = predecir_resultado(model, p)
        pred = {
            "deporte": p["deporte"],
            "liga": p["liga"],
            "partido": p["nombre_partido"],
            "hora": p["hora"],
            "pronostico_1": resultado,
            "confianza_1": confianza
        }
        upsert_partido(pred)
    print("Predicciones generadas correctamente.")

# --- Pipeline completo ---
def ejecutar_pipeline():
    try:
        # Insertar ligas y partidos
        for p in obtener_datos_actualizados():
            insertar_league(p["liga"], p["deporte"])
            upsert_partido(p)
        # Generar predicciones
        procesar_predicciones()
    except Exception as e:
        print("Error global en el pipeline:", e)

if __name__ == "__main__":
    ejecutar_pipeline()
