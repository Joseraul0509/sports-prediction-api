import os
import uuid
import time
from datetime import datetime
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
import requests
from supabase import create_client, Client
from dotenv import load_dotenv

# Carga de .env
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Claves API
API_SPORTS_KEY = os.getenv("API_SPORTS_KEY")
BALLDONTLIE_KEY = "5293380d-9c4f-4e89-be15-bf19b1042182"
BALLDONTLIE_BASE = "https://api.balldontlie.io/v1"

# ===============================
# Helpers
# ===============================

def safe_numeric(x, fallback=0.0):
    """Convierte x a float o devuelve fallback."""
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, dict):
        for k in ("total","points","goals","value"):
            v = x.get(k)
            if isinstance(v, (int, float)):
                return float(v)
    try:
        return float(x)
    except:
        return fallback

def upsert_with_retries(table, data, conflict_cols, max_retries=3):
    """Upsert con reintentos y back-off. Retorna response o None."""
    for i in range(max_retries):
        try:
            return supabase.table(table).upsert(data, on_conflict=conflict_cols).execute()
        except Exception as e:
            print(f"Error en upsert {table}, intento {i+1}: {e}")
            time.sleep(1)
    return None

# ===============================
# Inserción en Supabase
# ===============================

def insertar_league(name: str) -> str:
    """Inserta o busca liga, retorna su UUID."""
    if name.lower() == "desconocida":
        return None
    resp = supabase.table("leagues").select("id").eq("name", name).execute()
    if resp.data:
        return resp.data[0]["id"]
    new_id = str(uuid.uuid4())
    supabase.table("leagues").insert({
        "id": new_id, "name": name, "country": None, "flag": None
    }).execute()
    return new_id

def insertar_partido(d: dict):
    """Upsert de partido. D debe contener league_id y hora ISO."""
    record = {
        "id": str(uuid.uuid4()),
        "league_id": d["league_id"],
        "nombre_partido": d["nombre_partido"],
        "hora": d["hora"],
        "goles_local_prom": safe_numeric(d.get("goles_local_prom")),
        "goles_visita_prom": safe_numeric(d.get("goles_visita_prom")),
        "racha_local": int(d.get("racha_local", 0)),
        "racha_visita": int(d.get("racha_visita", 0)),
        "clima": int(d.get("clima", 1)),
        "importancia_partido": int(d.get("importancia_partido", 1)),
        "stats": d.get("stats") or {}
    }
    upsert_with_retries("partidos", record, ["nombre_partido","hora"])

def insertar_prediccion(p: dict):
    """Upsert de predicción. p debe contener partido_id y hora ISO."""
    rec = {
        "id": str(uuid.uuid4()),
        "partido_id": p["partido_id"],
        "deporte": p["deporte"],
        "league_id": p.get("league_id"),
        "hora": p["hora"],
        "pronostico_1": p["pronostico_1"],
        "confianza_1": p["confianza_1"],
        "pronostico_2": p["pronostico_2"],
        "confianza_2": p["confianza_2"],
        "pronostico_3": p["pronostico_3"],
        "confianza_3": p["confianza_3"],
        "jugador": p.get("jugador")
    }
    upsert_with_retries("predicciones", rec, ["partido_id","hora"])

# ===============================
# Recolección de datos
# ===============================

def get_today(): return datetime.utcnow().strftime("%Y-%m-%d")

def obtener_datos_futbol():
    datos, t = [], get_today()
    print("→ Fetching Fútbol fixtures …")
    try:
        url = f"https://v3.football.api-sports.io/fixtures?date={t}"
        h = {"x-apisports-key": API_SPORTS_KEY}
        r = requests.get(url, headers=h, timeout=10).json().get("response",[])
        print(f"  • Fixtures fútbol API-Sports: {len(r)}")
        for f in r:
            league = f["league"]["name"]
            home, away = f["teams"]["home"], f["teams"]["away"]
            stats_url = ("https://v3.football.api-sports.io/"
                         f"teams/statistics?league={f['league']['id']}"
                         f"&team={home['id']}&season={f['league']['season']}")
            # … aquí podrías fetch stats y extraer campos reales …
            datos.append({
                "nombre_partido": home["name"]+" vs "+away["name"],
                "liga": league,
                "deporte": "futbol",
                # provisional:
                "goles_local_prom": f["goals"]["home"] if f["goals"]["home"] is not None else 0,
                "goles_visita_prom": f["goals"]["away"] if f["goals"]["away"] is not None else 0,
                "racha_local": 0, "racha_visita": 0,
                "clima":1, "importancia_partido":1,
                "hora": f["fixture"]["date"],
            })
    except Exception as e:
        print("  ! Error futbol:", e)
    return datos

def obtener_datos_nba():
    datos, t = [], get_today()
    print("→ Fetching NBA fixtures …")
    # API‑Sports v2
    try:
        url = f"https://v2.nba.api-sports.io/games?date={t}"
        h = {"x-apisports-key": API_SPORTS_KEY}
        resp = requests.get(url, headers=h, timeout=10)
        body = resp.json().get("response",[])
        print(f"  • NBA v2 games: {len(body)}")
        for g in body:
            home, away = g["teams"]["home"], g["teams"]["away"]
            datos.append({
                "nombre_partido": home["name"]+" vs "+away["name"],
                "liga": g["league"]["name"],
                "deporte": "NBA",
                "goles_local_prom": safe_numeric(g["scores"]["home"]),
                "goles_visita_prom": safe_numeric(g["scores"]["away"]),
                "racha_local":0, "racha_visita":0,
                "clima":1, "importancia_partido":1,
                "hora": g["date"]
            })
    except Exception as e:
        print("  ! NBA v2 error:", e)
    # fallback v1…
    return datos

def obtener_datos_deporte_api(dep):
    datos, t = [], get_today()
    print(f"→ Fetching {dep} fixtures …")
    ENDPOINT = {
        "MLB":"https://v1.baseball.api-sports.io/games",
        "NHL":"https://v1.hockey.api-sports.io/games"
    }.get(dep.upper())
    if not ENDPOINT: return datos
    try:
        r = requests.get(ENDPOINT, headers={"x-apisports-key":API_SPORTS_KEY},
                         params={"date":t}, timeout=10).json().get("response",[])
        print(f"  • {dep} games: {len(r)}")
        for g in r:
            h, a = g["teams"]["home"]["name"], g["teams"]["away"]["name"]
            datos.append({
                "nombre_partido": h+" vs "+a,
                "liga": g["league"]["name"],
                "deporte": dep,
                "goles_local_prom": safe_numeric(g["scores"]["home"]),
                "goles_visita_prom": safe_numeric(g["scores"]["away"]),
                "racha_local":0, "racha_visita":0,
                "clima":1, "importancia_partido":1,
                "hora": g["date"]
            })
    except Exception as e:
        print(f"  ! {dep} error:", e)
    return datos

def obtener_datos_actualizados():
    datos = []
    datos += obtener_datos_futbol()
    datos += obtener_datos_nba()
    for dep in ("MLB","NHL"):
        datos += obtener_datos_deporte_api(dep)
    print(f"Total fixtures brutales: {len(datos)}")
    return datos

# ===============================
# Entrenamiento y predicción
# ===============================

def obtener_datos_entrenamiento():
    df = obtener_datos_actualizados()
    # duplicamos para volumen
    rows = []
    for d in df*5:
        try:
            rows.append({
                "goles_local_prom": safe_numeric(d["goles_local_prom"]),
                "goles_visita_prom": safe_numeric(d["goles_visita_prom"]),
                "racha_local": int(d["racha_local"]),
                "racha_visita": int(d["racha_visita"]),
                "clima": int(d["clima"]),
                "importancia_partido": int(d["importancia_partido"]),
                "resultado": uuid.uuid4().int % 3  # simulado
            })
        except Exception as e:
            print("  ! Error construyendo fila ent:", e)
    df_train = pd.DataFrame(rows)
    print("Datos de entrenamiento:", df_train.shape)
    return df_train

def entrenar_modelo():
    df = obtener_datos_entrenamiento()
    if df.empty or len(df["resultado"].unique())<2:
        print("  ! Datos insuficientes, entrenando con todo")
    X = df.drop("resultado",1)
    y = df["resultado"]
    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss")
    model.fit(X,y)
    print("Modelo entrenado.")
    return model

def predecir_resultado(model, d):
    X = pd.DataFrame([{
        "goles_local_prom": safe_numeric(d["goles_local_prom"]),
        "goles_visita_prom": safe_numeric(d["goles_visita_prom"]),
        "racha_local": d["racha_local"],
        "racha_visita": d["racha_visita"],
        "clima": d["clima"],
        "importancia_partido": d["importancia_partido"]
    }])
    pred = model.predict(X)[0]
    opts = {0:("Local",0.7),1:("Empate",0.65),2:("Visitante",0.68)}
    return opts.get(pred,("Indef",0.0))

# ===============================
# Pipeline completo
# ===============================

def actualizar_datos_partidos():
    fixtures = obtener_datos_actualizados()
    for f in fixtures:
        # liga
        lid = insertar_league(f["liga"])
        if not lid: continue
        # hora ISO con timezone
        dt = datetime.fromisoformat(f["hora"])
        f["hora"] = dt.astimezone().isoformat()
        f["league_id"] = lid
        insertar_partido(f)

def procesar_predicciones():
    model = entrenar_modelo()
    # recupera TODOS los partidos del día
    resp = supabase.table("partidos").select("id,nombre_partido,hora,league_id").execute()
    for row in resp.data:
        partido_id = row["id"]
        hora = row["hora"]
        d = {
            "goles_local_prom":  row.get("goles_local_prom",0),
            "goles_visita_prom": row.get("goles_visita_prom",0),
            "racha_local":       row.get("racha_local",0),
            "racha_visita":      row.get("racha_visita",0),
            "clima":             row.get("clima",1),
            "importancia_partido":row.get("importancia_partido",1)
        }
        res, conf = predecir_resultado(model, d)
        insertar_prediccion({
            "partido_id": partido_id,
            "deporte":    row.get("deporte"),
            "league_id":  row.get("league_id"),
            "hora":       hora,
            "pronostico_1": res,
            "confianza_1": conf,
            "pronostico_2": "Under 2.5", "confianza_2": 0.7,
            "pronostico_3": "Ambos no anotan", "confianza_3": 0.65,
            "jugador": None
        })

def ejecutar_pipeline():
    print("=== INICIA PIPELINE ===")
    actualizar_datos_partidos()
    procesar_predicciones()
    print("=== PIPELINE COMPLETADO ===")
    return {"message":"¡Listo!"}

if __name__ == "__main__":
    ejecutar_pipeline()
