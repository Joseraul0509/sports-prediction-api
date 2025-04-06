import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from supabase import create_client, Client
from dotenv import load_dotenv
import xgboost as xgb
import numpy as np
import pandas as pd
from datetime import datetime

# Cargar variables de entorno
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("Faltan claves de Supabase en el archivo .env")

# Conectar con Supabase
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Crear FastAPI
app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Simulación de partidos por deporte
def obtener_partidos_del_dia(deporte: str):
    hoy = datetime.now().isoformat()
    partidos = {
        "nba": [{"team_home": "Lakers", "team_away": "Warriors", "score_home": 0, "score_away": 0, "match_date": hoy}],
        "mlb": [{"team_home": "Yankees", "team_away": "Red Sox", "score_home": 0, "score_away": 0, "match_date": hoy}],
        "nhl": [{"team_home": "Maple Leafs", "team_away": "Bruins", "score_home": 0, "score_away": 0, "match_date": hoy}],
        "futbol": [{"team_home": "Barcelona", "team_away": "Real Madrid", "score_home": 0, "score_away": 0, "match_date": hoy}]
    }
    return partidos.get(deporte, [])

@app.get("/")
def root():
    return {"mensaje": "API de predicciones deportivas activada."}

@app.post("/api/v1/auto")
def cargar_y_predecir_automaticamente():
    deportes = ["futbol", "nba", "mlb", "nhl"]
    partidos_guardados = []
    predicciones_guardadas = []

    for deporte in deportes:
        try:
            partidos = obtener_partidos_del_dia(deporte)
            for partido in partidos:
                partido["deporte"] = deporte
                res = supabase.table("matches").insert(partido).execute()
                partido["id"] = res.data[0]["id"] if res.data else None
                partidos_guardados.append(partido)
        except Exception as e:
            print(f"Error al guardar partido ({deporte}): {e}")
            continue

        try:
            res = supabase.table("matches").select("*").eq("deporte", deporte).execute()
            datos = res.data
            if not datos:
                continue

            df = pd.DataFrame(datos)
            if not {"score_home", "score_away", "id"}.issubset(df.columns):
                continue

            df.fillna(0, inplace=True)
            X = df[["score_home", "score_away"]]
            if X.empty:
                continue

            y = np.random.randint(0, 2, size=len(df))
            model = xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss")
            model.fit(X, y)

            predicciones = model.predict(X)
            probabilidades = model.predict_proba(X)

            for i, fila in df.iterrows():
                confianza = float(np.max(probabilidades[i]) * 100)
                pred = {
                    "sport": deporte,
                    "match_id": fila["id"],
                    "prediction": int(predicciones[i]),
                    "confidence": confianza,
                    "created_at": datetime.now().isoformat()
                }

                try:
                    supabase.table("predictions").insert(pred).execute()
                    predicciones_guardadas.append(pred)
                except Exception as e:
                    print(f"Error al guardar predicción: {e}")

        except Exception as e:
            print(f"Error en predicción automática ({deporte}): {e}")

    return {
        "mensaje": "Predicciones completadas",
        "partidos_guardados": len(partidos_guardados),
        "predicciones_guardadas": len(predicciones_guardadas)
    }
