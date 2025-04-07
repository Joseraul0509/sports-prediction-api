import os
import time
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from supabase import create_client, Client
from dotenv import load_dotenv
import xgboost as xgb
import numpy as np
import pandas as pd
from datetime import datetime
import requests

load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
X_AUTH_TOKEN = os.getenv("X_AUTH_TOKEN")
API_FUTBOL_KEY = os.getenv("API_FUTBOL_KEY")
SCRAPER_API_URL = os.getenv("SCRAPER_API_URL")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("Faltan claves de Supabase en el archivo .env")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Simulador inicial - reemplazar por funciones reales

def obtener_partidos_externos():
    partidos = []
    headers = {"X-Auth-Token": X_AUTH_TOKEN}
    urls_openliga = [
        "https://api.openligadb.de/getmatchdata/dfb/2024/5",
        "https://api.openligadb.de/getmatchdata/bl2/2024/28",
        "https://api.openligadb.de/getmatchdata/bl3/2024/31",
        "https://api.openligadb.de/getmatchdata/ucl24/2024/12",
        "https://api.openligadb.de/getmatchdata/bl1/2024/28",
        "https://api.openligadb.de/getmatchdata/ucl2024/2024/4",
        "https://api.openligadb.de/getmatchdata/uel24/2024/12",
    ]
    for url in urls_openliga:
        try:
            res = requests.get(url)
            data = res.json()
            for match in data:
                partidos.append({
                    "team_home": match["Team1"]["TeamName"],
                    "team_away": match["Team2"]["TeamName"],
                    "score_home": match.get("MatchResults", [{}])[0].get("PointsTeam1", 0),
                    "score_away": match.get("MatchResults", [{}])[0].get("PointsTeam2", 0),
                    "match_date": match.get("MatchDateTime", datetime.now().isoformat()),
                    "deporte": "futbol",
                    "liga": match.get("LeagueName", "")
                })
        except:
            continue
    return partidos

@app.get("/")
def root():
    return {"mensaje": "API de predicciones deportivas activa."}

@app.post("/api/v1/auto")
def generar_predicciones():
    partidos = obtener_partidos_externos()
    partidos_guardados = []
    predicciones_guardadas = []

    for partido in partidos:
        try:
            res = supabase.table("matches").insert(partido).execute()
            time.sleep(0.2)
            partido_id = res.data[0]["id"] if res.data else None
            if not partido_id:
                continue

            X = np.array([[partido["score_home"], partido["score_away"]]])
            y = np.array([1])
            model = xgb.XGBClassifier(eval_metric="logloss")
            model.fit(X, y)
            pred = model.predict(X)[0]
            prob = float(model.predict_proba(X)[0][pred] * 100)

            # Scraping de cuotas en tiempo real
            cuotas = requests.get(SCRAPER_API_URL, params={
                "deporte": partido["deporte"],
                "local": partido["team_home"],
                "visitante": partido["team_away"]
            }).json()

            prediccion = {
                "match_id": partido_id,
                "sport": partido["deporte"],
                "prediction": int(pred),
                "confidence": prob,
                "odds": cuotas.get("mejor_cuota", 1.5),
                "type": "ganador",
                "created_at": datetime.now().isoformat()
            }
            supabase.table("predictions").insert(prediccion).execute()
            time.sleep(0.2)
            partidos_guardados.append(partido)
            predicciones_guardadas.append(prediccion)
        except Exception as e:
            print("Error en predicción:", e)
            continue

    return {
        "mensaje": "Predicciones completadas",
        "partidos_guardados": len(partidos_guardados),
        "predicciones_guardadas": len(predicciones_guardadas)
    }
