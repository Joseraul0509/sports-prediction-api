import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
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
    raise ValueError("Las claves de Supabase no están configuradas correctamente.")

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

@app.get("/")
def read_root():
    return {"message": "API de predicciones deportivas con XGBoost"}

@app.get("/api/v1/coincidencias")
def get_matches():
    try:
        response = supabase.table("matches").select("*").execute()
        if response.data is None:
            raise HTTPException(status_code=404, detail="No se encontraron coincidencias")
        return response.data
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error al obtener coincidencias: {str(e)}")

@app.get("/api/v1/predicciones")
def get_predictions():
    response = supabase.table("predictions").select("*").execute()
    return response.data

@app.get("/api/v1/ligas")
def get_ligas():
    return {
        "ligas": [
            "La Liga", "Premier League", "NBA", "MLB", "NHL"
        ]
    }

@app.get("/api/v1/equipos")
def get_equipos():
    return {
        "equipos": [
            "Real Madrid", "Barcelona", "Lakers", "Warriors",
            "Yankees", "Red Sox", "Maple Leafs", "Bruins"
        ]
    }

@app.post("/api/v1/predicciones")
def generar_predicciones():
    deportes = ["futbol", "nba", "mlb", "nhl"]
    nuevas_predicciones = []

    for deporte in deportes:
        response = supabase.table("matches").select("*").eq("deporte", deporte).execute()
        datos = response.data

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
                "deporte": deporte,
                "match_id": fila["id"],
                "prediccion": int(predicciones[i]),
                "confidence": confianza,
                "odds": 1.75,
                "type": "ML_PREDICTION",
                "fecha": datetime.now().isoformat(),
                "reasoning": "Predicción basada en rendimiento histórico"
            }

            try:
                supabase.table("predictions").insert(pred).execute()
                nuevas_predicciones.append(pred)
            except Exception as e:
                print(f"Error al guardar predicción: {e}")

    return {"mensaje": "Predicciones guardadas", "total": len(nuevas_predicciones)}

@app.put("/api/v1/predicciones/{id}")
def actualizar_prediccion(id: str):
    response = supabase.table("predictions").update({"verificado": True}).eq("id", id).execute()
    if not response.data:
        raise HTTPException(status_code=404, detail="Predicción no encontrada")
    return {"mensaje": f"Predicción {id} actualizada"}
