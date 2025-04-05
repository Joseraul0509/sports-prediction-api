import os
from fastapi import FastAPI
import uvicorn
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

# Crear la app de FastAPI
app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "API de predicciones deportivas en Render"}

@app.get("/api/v1/coincidencias")
def get_matches():
    response = supabase.table("matches").select("*").execute()
    return response.data

@app.get("/api/v1/predicciones")
def get_predictions():
    """ Genera predicciones usando XGBoost con datos de Supabase """

    # Obtener datos de Supabase
    response = supabase.table("matches").select("*").execute()
    matches = response.data

    if not matches:
        return {"error": "No hay datos suficientes para hacer predicciones"}

    # Convertir a DataFrame
    df = pd.DataFrame(matches)

    # Verificar si las columnas necesarias existen
    if "score_home" not in df.columns or "score_away" not in df.columns:
        return {"error": "Faltan columnas necesarias en la base de datos"}

    # Llenar valores nulos con 0
    df.fillna(0, inplace=True)

    # Variables de entrada (simples por ahora, mejoraremos después)
    X = df[["score_home", "score_away"]]

    # Etiquetas simuladas (0 o 1) - Esto luego lo cambiaremos con datos reales
    y = np.random.randint(0, 2, size=len(df))

    # Crear y entrenar modelo XGBoost (temporalmente con datos ficticios)
    model = xgb.XGBClassifier()
    model.fit(X, y)

    # Generar predicciones
    predictions = model.predict(X)

    # Agregar las predicciones al DataFrame
    df["prediccion"] = predictions

    return df.to_dict(orient="records")

@app.post("/api/v1/generar_predicciones")
def generar_y_guardar_predicciones():
    """ Genera predicciones y las guarda en Supabase """

    # Obtener datos de Supabase
    response = supabase.table("matches").select("*").execute()
    matches = response.data

    if not matches:
        return {"error": "No hay datos suficientes para hacer predicciones"}

    # Convertir a DataFrame
    df = pd.DataFrame(matches)
    df.fillna(0, inplace=True)

    # Variables de entrada
    X = df[["score_home", "score_away"]]
    y = np.random.randint(0, 2, size=len(df))  # Simulación de etiquetas

    # Entrenar modelo XGBoost
    model = xgb.XGBClassifier()
    model.fit(X, y)

    # Generar predicciones
    df["prediccion"] = model.predict(X)
    df["timestamp"] = datetime.utcnow().isoformat()

    # Guardar en Supabase
    for _, row in df.iterrows():
        supabase.table("predictions").insert({
            "match_id": row["id"],
            "team_home": row["team_home"],
            "team_away": row["team_away"],
            "prediccion": int(row["prediccion"]),
            "timestamp": row["timestamp"]
        }).execute()

    return {"message": "Predicciones generadas y guardadas en Supabase"}

@app.get("/api/v1/ligas")
async def get_ligas():
    return {"message": "Lista de ligas"}

@app.get("/api/v1/equipos")
async def get_equipos():
    return {"message": "Lista de equipos"}

@app.post("/api/v1/predicciones")
async def post_predicciones():
    return {"message": "Predicción creada"}

@app.put("/api/v1/predicciones/{id}")
async def put_predicciones(id: int):
    return {"message": f"Predicción {id} actualizada"}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
