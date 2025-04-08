from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from supabase import create_client, Client
from pydantic import BaseModel
from dotenv import load_dotenv
import os
import time
from datetime import datetime
import pytz
import requests
from pipeline_predictor import procesar_partidos

# Cargar las variables de entorno
load_dotenv()

url = os.getenv("SUPABASE_URL")
key = os.getenv("SUPABASE_KEY")

supabase: Client = create_client(url, key)

app = FastAPI()

# Permitir CORS para conexión con frontend o herramientas externas
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------
# Modelos para validación
# -----------------------
class DeporteInput(BaseModel):
    deporte: str

class Prediccion(BaseModel):
    deporte: str
    liga: str
    partido: str
    hora: str
    pronostico_1: str
    confianza_1: float
    pronostico_2: str
    confianza_2: float
    pronostico_3: str
    confianza_3: float

# -----------------------
# Rutas básicas
# -----------------------

@app.get("/")
def root():
    return {"message": "API de Predicción Deportiva funcionando."}

@app.post("/deportes_disponibles")
def deportes_disponibles(input: DeporteInput):
    try:
        response = supabase.table("predicciones").select("deporte").eq("deporte", input.deporte).execute()
        if response.data:
            return {"deporte": input.deporte}
        else:
            raise HTTPException(status_code=404, detail="Deporte no encontrado.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# -----------------------
# Función para guardar predicciones
# -----------------------

def guardar_prediccion(prediccion):
    try:
        existe = supabase.table("predicciones").select("*").match({
            "partido": prediccion["partido"],
            "liga": prediccion["liga"],
            "hora": prediccion["hora"]
        }).execute()

        if not existe.data:
            supabase.table("predicciones").insert(prediccion).execute()
            print("Predicción guardada:", prediccion["partido"])
        else:
            print("Ya existe predicción para:", prediccion["partido"])
    except Exception as e:
        print("Error al guardar predicción:", str(e))

# -----------------------
# Ruta para recibir y guardar predicciones desde tu modelo
# -----------------------

@app.post("/guardar_prediccion")
def endpoint_guardar_prediccion(prediccion: Prediccion):
    try:
        guardar_prediccion(prediccion.dict())
        return {"message": "Predicción procesada correctamente."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# -----------------------
# TEST: Simulación manual para probar
# -----------------------

@app.get("/test_guardar")
def test_guardar():
    prediccion_test = {
        "deporte": "futbol",
        "liga": "Premier League",
        "partido": "Arsenal vs Chelsea",
        "hora": "2025-04-08 15:00",
        "pronostico_1": "Gana Arsenal",
        "confianza_1": 0.82,
        "pronostico_2": "Menos de 2.5 goles",
        "confianza_2": 0.77,
        "pronostico_3": "Ambos no anotan",
        "confianza_3": 0.69,
    }
    guardar_prediccion(prediccion_test)
    return {"message": "Test ejecutado"}

# -----------------------
# Ejecutar la predicción en background
# -----------------------

@app.post("/ejecutar_predicciones")
def ejecutar_predicciones():
    try:
        procesar_partidos()  # Llama la función del pipeline
        return {"message": "Predicciones procesadas correctamente."}
    except Exception as e:
        return HTTPException(status_code=500, detail=str(e))
