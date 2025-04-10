from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from supabase import create_client, Client
from pydantic import BaseModel
from dotenv import load_dotenv
from datetime import datetime
import os
import sys
from pipeline_predictor import procesar_partidos, actualizar_datos_partidos

# Cargar variables de entorno
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
if not SUPABASE_URL or not SUPABASE_KEY:
    sys.exit("Error: Faltan variables de entorno en .env")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

app = FastAPI()

# Configuración CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Modelo de datos para validación
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

@app.get("/")
def root():
    return {"mensaje": "API de Predicción Deportiva funcionando."}

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

def guardar_prediccion(prediccion: dict):
    try:
        # Evitar duplicados: si ya existe registro para el mismo partido y hora, no lo inserta.
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

@app.post("/guardar_prediccion")
def endpoint_guardar_prediccion(prediccion: Prediccion):
    try:
        guardar_prediccion(prediccion.dict())
        return {"message": "Predicción procesada correctamente."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/test_guardar")
def test_guardar():
    prediccion_test = {
        "deporte": "futbol",
        "liga": "Premier League",
        "partido": "Arsenal vs Chelsea",
        "hora": datetime.utcnow().isoformat(),
        "pronostico_1": "Gana Arsenal",
        "confianza_1": 0.82,
        "pronostico_2": "Menos de 2.5 goles",
        "confianza_2": 0.77,
        "pronostico_3": "Ambos no anotan",
        "confianza_3": 0.69,
    }
    guardar_prediccion(prediccion_test)
    return {"message": "Test ejecutado"}

@app.post("/ejecutar_predicciones")
def ejecutar_predicciones():
    try:
        # Actualiza datos de partidos desde APIs deportivas reales
        actualizar_datos_partidos()
        # Procesa partidos: entrena el modelo y genera predicciones
        procesar_partidos()
        return {"message": "Predicciones procesadas correctamente."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
