from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from supabase import create_client, Client
from pydantic import BaseModel
from dotenv import load_dotenv
import os
import time
from datetime import datetime
import pytz

load_dotenv()

url = os.getenv("SUPABASE_URL")
key = os.getenv("SUPABASE_KEY")

supabase: Client = create_client(url, key)

app = FastAPI()

# Permitir CORS para conexión con frontend o herramientas externas
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # luego podemos limitar esto
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Modelo para recibir parámetros en peticiones
class DeporteInput(BaseModel):
    deporte: str

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
