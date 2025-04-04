import os
from fastapi import FastAPI
import uvicorn
from supabase import create_client, Client
from dotenv import load_dotenv  # Solo para pruebas locales

# Cargar variables de entorno (para pruebas locales)
load_dotenv()

# Leer claves de Supabase desde las variables de entorno
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# Verificar que las claves se han cargado correctamente
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
async def get_predicciones():
    return {"message": "Lista de predicciones"}

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