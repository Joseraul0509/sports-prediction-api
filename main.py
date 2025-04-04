import os
from fastapi import FastAPI
import uvicorn

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "API de predicciones deportivas en Render"}

@app.get("/api/v1/coincidencias")
def get_matches():
    return {"message": "Lista de coincidencias"}

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
