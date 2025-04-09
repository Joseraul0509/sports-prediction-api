import xgboost as xgb
import numpy as np
import json
from supabase import create_client
import os

url = os.getenv("SUPABASE_URL")
key = os.getenv("SUPABASE_KEY")
supabase = create_client(url, key)

# Cargar el modelo directamente desde Supabase
modelo_binario = supabase.storage.from_("modelos").download("xgboost_model.json")
modelo = xgb.Booster()
modelo.load_model(modelo_binario.read())

def generar_boletos():
    datos = supabase.table("partidos").select("*").execute().data
    boletos = []

    for partido in datos:
        caracteristicas = np.array([
            partido["goles_local_prom"], partido["goles_visita_prom"],
            partido["racha_local"], partido["racha_visita"],
            partido["clima"], partido["importancia_partido"]
        ]).reshape(1, -1)

        dmatrix = xgb.DMatrix(caracteristicas)
        probabilidad = modelo.predict(dmatrix)[0]

        if probabilidad >= 0.7:
            boletos.append({
                "partido": partido["nombre_partido"],
                "probabilidad": float(probabilidad),
                "resultado_esperado": "Local" if probabilidad > 0.5 else "Visita"
            })

    return {"boletos": boletos}