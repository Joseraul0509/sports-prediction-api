import os
from datetime import datetime
from supabase import create_client
from xgboost import XGBClassifier
import numpy as np
import random

url = os.getenv("SUPABASE_URL")
key = os.getenv("SUPABASE_KEY")
supabase = create_client(url, key)

def obtener_datos_actualizados():
    return [
        {
            "nombre_partido": "Equipo A vs Equipo B",
            "liga": "Premier League",
            "hora": "2025-04-08 15:00",
            "deporte": "futbol",
            "goles_local_prom": 1.8,
            "goles_visita_prom": 1.2,
            "racha_local": 4,
            "racha_visita": 2,
            "clima": 1,
            "importancia_partido": 3,
        }
    ]

def entrenar_y_predecir(partido):
    X = np.random.rand(100, 6)
    y = np.random.randint(0, 3, 100)

    modelo = XGBClassifier()
    modelo.fit(X, y)

    entrada = np.array([[
        partido["goles_local_prom"],
        partido["goles_visita_prom"],
        partido["racha_local"],
        partido["racha_visita"],
        partido["clima"],
        partido["importancia_partido"]
    ]])

    prediccion = modelo.predict_proba(entrada)[0]
    opciones = ["Gana Local", "Empate", "Gana Visitante"]

    top_indices = np.argsort(prediccion)[::-1][:3]

    return [
        {"resultado": opciones[i], "confianza": round(float(prediccion[i]), 2)}
        for i in top_indices
    ]

def procesar_partidos():
    partidos = obtener_datos_actualizados()

    for partido in partidos:
        predicciones = entrenar_y_predecir(partido)

        data = {
            "deporte": partido["deporte"],
            "liga": partido["liga"],
            "partido": partido["nombre_partido"],
            "hora": partido["hora"],
            "pronostico_1": predicciones[0]["resultado"],
            "confianza_1": predicciones[0]["confianza"],
            "pronostico_2": predicciones[1]["resultado"],
            "confianza_2": predicciones[1]["confianza"],
            "pronostico_3": predicciones[2]["resultado"],
            "confianza_3": predicciones[2]["confianza"],
        }

        supabase.table("predicciones").upsert(data, on_conflict=["partido", "hora"]).execute()
        print("Predicción guardada para:", data["partido"])
