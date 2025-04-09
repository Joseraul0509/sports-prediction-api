import requests
from datetime import datetime
from supabase import create_client
import os

url = os.getenv("SUPABASE_URL")
key = os.getenv("SUPABASE_KEY")
supabase = create_client(url, key)

# Simula extracción de datos actualizados

def obtener_datos_actualizados():
    # En un caso real, harías scraping o usarías una API
    return [
        {
            "nombre_partido": "Equipo A vs Equipo B",
            "goles_local_prom": 1.8,
            "goles_visita_prom": 1.2,
            "racha_local": 4,
            "racha_visita": 2,
            "clima": 1,  # 1 = buen clima
            "importancia_partido": 3,
            "fecha_actualizacion": datetime.utcnow().isoformat()
        }
    ]

def actualizar_datos_partidos():
    nuevos_datos = obtener_datos_actualizados()
    for partido in nuevos_datos:
        supabase.table("partidos").upsert(partido, on_conflict=["nombre_partido"]).execute()

    return {"status": "Datos actualizados correctamente"}