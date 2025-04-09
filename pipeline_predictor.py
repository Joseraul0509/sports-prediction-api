import os
from datetime import datetime
from supabase import create_client
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Conexión a Supabase
url = os.getenv("SUPABASE_URL")
key = os.getenv("SUPABASE_KEY")
supabase = create_client(url, key)

def obtener_datos_entrenamiento():
    # Simula una tabla de entrenamiento. Reemplaza por tu propia lógica si ya tienes datos reales
    return pd.DataFrame([
        {"goles_local_prom": 1.5, "goles_visita_prom": 1.0, "racha_local": 3, "racha_visita": 2, "clima": 1, "importancia_partido": 2, "resultado": 1},
        {"goles_local_prom": 0.9, "goles_visita_prom": 1.7, "racha_local": 1, "racha_visita": 4, "clima": 1, "importancia_partido": 3, "resultado": 2},
        {"goles_local_prom": 2.1, "goles_visita_prom": 0.8, "racha_local": 5, "racha_visita": 1, "clima": 1, "importancia_partido": 1, "resultado": 1},
    ])

def entrenar_modelo(df):
    X = df.drop("resultado", axis=1)
    y = df["resultado"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("Precisión del modelo:", accuracy_score(y_test, y_pred))

    return model

def predecir_resultado(modelo, datos_partido):
    df = pd.DataFrame([datos_partido])
    prediccion = modelo.predict(df)[0]

    if prediccion == 1:
        return "Gana Local", 0.75
    elif prediccion == 2:
        return "Gana Visitante", 0.72
    else:
        return "Empate", 0.68

def procesar_partidos():
    # Entrenar el modelo con tus datos (simulado por ahora)
    datos_entrenamiento = obtener_datos_entrenamiento()
    modelo = entrenar_modelo(datos_entrenamiento)

    # Obtener datos actuales para predecir
    partidos = supabase.table("partidos").select("*").execute().data

    for partido in partidos:
        resultado, confianza = predecir_resultado(modelo, partido)

        prediccion = {
            "deporte": "futbol",
            "liga": "Simulada",
            "partido": partido["nombre_partido"],
            "hora": datetime.utcnow().isoformat(),
            "pronostico_1": resultado,
            "confianza_1": confianza,
            "pronostico_2": "Menos de 2.5 goles",
            "confianza_2": 0.70,
            "pronostico_3": "Ambos anotan",
            "confianza_3": 0.65
        }

        supabase.table("predicciones").upsert(prediccion, on_conflict=["partido", "hora"]).execute()

    return {"message": "Predicciones generadas correctamente"}
