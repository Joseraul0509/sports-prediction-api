import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 1. Función para obtener datos simulados de entrenamiento
def obtener_datos_entrenamiento():
    # Clases: 0 = Gana Local, 1 = Empate, 2 = Gana Visitante
    return pd.DataFrame([
        {"goles_local_prom": 1.5, "goles_visita_prom": 1.0, "racha_local": 3, "racha_visita": 2, "clima": 1, "importancia_partido": 2, "resultado": 0},
        {"goles_local_prom": 1.2, "goles_visita_prom": 1.2, "racha_local": 2, "racha_visita": 2, "clima": 1, "importancia_partido": 2, "resultado": 1},
        {"goles_local_prom": 0.9, "goles_visita_prom": 1.7, "racha_local": 1, "racha_visita": 4, "clima": 2, "importancia_partido": 3, "resultado": 2},
        {"goles_local_prom": 2.1, "goles_visita_prom": 0.8, "racha_local": 5, "racha_visita": 1, "clima": 1, "importancia_partido": 1, "resultado": 0},
        {"goles_local_prom": 1.0, "goles_visita_prom": 1.0, "racha_local": 2, "racha_visita": 2, "clima": 1, "importancia_partido": 2, "resultado": 1},
        {"goles_local_prom": 1.3, "goles_visita_prom": 1.6, "racha_local": 2, "racha_visita": 3, "clima": 2, "importancia_partido": 2, "resultado": 2}
    ])

# 2. Función para entrenar el modelo
def entrenar_modelo():
    datos = obtener_datos_entrenamiento()
    X = datos.drop("resultado", axis=1)
    y = datos["resultado"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    modelo = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', num_class=3)
    modelo.fit(X_train, y_train)

    y_pred = modelo.predict(X_test)
    precision = accuracy_score(y_test, y_pred)
    print(f"Precisión del modelo: {precision:.2f}")

    return modelo

# 3. Función para predecir el resultado de un partido
def predecir_resultado(modelo, datos_partido):
    df = pd.DataFrame([datos_partido])
    pred = modelo.predict(df)[0]

    resultados = {
        0: ("Gana Local", 0.70),
        1: ("Empate", 0.65),
        2: ("Gana Visitante", 0.68)
    }

    texto_resultado, probabilidad = resultados.get(pred, ("Resultado desconocido", 0.0))
    return texto_resultado, probabilidad
