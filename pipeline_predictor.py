import os
import uuid
from datetime import datetime
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import requests
from supabase import create_client, Client
import random

# Cargar variables de entorno (asegúrate de tener un archivo .env configurado)
from dotenv import load_dotenv
load_dotenv()

# Conexión a Supabase (usando variables de entorno definidas en .env)
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Claves API
# API-Sports se utiliza para Fútbol, MLB, NHL y como fuente principal para NBA
api_sports_key = os.getenv("API_SPORTS_KEY")
# Para Football Data API se omite, pues ya retorna 403 y se obtiene info por API-Sports.
#
# Para NBA, además se integran las URL de la API de balldontlie
# URL para juegos del día, equipos y estadísticas:
balldontlie_base_url = "https://api.balldontlie.io/v1"
# Clave API para balldontlie (en este caso se integra en el código ya que la API es pública)
balldontlie_api_key = "5293380d-9c4f-4e89-be15-bf19b1042182"

# ===============================
# NUEVAS FUNCIONES DE INSERCIÓN
# ===============================

def insertar_league(nombre_liga: str, _deporte: str):
    try:
        if nombre_liga.lower() == "desconocida":
            print("Liga no insertada: valor 'Desconocida'")
            return
        existe = supabase.table("leagues").select("*").match({"name": nombre_liga}).execute()
        if not existe.data:
            supabase.table("leagues").insert({
                "id": str(uuid.uuid4()),
                "name": nombre_liga,
                "country": "Unknown",
                "flag": None,
            }).execute()
            print(f"Liga insertada: {nombre_liga}")
    except Exception as e:
        print(f"Error al insertar liga {nombre_liga}: {str(e)}")

def insertar_partido(nombre_partido: str, liga: str, hora: str):
    try:
        if nombre_partido.lower() == "partido desconocido":
            print("Partido no insertado: valor 'Partido Desconocido'")
            return
        existe = supabase.table("partidos").select("*").match({
            "nombre_partido": nombre_partido,
            "liga": liga,
            "hora": hora
        }).execute()
        if not existe.data:
            supabase.table("partidos").insert({
                "nombre_partido": nombre_partido,
                "liga": liga,
                "hora": hora
            }).execute()
            print(f"Partido insertado: {nombre_partido}")
    except Exception as e:
        print(f"Error al insertar partido {nombre_partido}: {str(e)}")

# ===============================
# FIN DE NUEVAS FUNCIONES
# ===============================

# CONFIGURACIÓN DE APIs DEPORTIVAS

# Para Fútbol:
# Se utiliza API-Sports (v3) como fuente principal, complementada por OpenLigaDB.
OPENLIGADB_ENDPOINTS = [
    "https://api.openligadb.de/getmatchdata/dfb/2024/5",
    "https://api.openligadb.de/getmatchdata/bl2/2024/28",
    "https://api.openligadb.de/getmatchdata/bl3/2024/31",
    "https://api.openligadb.de/getmatchdata/ucl24/2024/12",
    "https://api.openligadb.de/getmatchdata/bl1/2024/28",
    "https://api.openligadb.de/getmatchdata/ucl2024/2024/4",
    "https://api.openligadb.de/getmatchdata/uel24/2024/12"
]
# Football Data API se omite

# Para NBA:
# Fuente principal: API-Sports (se intenta v2 y, si falla, se usa URL alternativa v1)
# Complementada con la API de balldontlie usando las URL:
#    - Juegos: /v1/games
#    - Equipos: /v1/teams
#    - Estadísticas: /v1/stats

def get_today():
    return datetime.utcnow().strftime("%Y-%m-%d")

# -------------------------------
# Funciones para Fútbol
# -------------------------------
def obtener_datos_futbol():
    datos = []
    today = get_today()
    try:
        url_api = f"https://v3.football.api-sports.io/fixtures?date={today}"
        headers_api = {"x-apisports-key": api_sports_key}
        resp = requests.get(url_api, headers=headers_api, timeout=10)
        if resp.status_code == 200:
            respuesta = resp.json().get("response", [])
            if not isinstance(respuesta, list):
                print("Respuesta inesperada en API-Sports fútbol:", resp.json())
            else:
                for fixture in respuesta:
                    info = fixture.get("fixture", {})
                    teams = fixture.get("teams", {})
                    datos.append({
                        "nombre_partido": teams.get("home", {}).get("name", "Partido Desconocido") + " vs " +
                                          teams.get("away", {}).get("name", ""),
                        "liga": fixture.get("league", {}).get("name", "Desconocida"),
                        "deporte": "futbol",
                        "goles_local_prom": fixture.get("goals", {}).get("home", 1.5),
                        "goles_visita_prom": fixture.get("goals", {}).get("away", 1.2),
                        "racha_local": 3,
                        "racha_visita": 2,
                        "clima": 1,
                        "importancia_partido": 3,
                        "hora": info.get("date", datetime.utcnow().isoformat())
                    })
        else:
            print("Error en API-Sports fútbol:", resp.status_code, resp.text)
    except Exception as e:
        print("Excepción en API-Sports fútbol:", e)
    # Complemento con OpenLigaDB
    for endpoint in OPENLIGADB_ENDPOINTS:
        try:
            resp = requests.get(endpoint, timeout=10)
            if resp.status_code == 200:
                for partido in resp.json():
                    datos.append({
                        "nombre_partido": partido.get("MatchName", "Partido Desconocido"),
                        "liga": partido.get("League", "Desconocida"),
                        "deporte": "futbol",
                        "goles_local_prom": partido.get("MatchResults", [{}])[0].get("PointsTeam1", 1.5),
                        "goles_visita_prom": partido.get("MatchResults", [{}])[0].get("PointsTeam2", 1.2),
                        "racha_local": 3,
                        "racha_visita": 2,
                        "clima": partido.get("clima", 1),
                        "importancia_partido": partido.get("importancia_partido", 3),
                        "hora": partido.get("MatchDateTime", datetime.utcnow().isoformat())
                    })
        except Exception as e:
            print("Error en OpenLigaDB:", e)
    return datos

# -------------------------------
# Funciones para NBA
# -------------------------------
def obtener_datos_nba():
    datos = []
    today = get_today()
    success = False
    # Fuente principal: API-Sports NBA v2
    try:
        url_api = f"https://v2.nba.api-sports.io/games?date={today}"
        headers_api = {"x-apisports-key": api_sports_key}
        resp = requests.get(url_api, headers=headers_api, timeout=10)
        if resp.status_code == 200:
            respuesta = resp.json().get("response", [])
            if isinstance(respuesta, list):
                for game in respuesta:
                    if not isinstance(game, dict):
                        print("Elemento inesperado en API-Sports NBA (v2):", game)
                        continue
                    equipos = game.get("teams", {})
                    score_home = game.get("scores", {}).get("home", 100)
                    if isinstance(score_home, dict):
                        score_home = score_home.get("points", 100)
                    score_away = game.get("scores", {}).get("away", 100)
                    if isinstance(score_away, dict):
                        score_away = score_away.get("points", 100)
                    datos.append({
                        "nombre_partido": equipos.get("home", {}).get("name", "Partido Desconocido") + " vs " +
                                          equipos.get("away", {}).get("name", ""),
                        "liga": game.get("league", {}).get("name", "NBA"),
                        "deporte": "NBA",
                        "goles_local_prom": float(score_home) / 10,
                        "goles_visita_prom": float(score_away) / 10,
                        "racha_local": 3,
                        "racha_visita": 2,
                        "clima": 1,
                        "importancia_partido": 3,
                        "hora": game.get("date", datetime.utcnow().isoformat())
                    })
                success = True
            else:
                print("Respuesta inesperada en API-Sports NBA (v2):", resp.json())
        else:
            print("Error en API-Sports NBA (v2):", resp.status_code, resp.text)
    except Exception as e:
        print("Excepción en API-Sports NBA (v2):", e)

    # Intentar URL alternativa (v1) si v2 falló
    if not success:
        try:
            url_api = f"https://v1.basketball.api-sports.io/games?date={today}"
            headers_api = {"x-apisports-key": api_sports_key}
            resp = requests.get(url_api, headers=headers_api, timeout=10)
            if resp.status_code == 200:
                respuesta = resp.json().get("response", [])
                if isinstance(respuesta, list):
                    for game in respuesta:
                        if not isinstance(game, dict):
                            print("Elemento inesperado en API-Sports NBA (v1):", game)
                            continue
                        equipos = game.get("teams", {})
                        score_home = game.get("scores", {}).get("home", 100)
                        if isinstance(score_home, dict):
                            score_home = score_home.get("points", 100)
                        score_away = game.get("scores", {}).get("away", 100)
                        if isinstance(score_away, dict):
                            score_away = score_away.get("points", 100)
                        datos.append({
                            "nombre_partido": equipos.get("home", {}).get("name", "Partido Desconocido") + " vs " +
                                              equipos.get("away", {}).get("name", ""),
                            "liga": game.get("league", {}).get("name", "NBA"),
                            "deporte": "NBA",
                            "goles_local_prom": float(score_home) / 10,
                            "goles_visita_prom": float(score_away) / 10,
                            "racha_local": 3,
                            "racha_visita": 2,
                            "clima": 1,
                            "importancia_partido": 3,
                            "hora": game.get("date", datetime.utcnow().isoformat())
                        })
                    success = True
                else:
                    print("Respuesta inesperada en API-Sports NBA (v1):", resp.json())
            else:
                print("Error en API-Sports NBA (v1):", resp.status_code, resp.text)
        except Exception as e:
            print("Excepción en API-Sports NBA (v1):", e)
    # Complemento: Usar la API de balldontlie para NBA
    try:
        url_bd = f"{balldontlie_base_url}/games"
        params = {"start_date": today, "end_date": today, "per_page": 100}
        resp_bd = requests.get(url_bd, params=params, timeout=10)
        if resp_bd.status_code == 200:
            games = resp_bd.json().get("data", [])
            # Obtener estadísticas para estos juegos
            url_stats = f"{balldontlie_base_url}/stats"
            params_stats = {"start_date": today, "end_date": today, "per_page": 100}
            resp_stats = requests.get(url_stats, params=params_stats, timeout=10)
            stats = []
            if resp_stats.status_code == 200:
                stats = resp_stats.json().get("data", [])
            else:
                print("Error en balldontlie (NBA - stats):", resp_stats.status_code, resp_stats.text)
            for game in games:
                try:
                    game_id = game.get("id")
                    game_stats = [s for s in stats if s.get("game", {}).get("id") == game_id]
                    datos.append({
                        "nombre_partido": game["home_team"]["full_name"] + " vs " + game["visitor_team"]["full_name"],
                        "liga": "NBA",
                        "deporte": "NBA",
                        "goles_local_prom": float(game.get("home_team_score", 0)) / 10,
                        "goles_visita_prom": float(game.get("visitor_team_score", 0)) / 10,
                        "racha_local": 3,
                        "racha_visita": 2,
                        "clima": 1,
                        "importancia_partido": 3,
                        "hora": game.get("date", datetime.utcnow().isoformat()),
                        "stats": game_stats
                    })
                except Exception as ex:
                    print("Error procesando juego en balldontlie (NBA):", ex)
        else:
            print("Error en balldontlie (NBA - games):", resp_bd.status_code, resp_bd.text)
    except Exception as e:
        print("Excepción en balldontlie (NBA - games):", e)
    return datos

# -------------------------------
# Funciones para MLB y NHL (usando API-Sports)
# -------------------------------
def obtener_datos_deporte_api(deporte):
    datos = []
    today = get_today()
    if deporte.upper() == "MLB":
        ENDPOINT = "https://v1.baseball.api-sports.io/games"
    elif deporte.upper() == "NHL":
        ENDPOINT = "https://v1.hockey.api-sports.io/games"
    else:
        return datos
    params = {"date": today}
    headers_api = {"x-apisports-key": api_sports_key}
    try:
        resp = requests.get(ENDPOINT, headers=headers_api, params=params, timeout=10)
        if resp.status_code == 200:
            for game in resp.json().get("response", []):
                if deporte.upper() == "MLB":
                    datos.append({
                        "nombre_partido": game.get("teams", {}).get("home", {}).get("name", "Equipo Local") +
                                          " vs " +
                                          game.get("teams", {}).get("away", {}).get("name", "Equipo Visitante"),
                        "liga": game.get("league", {}).get("name", "Desconocida"),
                        "deporte": "MLB",
                        "goles_local_prom": game.get("scores", {}).get("home", 0.0),
                        "goles_visita_prom": game.get("scores", {}).get("away", 0.0),
                        "racha_local": 0,
                        "racha_visita": 0,
                        "clima": 1,
                        "importancia_partido": 3,
                        "hora": game.get("date", datetime.utcnow().isoformat())
                    })
                elif deporte.upper() == "NHL":
                    datos.append({
                        "nombre_partido": game.get("teams", {}).get("home", {}).get("name", "Equipo Local") +
                                          " vs " +
                                          game.get("teams", {}).get("away", {}).get("name", "Equipo Visitante"),
                        "liga": game.get("league", {}).get("name", "Desconocida"),
                        "deporte": "NHL",
                        "goles_local_prom": game.get("scores", {}).get("home", 0.0),
                        "goles_visita_prom": game.get("scores", {}).get("away", 0.0),
                        "racha_local": 0,
                        "racha_visita": 0,
                        "clima": 1,
                        "importancia_partido": 3,
                        "hora": game.get("date", datetime.utcnow().isoformat())
                    })
        else:
            print(f"Error en API-Sports {deporte}:", resp.status_code, resp.text)
    except Exception as e:
        print(f"Excepción en API-Sports {deporte}:", e)
    return datos

def obtener_datos_actualizados():
    datos = []
    datos.extend(obtener_datos_futbol())
    datos.extend(obtener_datos_nba())
    for dep in ["MLB", "NHL"]:
        datos.extend(obtener_datos_deporte_api(dep))
    return datos if datos else [{
        "nombre_partido": "Arsenal vs Chelsea",
        "liga": "Premier League",
        "deporte": "futbol",
        "goles_local_prom": 1.8,
        "goles_visita_prom": 1.2,
        "racha_local": 4,
        "racha_visita": 2,
        "clima": 1,
        "importancia_partido": 3,
        "hora": datetime.utcnow().isoformat()
    }]

# -------------------------------
# Funciones para Entrenamiento y Predicción
# -------------------------------
def obtener_datos_entrenamiento():
    # Para entrenar, combinamos los datos obtenidos de los partidos actuales de todos los deportes.
    # Para aumentar la cantidad de datos y diversidad, duplicamos la información varias veces.
    datos_actuales = obtener_datos_actualizados()
    simulated_data = datos_actuales * 10  # Aumenta la cantidad de registros
    for d in simulated_data:
        # Simulamos el resultado de cada partido (0 o 1, por ejemplo, 1 = victoria del local)
        d["resultado"] = random.choice([0, 1])
    # Extraemos únicamente las variables numéricas necesarias para entrenar.
    training_rows = []
    for d in simulated_data:
        try:
            row = {
                "goles_local_prom": float(d.get("goles_local_prom", 1.5)),
                "goles_visita_prom": float(d.get("goles_visita_prom", 1.2)),
                "racha_local": int(d.get("racha_local", 3)),
                "racha_visita": int(d.get("racha_visita", 2)),
                "clima": int(d.get("clima", 1)),
                "importancia_partido": int(d.get("importancia_partido", 3)),
                "resultado": int(d["resultado"])
            }
            training_rows.append(row)
        except Exception as e:
            print("Error construyendo datos de entrenamiento:", e)
    return pd.DataFrame(training_rows)

def entrenar_modelo():
    df = obtener_datos_entrenamiento()
    df = df.astype({
        "goles_local_prom": float,
        "goles_visita_prom": float,
        "racha_local": int,
        "racha_visita": int,
        "clima": int,
        "importancia_partido": int,
        "resultado": int
    })
    X = df.drop("resultado", axis=1)
    y = df["resultado"]
    if len(y.unique()) < 2:
        print("Datos insuficientes para diferenciar clases; usando todo el dataset para entrenamiento.")
        model = xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss")
        model.fit(X, y)
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
        model = xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss")
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = (y_test == preds).mean()
        print(f"Precisión del modelo: {acc:.2f}")
    return model

def predecir_resultado(modelo, datos_partido):
    df = pd.DataFrame([{
        "goles_local_prom": datos_partido["goles_local_prom"],
        "goles_visita_prom": datos_partido["goles_visita_prom"],
        "racha_local": datos_partido["racha_local"],
        "racha_visita": datos_partido["racha_visita"],
        "clima": int(datos_partido["clima"]),
        "importancia_partido": datos_partido["importancia_partido"]
    }])
    pred = modelo.predict(df)[0]
    opciones = {
        0: ("Gana Local", 0.70),
        1: ("Empate", 0.65),
        2: ("Gana Visitante", 0.68)
    }
    return opciones.get(pred, ("Indefinido", 0.0))

def manual_upsert(table_name: str, data: dict, conflict_columns: list):
    query = supabase.table(table_name).select("*")
    for col in conflict_columns:
        query = query.eq(col, data[col])
    result = query.execute()
    if result.data and len(result.data) > 0:
        update_query = supabase.table(table_name).update(data)
        for col in conflict_columns:
            update_query = update_query.eq(col, data[col])
        update_query.execute()
        return "updated"
    else:
        supabase.table(table_name).insert(data).execute()
        return "inserted"

def actualizar_datos_partidos():
    nuevos_datos = obtener_datos_actualizados()
    for partido in nuevos_datos:
        hora_valor = partido["hora"]
        if isinstance(hora_valor, str):
            try:
                hora_valor = datetime.fromisoformat(hora_valor)
            except Exception as ex:
                print(f"Error al convertir hora: {ex}")
                hora_valor = datetime.utcnow()
        partido["hora"] = hora_valor.strftime("%Y-%m-%d %H:%M:%S")
        insertar_league(partido["liga"], partido["deporte"])
        insertar_partido(partido["nombre_partido"], partido["liga"], partido["hora"])
        result = manual_upsert("partidos", partido, ["nombre_partido", "hora"])
        print(f"Registro en partidos {partido['nombre_partido']}: {result}")
    return {"status": "Datos actualizados correctamente"}

def procesar_predicciones():
    model = entrenar_modelo()
    partidos = obtener_datos_actualizados()
    for partido in partidos:
        hora_valor = partido["hora"]
        if isinstance(hora_valor, str):
            try:
                hora_dt = datetime.fromisoformat(hora_valor)
            except Exception as ex:
                print(f"Error al convertir hora en predicción: {ex}")
                hora_dt = datetime.utcnow()
        else:
            hora_dt = hora_valor
        partido["hora"] = hora_dt.strftime("%Y-%m-%d %H:%M:%S")
        resultado, confianza = predecir_resultado(model, partido)
        prediccion = {
            "deporte": partido["deporte"],
            "liga": partido["liga"],
            "partido": partido["nombre_partido"],
            "hora": partido["hora"],
            "pronostico_1": resultado,
            "confianza_1": confianza,
            "pronostico_2": "Menos de 2.5 goles",
            "confianza_2": 0.70,
            "pronostico_3": "Ambos no anotan",
            "confianza_3": 0.65
        }
        result = manual_upsert("predicciones", prediccion, ["partido", "hora"])
        print(f"Registro en predicciones {prediccion['partido']}: {result}")
    return {"message": "Predicciones generadas correctamente"}

# -------------------------------
# Función para integrar datos de la NBA desde balldontlie
# -------------------------------
def insertar_nba_balldontlie():
    # Insertar equipos de la NBA
    try:
        response = requests.get(f"{balldontlie_base_url}/teams", timeout=10)
        if response.status_code == 200:
            equipos = response.json().get("data", [])
            for equipo in equipos:
                # Se inserta el equipo en la tabla "teams" (crea la tabla si es necesario o utiliza otra lógica)
                supabase.table("teams").upsert({
                    "id": equipo.get("id"),
                    "name": equipo.get("full_name"),
                    "abbreviation": equipo.get("abbreviation"),
                    "conference": equipo.get("conference")
                }, on_conflict=["id"]).execute()
        else:
            print("Error obteniendo equipos de balldontlie:", response.status_code, response.text)
    except Exception as e:
        print("Excepción al obtener equipos de balldontlie:", e)
    
    # Insertar juegos del día
    try:
        today = get_today()
        response = requests.get(f"{balldontlie_base_url}/games?start_date={today}&end_date={today}", timeout=10)
        if response.status_code == 200:
            juegos = response.json().get("data", [])
            for juego in juegos:
                data_game = {
                    "nombre_partido": juego["home_team"]["full_name"] + " vs " + juego["visitor_team"]["full_name"],
                    "liga": "NBA",
                    "deporte": "NBA",
                    "goles_local_prom": float(juego.get("home_team_score", 0)) / 10,
                    "goles_visita_prom": float(juego.get("visitor_team_score", 0)) / 10,
                    "racha_local": 3,
                    "racha_visita": 2,
                    "clima": 1,
                    "importancia_partido": 3,
                    "hora": juego.get("date", datetime.utcnow().isoformat())
                }
                supabase.table("partidos").upsert(data_game, on_conflict=["nombre_partido", "hora"]).execute()
        else:
            print("Error en juegos de balldontlie:", response.status_code, response.text)
    except Exception as e:
        print("Excepción en juegos de balldontlie:", e)
    
    # Insertar estadísticas de juegos del día como información complementaria (en la tabla predicciones)
    try:
        response = requests.get(f"{balldontlie_base_url}/stats?start_date={today}&end_date={today}&per_page=100", timeout=10)
        if response.status_code == 200:
            stats = response.json().get("data", [])
            for stat in stats:
                data_stat = {
                    "deporte": "NBA",
                    "liga": "NBA",
                    "partido": stat.get("game", {}).get("id", "Sin partido"),
                    "hora": datetime.utcnow().isoformat(),
                    "jugador": stat.get("player", {}).get("first_name", "") + " " + stat.get("player", {}).get("last_name", ""),
                    "pronostico_1": f"{stat.get('pts', 0)} pts",
                    "confianza_1": 0.65,
                    "pronostico_2": f"{stat.get('reb', 0)} rebotes",
                    "confianza_2": 0.60,
                    "pronostico_3": f"{stat.get('ast', 0)} asistencias",
                    "confianza_3": 0.60
                }
                supabase.table("predicciones").insert(data_stat).execute()
        else:
            print("Error en stats de balldontlie:", response.status_code, response.text)
    except Exception as e:
        print("Excepción en stats de balldontlie:", e)

# -------------------------------
# Función principal del pipeline
# -------------------------------
def ejecutar_pipeline():
    # Actualizar partidos y datos de todas las fuentes
    actualizar_datos_partidos()
    # Integrar los datos de la NBA desde balldontlie
    insertar_nba_balldontlie()
    # Procesar predicciones usando el modelo entrenado sobre datos de todos los deportes
    procesar_predicciones()

if __name__ == "__main__":
    ejecutar_pipeline()
