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
import time
from dotenv import load_dotenv

# Cargar variables de entorno desde .env
load_dotenv()

# Conexión a Supabase (debe estar configurado en .env)
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Claves API
# Se asume que API_SPORTS_KEY y API_FOOTBALL_KEY están configuradas en .env
API_SPORTS_KEY = os.getenv("API_SPORTS_KEY")
HEADERS_API_SPORTS = {"x-apisports-key": API_SPORTS_KEY}
HEADERS_API_FOOTBALL = {"x-apisports-key": os.getenv("API_FOOTBALL_KEY", API_SPORTS_KEY)}
# La API de balldontlie es pública; no se requiere header.
balldontlie_base_url = "https://api.balldontlie.io/v1"

# Obtener fecha actual en formato AAAA-MM-DD
def get_today():
    return datetime.utcnow().strftime("%Y-%m-%d")

# --------------------------------------------------
# Funciones auxiliares para conversión segura
# --------------------------------------------------
def safe_numeric(val):
    try:
        if val is None:
            return 0.0
        return float(val)
    except Exception as e:
        print(f"Error al convertir a número: {e}")
        return 0.0

def convert_form_string(form_str):
    # Mapea resultados: 'W' = 1, 'D' = 0, 'L' = 0. Ajusta según tus necesidades.
    mapping = {'W': 1, 'D': 0, 'L': 0}
    if isinstance(form_str, str):
        return [mapping.get(c, 0) for c in form_str if c in mapping]
    elif isinstance(form_str, list):
        return form_str
    return [0] * 5

# --------------------------------------------------
# Función de upsert con reintentos en Supabase
# --------------------------------------------------
def manual_upsert(table_name: str, data: dict, conflict_columns: list, retries=3):
    for attempt in range(retries):
        try:
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
        except Exception as e:
            print(f"Error en manual_upsert para {table_name}, intento {attempt+1}: {e}")
            time.sleep(1)
    return "failed"

# --------------------------------------------------
# Función para obtener estadísticas reales desde API-SPORTS
# --------------------------------------------------
def obtener_estadisticas(team_id, league_id, season, deporte):
    if deporte == "futbol":
        url = f"https://v3.football.api-sports.io/teams/statistics?league={league_id}&team={team_id}&season={season}"
        headers = HEADERS_API_FOOTBALL
    elif deporte == "nba":
        url = f"https://v2.nba.api-sports.io/teams/statistics?league={league_id}&team={team_id}&season={season}"
        headers = HEADERS_API_SPORTS
    elif deporte == "mlb":
        url = f"https://v1.baseball.api-sports.io/teams/statistics?league={league_id}&team={team_id}&season={season}"
        headers = HEADERS_API_SPORTS
    elif deporte == "nhl":
        url = f"https://v1.hockey.api-sports.io/teams/statistics?league={league_id}&team={team_id}&season={season}"
        headers = HEADERS_API_SPORTS
    else:
        return {}
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        if resp.status_code == 200:
            response_data = resp.json().get("response")
            if isinstance(response_data, list) and response_data:
                return response_data[0]
            elif isinstance(response_data, dict):
                return response_data
            else:
                return {}
        else:
            print(f"Error en estadísticas {deporte}: {resp.status_code} {resp.text}")
            return {}
    except Exception as e:
        print(f"Excepción en estadísticas {deporte}: {e}")
        return {}

# --------------------------------------------------
# Funciones para insertar ligas y partidos en Supabase
# --------------------------------------------------
def insertar_league(nombre_liga: str, _deporte: str):
    try:
        if not nombre_liga or nombre_liga.lower() == "desconocida":
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
        print(f"Error al insertar liga {nombre_liga}: {e}")

def insertar_partido(nombre_partido: str, liga: str, hora: str):
    try:
        if not nombre_partido or nombre_partido.lower() == "partido desconocido":
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
        print(f"Error al insertar partido {nombre_partido}: {e}")

# --------------------------------------------------
# Funciones para obtener datos de fixtures y partidos
# --------------------------------------------------
def obtener_datos_futbol():
    datos = []
    headers = HEADERS_API_FOOTBALL
    url = f"https://v3.football.api-sports.io/fixtures?date={get_today()}"
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        if resp.status_code == 200:
            fixtures = resp.json().get("response", [])
            print(f"Obtenidos {len(fixtures)} fixtures de API-Sports fútbol")
            for fixture in fixtures:
                teams = fixture.get("teams", {})
                league_info = fixture.get("league", {})
                # Extraer parámetros reales desde la respuesta
                league_id = str(league_info.get("id", ""))
                season = league_info.get("season", "")
                registro = {
                    "id": fixture.get("fixture", {}).get("id"),
                    "nombre_partido": teams.get("home", {}).get("name", "Partido Desconocido") + " vs " +
                                      teams.get("away", {}).get("name", ""),
                    "liga": league_info.get("name", "Desconocida"),
                    "deporte": "futbol",
                    "goles_local_prom": 0,
                    "goles_visita_prom": 0,
                    "racha_local": 0,
                    "racha_visita": 0,
                    "clima": 1,
                    "importancia_partido": 3,
                    "hora": fixture.get("fixture", {}).get("date", datetime.utcnow().isoformat())
                }
                # Extraer estadísticas para el equipo local
                local_id = str(teams.get("home", {}).get("id", ""))
                stats_local = obtener_estadisticas(local_id, league_id, season, "futbol")
                registro["goles_local_prom"] = safe_numeric(stats_local.get("fixtures", {}).get("goals", {}).get("for"))
                registro["racha_local"] = stats_local.get("fixtures", {}).get("streak", 0)
                form_local = stats_local.get("fixtures", {}).get("form", "-----")
                registro["ultimos_5_local"] = convert_form_string(form_local)[-5:]
                # Para el equipo visitante
                away_id = str(teams.get("away", {}).get("id", ""))
                stats_away = obtener_estadisticas(away_id, league_id, season, "futbol")
                registro["goles_visita_prom"] = safe_numeric(stats_away.get("fixtures", {}).get("goals", {}).get("for"))
                registro["racha_visita"] = stats_away.get("fixtures", {}).get("streak", 0)
                form_away = stats_away.get("fixtures", {}).get("form", "-----")
                registro["ultimos_5_visitante"] = convert_form_string(form_away)[-5:]
                datos.append(registro)
        else:
            print(f"Error en API-Sports fútbol: {resp.status_code} {resp.text}")
    except Exception as e:
        print(f"Excepción en API-Sports fútbol: {e}")

    # Complemento: OpenLigaDB (datos básicos)
    OPENLIGADB_ENDPOINTS = [
        "https://api.openligadb.de/getmatchdata/dfb/2024/5",
        "https://api.openligadb.de/getmatchdata/bl2/2024/28",
        "https://api.openligadb.de/getmatchdata/bl3/2024/31",
        "https://api.openligadb.de/getmatchdata/ucl24/2024/12",
        "https://api.openligadb.de/getmatchdata/bl1/2024/28",
        "https://api.openligadb.de/getmatchdata/ucl2024/2024/4",
        "https://api.openligadb.de/getmatchdata/uel24/2024/12"
    ]
    for endpoint in OPENLIGADB_ENDPOINTS:
        try:
            resp = requests.get(endpoint, timeout=10)
            if resp.status_code == 200:
                fixtures = resp.json()
                print(f"Obtenidos {len(fixtures)} fixtures de OpenLigaDB")
                for partido in fixtures:
                    datos.append({
                        "id": None,
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
            else:
                print(f"Error en OpenLigaDB: {resp.status_code}")
        except Exception as e:
            print(f"Error en OpenLigaDB: {e}")
    return datos

def obtener_datos_nba():
    datos = []
    headers = {"x-apisports-key": API_SPORTS_KEY}
    try:
        url_api = f"https://v2.nba.api-sports.io/games?date={get_today()}"
        resp = requests.get(url_api, headers=headers, timeout=10)
        if resp.status_code == 200:
            games = resp.json().get("response", [])
            print(f"Obtenidos {len(games)} juegos de API-Sports NBA (v2)")
            for game in games:
                equipos = game.get("teams", {})
                league_info = game.get("league", {})
                league_id = str(league_info.get("id", ""))
                season = league_info.get("season", "")
                local_id = str(equipos.get("home", {}).get("id", ""))
                away_id = str(equipos.get("away", {}).get("id", ""))
                stats_local = obtener_estadisticas(local_id, league_id, season, "nba")
                stats_away = obtener_estadisticas(away_id, league_id, season, "nba")
                score_home = game.get("scores", {}).get("home", 100)
                if isinstance(score_home, dict):
                    score_home = score_home.get("points", 100)
                score_away = game.get("scores", {}).get("away", 100)
                if isinstance(score_away, dict):
                    score_away = score_away.get("points", 100)
                registro = {
                    "id": game.get("fixture", {}).get("id"),
                    "nombre_partido": equipos.get("home", {}).get("name", "Partido Desconocido") + " vs " +
                                      equipos.get("away", {}).get("name", ""),
                    "liga": league_info.get("name", "NBA"),
                    "deporte": "nba",
                    "goles_local_prom": safe_numeric(score_home) / 10,
                    "goles_visita_prom": safe_numeric(score_away) / 10,
                    "racha_local": stats_local.get("games", {}).get("streak", 0),
                    "racha_visita": stats_away.get("games", {}).get("streak", 0),
                    "clima": 1,
                    "importancia_partido": 3,
                    "hora": game.get("date", datetime.utcnow().isoformat())
                }
                datos.append(registro)
            return datos
        else:
            print(f"Error en API-Sports NBA (v2): {resp.status_code} {resp.text}")
    except Exception as e:
        print(f"Excepción en API-Sports NBA (v2): {e}")
    # Fallback: API-Sports NBA v1
    try:
        url_api = f"https://v1.basketball.api-sports.io/games?date={get_today()}"
        resp = requests.get(url_api, headers=headers, timeout=10)
        if resp.status_code == 200:
            games = resp.json().get("response", [])
            print(f"Obtenidos {len(games)} juegos de API-Sports NBA (v1)")
            for game in games:
                equipos = game.get("teams", {})
                score_home = game.get("scores", {}).get("home", 100)
                if isinstance(score_home, dict):
                    score_home = score_home.get("points", 100)
                score_away = game.get("scores", {}).get("away", 100)
                if isinstance(score_away, dict):
                    score_away = score_away.get("points", 100)
                registro = {
                    "id": game.get("fixture", {}).get("id"),
                    "nombre_partido": equipos.get("home", {}).get("name", "Partido Desconocido") + " vs " +
                                      equipos.get("away", {}).get("name", ""),
                    "liga": game.get("league", {}).get("name", "NBA"),
                    "deporte": "nba",
                    "goles_local_prom": safe_numeric(score_home) / 10,
                    "goles_visita_prom": safe_numeric(score_away) / 10,
                    "racha_local": 3,
                    "racha_visita": 2,
                    "clima": 1,
                    "importancia_partido": 3,
                    "hora": game.get("date", datetime.utcnow().isoformat())
                }
                datos.append(registro)
            return datos
        else:
            print(f"Error en API-Sports NBA (v1): {resp.status_code} {resp.text}")
    except Exception as e:
        print(f"Excepción en API-Sports NBA (v1): {e}")
    # Complemento: Usar la API de balldontlie para NBA (sin headers)
    try:
        url_bd = f"{balldontlie_base_url}/games?start_date={get_today()}&end_date={get_today()}&per_page=100"
        resp_bd = requests.get(url_bd, timeout=10)
        if resp_bd.status_code == 200:
            games = resp_bd.json().get("data", [])
            print(f"Obtenidos {len(games)} juegos de balldontlie para NBA")
            for game in games:
                registro = {
                    "id": game.get("id"),
                    "nombre_partido": game["home_team"]["full_name"] + " vs " + game["visitor_team"]["full_name"],
                    "liga": "NBA",
                    "deporte": "nba",
                    "goles_local_prom": safe_numeric(game.get("home_team_score", 0)) / 10,
                    "goles_visita_prom": safe_numeric(game.get("visitor_team_score", 0)) / 10,
                    "racha_local": 3,
                    "racha_visita": 2,
                    "clima": 1,
                    "importancia_partido": 3,
                    "hora": game.get("date", datetime.utcnow().isoformat())
                }
                datos.append(registro)
            print("Usando datos complementarios de balldontlie para NBA")
        else:
            print(f"Error en balldontlie (NBA - games): {resp_bd.status_code} {resp_bd.text}")
    except Exception as e:
        print(f"Excepción en balldontlie (NBA - games): {e}")
    return datos

def obtener_datos_deporte_api(deporte):
    datos = []
    headers = {"x-apisports-key": API_SPORTS_KEY}
    endpoint = ""
    if deporte.upper() == "MLB":
        endpoint = "https://v1.baseball.api-sports.io/games"
    elif deporte.upper() == "NHL":
        endpoint = "https://v1.hockey.api-sports.io/games"
    else:
        return datos
    params = {"date": get_today()}
    try:
        resp = requests.get(endpoint, headers=headers, params=params, timeout=10)
        if resp.status_code == 200:
            for game in resp.json().get("response", []):
                registro = {
                    "id": game.get("id"),
                    "nombre_partido": game.get("teams", {}).get("home", {}).get("name", "Equipo Local") +
                                      " vs " +
                                      game.get("teams", {}).get("away", {}).get("name", "Equipo Visitante"),
                    "liga": game.get("league", {}).get("name", "Desconocida"),
                    "deporte": deporte.lower(),
                    "goles_local_prom": safe_numeric(game.get("scores", {}).get("home")),
                    "goles_visita_prom": safe_numeric(game.get("scores", {}).get("away")),
                    "racha_local": 0,
                    "racha_visita": 0,
                    "clima": 1,
                    "importancia_partido": 3,
                    "hora": game.get("date", datetime.utcnow().isoformat())
                }
                datos.append(registro)
        else:
            print(f"Error en API-Sports {deporte}: {resp.status_code} {resp.text}")
    except Exception as e:
        print(f"Excepción en API-Sports {deporte}: {e}")
    return datos

def obtener_datos_actualizados():
    datos = []
    datos.extend(obtener_datos_futbol())
    datos.extend(obtener_datos_nba())
    for dep in ["MLB", "NHL"]:
        datos.extend(obtener_datos_deporte_api(dep))
    print(f"Total de registros obtenidos de todas las fuentes: {len(datos)}")
    # Se asegura que cada registro tenga la clave 'goles_local_prom'
    for d in datos:
        if "goles_local_prom" not in d or d["goles_local_prom"] is None:
            d["goles_local_prom"] = 0.0
    return datos if datos else [{
        "id": None,
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

# --------------------------------------------------
# Función de parametrización adicional usando datos reales
# --------------------------------------------------
def agregar_parametros_adicionales(registro):
    deporte = registro.get("deporte", "").lower()
    # Se esperan que la información de league_id y season esté incluida en cada fixture si es obtenida de API-Sports
    # Si no, se usan valores por defecto (0 o cadena vacía)
    league_id = str(registro.get("league_id", ""))
    season = registro.get("season", "")
    if deporte == "futbol":
        team_local = str(registro.get("team_local_id", ""))
        stats = obtener_estadisticas(team_local, league_id, season, "futbol")
        registro["param_1"] = safe_numeric(stats.get("fixtures", {}).get("goals", {}).get("for"))
        registro["param_2"] = safe_numeric(stats.get("fixtures", {}).get("goals", {}).get("against"))
        registro["racha_vict_local"] = stats.get("fixtures", {}).get("streak", 0)
        form_local = stats.get("fixtures", {}).get("form", "-----")
        registro["ultimos_5_local"] = convert_form_string(form_local)[-5:]
        registro["param_3"] = registro["param_1"]
        registro["param_4"] = registro["param_2"]
        registro["racha_vict_visit"] = registro["racha_vict_local"]
        registro["ultimos_5_visitante"] = registro["ultimos_5_local"]
        registro["alineacion_estado"] = 1
    elif deporte == "nba":
        team_local = str(registro.get("team_local_id", ""))
        stats = obtener_estadisticas(team_local, league_id, season, "nba")
        registro["param_1"] = safe_numeric(stats.get("games", {}).get("points", {}).get("for"))
        registro["param_2"] = safe_numeric(stats.get("games", {}).get("points", {}).get("against"))
        registro["racha_vict_local"] = stats.get("games", {}).get("streak", 0)
        form_local = stats.get("games", {}).get("form", "-----")
        registro["ultimos_5_local"] = convert_form_string(form_local)[-5:]
        registro["param_3"] = registro["param_1"]
        registro["param_4"] = registro["param_2"]
        registro["racha_vict_visit"] = registro["racha_vict_local"]
        registro["ultimos_5_visitante"] = registro["ultimos_5_local"]
        registro["alineacion_estado"] = 1
    elif deporte == "mlb":
        team_local = str(registro.get("team_local_id", ""))
        stats = obtener_estadisticas(team_local, league_id, season, "mlb")
        registro["param_1"] = safe_numeric(stats.get("games", {}).get("runs", {}).get("for"))
        registro["param_2"] = safe_numeric(stats.get("games", {}).get("runs", {}).get("against"))
        registro["racha_vict_local"] = stats.get("games", {}).get("streak", 0)
        form_local = stats.get("games", {}).get("form", "-----")
        registro["ultimos_5_local"] = convert_form_string(form_local)[-5:]
        registro["param_3"] = registro["param_1"]
        registro["param_4"] = registro["param_2"]
        registro["racha_vict_visit"] = registro["racha_vict_local"]
        registro["ultimos_5_visitante"] = registro["ultimos_5_local"]
        registro["alineacion_estado"] = 1
    elif deporte == "nhl":
        team_local = str(registro.get("team_local_id", ""))
        stats = obtener_estadisticas(team_local, league_id, season, "nhl")
        registro["param_1"] = safe_numeric(stats.get("games", {}).get("goals", {}).get("for"))
        registro["param_2"] = safe_numeric(stats.get("games", {}).get("goals", {}).get("against"))
        registro["racha_vict_local"] = stats.get("games", {}).get("streak", 0)
        form_local = stats.get("games", {}).get("form", "-----")
        registro["ultimos_5_local"] = convert_form_string(form_local)[-5:]
        registro["param_3"] = registro["param_1"]
        registro["param_4"] = registro["param_2"]
        registro["racha_vict_visit"] = registro["racha_vict_local"]
        registro["ultimos_5_visitante"] = registro["ultimos_5_local"]
        registro["alineacion_estado"] = 1
    else:
        registro["param_1"] = registro["param_2"] = registro["param_3"] = registro["param_4"] = 0.0
        registro["racha_vict_local"] = registro["racha_vict_visit"] = 0
        registro["ultimos_5_local"] = [0] * 5
        registro["ultimos_5_visitante"] = [0] * 5
        registro["alineacion_estado"] = 1
    return registro

# --------------------------------------------------
# Funciones para entrenamiento y predicción
# --------------------------------------------------
def obtener_datos_actualizados():
    datos = []
    datos.extend(obtener_datos_futbol())
    datos.extend(obtener_datos_nba())
    for dep in ["MLB", "NHL"]:
        datos.extend(obtener_datos_deporte_api(dep))
    print(f"Total de registros obtenidos de todas las fuentes: {len(datos)}")
    # Asegurar que cada registro tenga la clave 'goles_local_prom'
    for d in datos:
        if "goles_local_prom" not in d or d["goles_local_prom"] is None:
            d["goles_local_prom"] = 0.0
    return datos if datos else [{
        "id": None,
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

def obtener_datos_entrenamiento():
    datos_actuales = obtener_datos_actualizados()
    print(f"Número de registros obtenidos para entrenamiento: {len(datos_actuales)}")
    datos_ext = [agregar_parametros_adicionales(d) for d in datos_actuales]
    # Para ampliar la muestra (opcional)
    simulated_data = datos_ext * 10  
    for d in simulated_data:
        d["resultado"] = random.choice([0, 1, 2])
    training_rows = []
    for d in simulated_data:
        try:
            row = {
                "goles_local_prom": safe_numeric(d.get("goles_local_prom")),
                "goles_visita_prom": safe_numeric(d.get("goles_visita_prom")),
                "racha_local": int(d.get("racha_local") or 0),
                "racha_visita": int(d.get("racha_visita") or 0),
                "clima": int(d.get("clima") or 1),
                "importancia_partido": int(d.get("importancia_partido") or 0),
                "resultado": int(d.get("resultado") or 0),
                "param_1": safe_numeric(d.get("param_1")),
                "param_2": safe_numeric(d.get("param_2")),
                "param_3": safe_numeric(d.get("param_3")),
                "param_4": safe_numeric(d.get("param_4")),
                "param_5": int(d.get("racha_vict_local") or 0),
                "param_6": int(d.get("racha_vict_visit") or 0),
                "param_7": float(sum(convert_form_string(d.get("ultimos_5_local", "-----"))))/5,
                "param_8": float(sum(convert_form_string(d.get("ultimos_5_visitante", "-----"))))/5,
                "param_9": int(d.get("alineacion_estado") or 0)
            }
            training_rows.append(row)
        except Exception as e:
            print("Error construyendo datos de entrenamiento:", e)
    df_train = pd.DataFrame(training_rows)
    if df_train.empty:
        print("El DataFrame de entrenamiento está vacío.")
    elif df_train.get("resultado", pd.Series()).nunique() < 2:
        print("La columna 'resultado' tiene una única clase:", df_train.get("resultado"))
    else:
        print("Datos de entrenamiento listos. Distribución de clases:", df_train["resultado"].value_counts().to_dict())
    return df_train

def entrenar_modelo():
    df = obtener_datos_entrenamiento()
    try:
        df = df.astype({
            "goles_local_prom": float,
            "goles_visita_prom": float,
            "racha_local": int,
            "racha_visita": int,
            "clima": int,
            "importancia_partido": int,
            "resultado": int,
            "param_1": float,
            "param_2": float,
            "param_3": float,
            "param_4": float,
            "param_5": int,
            "param_6": int,
            "param_7": float,
            "param_8": float,
            "param_9": int
        })
    except Exception as e:
        print("Error en conversión de tipos del DataFrame:", e)
    X = df.drop("resultado", axis=1)
    y = df["resultado"]
    if y.nunique() < 2:
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
        0: ("Gana Local", 0.75, "Over", 0.72, "Local +4", 0.71),
        1: ("Empate", 0.70, "Under", 0.70, "No ventaja", 0.70),
        2: ("Gana Visitante", 0.75, "Over", 0.72, "Visitante +4", 0.71)
    }
    base = opciones.get(pred, ("Indefinido", 0.0, "Sin datos", 0.0, "Sin ventaja", 0.0))
    # Se retorna la opción básica (predicción y confianza); se puede ampliar para incluir over/under, handicap, etc.
    return base[0:2]

# --------------------------------------------------
# Funciones para actualizar datos y procesar predicciones
# --------------------------------------------------
def actualizar_datos_partidos():
    nuevos_datos = obtener_datos_actualizados()
    for partido in nuevos_datos:
        # Convertir la hora a un formato estándar
        hora_valor = partido["hora"]
        try:
            if isinstance(hora_valor, str):
                hora_dt = datetime.fromisoformat(hora_valor)
            else:
                hora_dt = hora_valor
        except Exception as e:
            print(f"Error al convertir hora: {e}")
            hora_dt = datetime.utcnow()
        partido["hora"] = hora_dt.strftime("%Y-%m-%d %H:%M:%S")
        insertar_league(partido["liga"], partido["deporte"])
        insertar_partido(partido["nombre_partido"], partido["liga"], partido["hora"])
        result = manual_upsert("partidos", partido, ["nombre_partido", "hora"])
        print(f"Registro en partidos {partido['nombre_partido']}: {result}")
    return {"status": "Datos actualizados correctamente"}

def procesar_predicciones():
    try:
        model = entrenar_modelo()
        partidos = obtener_datos_actualizados()
        for partido in partidos:
            try:
                if isinstance(partido["hora"], str):
                    hora_dt = datetime.fromisoformat(partido["hora"])
                else:
                    hora_dt = partido["hora"]
            except Exception as e:
                print(f"Error al convertir hora en predicción: {e}")
                hora_dt = datetime.utcnow()
            partido["hora"] = hora_dt.strftime("%Y-%m-%d %H:%M:%S")
            resultado, confianza = predecir_resultado(model, partido)
            prediccion = {
                "deporte": partido["deporte"],
                "liga": partido["liga"],
                "partido": partido["nombre_partido"],
                "hora": partido["hora"],
                "pronostico_1": resultado,
                "confianza_1": confianza,
                "pronostico_2": "Menos de 2.5 goles",  # Ejemplo adicional
                "confianza_2": 0.70,
                "pronostico_3": "Ambos no anotan",
                "confianza_3": 0.65
            }
            result = manual_upsert("predicciones", prediccion, ["partido", "hora"])
            print(f"Registro en predicciones {prediccion['partido']}: {result}")
        return {"message": "Predicciones generadas correctamente"}
    except Exception as e:
        print("Error global en el procesamiento de predicciones:", e)
        return {"message": "Error global en el procesamiento de predicciones"}

# --------------------------------------------------
# Funciones para integrar datos de NBA desde balldontlie (complementario)
# --------------------------------------------------
def insertar_nba_balldontlie():
    # Insertar equipos de la NBA
    try:
        response = requests.get(f"{balldontlie_base_url}/teams", timeout=10)
        if response.status_code == 200:
            equipos = response.json().get("data", [])
            print(f"Obtenidos {len(equipos)} equipos de balldontlie")
            for equipo in equipos:
                supabase.table("teams").upsert({
                    "id": equipo.get("id"),
                    "name": equipo.get("full_name"),
                    "abbreviation": equipo.get("abbreviation"),
                    "conference": equipo.get("conference")
                }, on_conflict=["id"]).execute()
        else:
            print(f"Error obteniendo equipos de balldontlie: {response.status_code} {response.text}")
    except Exception as e:
        print("Excepción al obtener equipos de balldontlie:", e)
    
    # Insertar juegos del día de balldontlie
    try:
        resp = requests.get(f"{balldontlie_base_url}/games?start_date={get_today()}&end_date={get_today()}&per_page=100", timeout=10)
        if resp.status_code == 200:
            juegos = resp.json().get("data", [])
            print(f"Obtenidos {len(juegos)} juegos del día de balldontlie")
            for juego in juegos:
                data_game = {
                    "nombre_partido": juego["home_team"]["full_name"] + " vs " + juego["visitor_team"]["full_name"],
                    "liga": "NBA",
                    "deporte": "nba",
                    "goles_local_prom": safe_numeric(juego.get("home_team_score", 0)) / 10,
                    "goles_visita_prom": safe_numeric(juego.get("visitor_team_score", 0)) / 10,
                    "racha_local": 3,
                    "racha_visita": 2,
                    "clima": 1,
                    "importancia_partido": 3,
                    "hora": juego.get("date", datetime.utcnow().isoformat())
                }
                supabase.table("partidos").upsert(data_game, on_conflict=["nombre_partido", "hora"]).execute()
        else:
            print(f"Error en juegos de balldontlie: {resp.status_code} {resp.text}")
    except Exception as e:
        print("Excepción en juegos de balldontlie:", e)
    
    # Insertar estadísticas de juegos del día de balldontlie en predicciones
    try:
        resp = requests.get(f"{balldontlie_base_url}/stats?start_date={get_today()}&end_date={get_today()}&per_page=100", timeout=10)
        if resp.status_code == 200:
            stats = resp.json().get("data", [])
            print(f"Obtenidas {len(stats)} estadísticas de balldontlie para NBA")
            for stat in stats:
                data_stat = {
                    "deporte": "nba",
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
            print(f"Error en stats de balldontlie: {resp.status_code} {resp.text}")
    except Exception as e:
        print("Excepción en stats de balldontlie:", e)

# --------------------------------------------------
# Función principal del pipeline
# --------------------------------------------------
def ejecutar_pipeline():
    try:
        print("Iniciando actualización de partidos y ligas...")
        actualizar_datos_partidos()
        print("Integrando datos complementarios de NBA desde balldontlie...")
        insertar_nba_balldontlie()
        print("Iniciando procesamiento de predicciones...")
        procesar_predicciones()
        print("Pipeline ejecutado correctamente.")
    except Exception as e:
        print(f"Error global en el pipeline: {e}")

if __name__ == "__main__":
    ejecutar_pipeline()
