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

# Cargar variables de entorno (asegúrate de tener un archivo .env configurado)
from dotenv import load_dotenv
load_dotenv()

# Conexión a Supabase (usando variables de entorno definidas en .env)
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Claves API
api_sports_key = os.getenv("API_SPORTS_KEY")
# Para NBA, se integran además las URL de la API de balldontlie (la API es pública)
balldontlie_base_url = "https://api.balldontlie.io/v1"
balldontlie_api_key = "5293380d-9c4f-4e89-be15-bf19b1042182"  # Nota: Se inserta en el código si fuera necesaria la autorización

# --- VARIABLES DE EJEMPLO PARA CONSULTAS REALES (reemplazar según corresponda) ---
# Fútbol (Fulbot)
football_league_id = "39"  # Ejemplo: Premier League
football_team_id = "33"    # Ejemplo: Manchester United
football_season = "2023"

# NBA
nba_league_id = "1"        # Ejemplo (depende de la API)
nba_team_id = "14"         # Ejemplo: Golden State Warriors (cambiar según datos reales)
nba_season = "2023-2024"

# MLB
mlb_league_id = "1"        # Ejemplo
mlb_team_id = "1"          # Ejemplo
mlb_season = "2023"

# NHL
nhl_league_id = "1"        # Ejemplo
nhl_team_id = "1"          # Ejemplo
nhl_season = "2023"

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
        print(f"Error al insertar liga {nombre_liga}: {e}")

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
        print(f"Error al insertar partido {nombre_partido}: {e}")

# Función de upsert con reintentos
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

# ===============================
# FIN DE NUEVAS FUNCIONES
# ===============================

# CONFIGURACIÓN DE APIs DEPORTIVAS

# Para Fútbol: Se utiliza API-Sports (v3) y OpenLigaDB.
OPENLIGADB_ENDPOINTS = [
    "https://api.openligadb.de/getmatchdata/dfb/2024/5",
    "https://api.openligadb.de/getmatchdata/bl2/2024/28",
    "https://api.openligadb.de/getmatchdata/bl3/2024/31",
    "https://api.openligadb.de/getmatchdata/ucl24/2024/12",
    "https://api.openligadb.de/getmatchdata/bl1/2024/28",
    "https://api.openligadb.de/getmatchdata/ucl2024/2024/4",
    "https://api.openligadb.de/getmatchdata/uel24/2024/12"
]
# Para NBA: Fuente principal: API-Sports (v2, con fallback v1) y se complementa con la API de balldontlie.
# Para MLB y NHL: se usan los endpoints de API-Sports.

def get_today():
    return datetime.utcnow().strftime("%Y-%m-%d")

# -------------------------------
# Funciones para Fútbol
# -------------------------------
def obtener_datos_futbol():
    datos = []
    today = get_today()
    try:
        # Obtener fixtures del día desde API-Sports (v3)
        url_api = f"https://v3.football.api-sports.io/fixtures?date={today}"
        headers_api = {"x-apisports-key": api_sports_key}
        resp = requests.get(url_api, headers=headers_api, timeout=10)
        if resp.status_code == 200:
            respuesta = resp.json().get("response", [])
            print(f"Obtenidos {len(respuesta)} fixtures de API-Sports fútbol")
            for fixture in respuesta:
                info = fixture.get("fixture", {})
                teams = fixture.get("teams", {})
                registro = {
                    "nombre_partido": teams.get("home", {}).get("name", "Partido Desconocido") + " vs " +
                                      teams.get("away", {}).get("name", ""),
                    "liga": fixture.get("league", {}).get("name", "Desconocida"),
                    "deporte": "futbol",
                    "goles_local_prom": None,  # A rellenar con estadísticas reales
                    "goles_visita_prom": None,
                    "racha_local": None,
                    "racha_visita": None,
                    "clima": 1,
                    "importancia_partido": None,  # Se obtiene mediante análisis de fixture
                    "hora": info.get("date", datetime.utcnow().isoformat())
                }
                # Obtener estadísticas reales para cada equipo usando el endpoint de statistics:
                # Ejemplo para el equipo local (se requieren valores reales de team_id, league_id y season)
                local_stats_url = f"https://v3.football.api-sports.io/teams/statistics?league={football_league_id}&team={teams.get('home', {}).get('id', football_team_id)}&season={football_season}"
                try:
                    stats_resp = requests.get(local_stats_url, headers=headers_api, timeout=10)
                    if stats_resp.status_code == 200:
                        stats = stats_resp.json().get("response", {})
                        # Suponemos que stats incluye "fixtures" con claves "goals" y "form" (ajustar según la respuesta real)
                        registro["goles_local_prom"] = stats.get("fixtures", {}).get("goals", {}).get("for", None)
                        registro["racha_local"] = stats.get("fixtures", {}).get("streak", None)
                    else:
                        print("Error en estadísticas de equipo local:", stats_resp.status_code)
                except Exception as e:
                    print("Excepción al obtener estadísticas local:", e)
                # De forma similar, obtener para el visitante
                away_stats_url = f"https://v3.football.api-sports.io/teams/statistics?league={football_league_id}&team={teams.get('away', {}).get('id', football_team_id)}&season={football_season}"
                try:
                    stats_resp = requests.get(away_stats_url, headers=headers_api, timeout=10)
                    if stats_resp.status_code == 200:
                        stats = stats_resp.json().get("response", {})
                        registro["goles_visita_prom"] = stats.get("fixtures", {}).get("goals", {}).get("for", None)
                        registro["racha_visita"] = stats.get("fixtures", {}).get("streak", None)
                    else:
                        print("Error en estadísticas de equipo visitante:", stats_resp.status_code)
                except Exception as e:
                    print("Excepción al obtener estadísticas visitante:", e)
                # Importancia de partido y estado de alineación se derivan de un análisis del fixture (aquí se deja como None)
                registro["importancia_partido"] = 3
                datos.append(registro)
        else:
            print("Error en API-Sports fútbol:", resp.status_code, resp.text)
    except Exception as e:
        print("Excepción en API-Sports fútbol:", e)
    # Complemento: Obtener fixtures de OpenLigaDB (se mantienen datos básicos; aquí no se obtienen parámetros adicionales reales)
    for endpoint in OPENLIGADB_ENDPOINTS:
        try:
            resp = requests.get(endpoint, timeout=10)
            if resp.status_code == 200:
                fixtures = resp.json()
                print(f"Obtenidos {len(fixtures)} fixtures de OpenLigaDB")
                for partido in fixtures:
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
            else:
                print("Error en OpenLigaDB:", resp.status_code)
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
    # Fuente principal: API-Sports NBA v2 (ejemplo: obtener estadísticas reales)
    try:
        url_api = f"https://v2.nba.api-sports.io/games?date={today}"
        headers_api = {"x-apisports-key": api_sports_key}
        resp = requests.get(url_api, headers=headers_api, timeout=10)
        if resp.status_code == 200:
            respuesta = resp.json().get("response", [])
            print(f"Obtenidos {len(respuesta)} juegos de API-Sports NBA (v2)")
            if isinstance(respuesta, list):
                for game in respuesta:
                    if not isinstance(game, dict):
                        print("Elemento inesperado en API-Sports NBA (v2):", game)
                        continue
                    equipos = game.get("teams", {})
                    # Obtener estadísticas reales del equipo local usando endpoint de estadísticas
                    nba_stats_url_local = f"https://v2.nba.api-sports.io/teams/statistics?league={nba_league_id}&team={equipos.get('home', {}).get('id', nba_team_id)}&season={nba_season}"
                    try:
                        stats_local_resp = requests.get(nba_stats_url_local, headers=headers_api, timeout=10)
                        if stats_local_resp.status_code == 200:
                            stats_local = stats_local_resp.json().get("response", {})
                        else:
                            print("Error en estadísticas NBA local:", stats_local_resp.status_code)
                            stats_local = {}
                    except Exception as e:
                        print("Excepción en estadísticas NBA local:", e)
                        stats_local = {}
                    # Similar para el visitante
                    nba_stats_url_away = f"https://v2.nba.api-sports.io/teams/statistics?league={nba_league_id}&team={equipos.get('away', {}).get('id', nba_team_id)}&season={nba_season}"
                    try:
                        stats_away_resp = requests.get(nba_stats_url_away, headers=headers_api, timeout=10)
                        if stats_away_resp.status_code == 200:
                            stats_away = stats_away_resp.json().get("response", {})
                        else:
                            print("Error en estadísticas NBA visitante:", stats_away_resp.status_code)
                            stats_away = {}
                    except Exception as e:
                        print("Excepción en estadísticas NBA visitante:", e)
                        stats_away = {}
                    # Extraer puntajes básicos desde API-Sports
                    score_home = game.get("scores", {}).get("home", 100)
                    if isinstance(score_home, dict):
                        score_home = score_home.get("points", 100)
                    score_away = game.get("scores", {}).get("away", 100)
                    if isinstance(score_away, dict):
                        score_away = score_away.get("points", 100)
                    registro = {
                        "nombre_partido": equipos.get("home", {}).get("name", "Partido Desconocido") + " vs " +
                                          equipos.get("away", {}).get("name", ""),
                        "liga": game.get("league", {}).get("name", "NBA"),
                        "deporte": "nba",
                        "goles_local_prom": float(score_home) / 10,  # Aquí se interpretan los puntos
                        "goles_visita_prom": float(score_away) / 10,
                        "racha_local": stats_local.get("fixtures", {}).get("streak", None),
                        "racha_visita": stats_away.get("fixtures", {}).get("streak", None),
                        "clima": 1,
                        "importancia_partido": 3,  # Se podría derivar de la información del fixture
                        "hora": game.get("date", datetime.utcnow().isoformat())
                    }
                    datos.append(registro)
                success = True
            else:
                print("Respuesta inesperada en API-Sports NBA (v2):", resp.json())
        else:
            print("Error en API-Sports NBA (v2):", resp.status_code, resp.text)
    except Exception as e:
        print("Excepción en API-Sports NBA (v2):", e)
    # Fallback: si v2 no funciona, intentar con la URL alternativa de Basketball API (v1)
    if not success:
        try:
            url_api = f"https://v1.basketball.api-sports.io/games?date={today}"
            headers_api = {"x-apisports-key": api_sports_key}
            resp = requests.get(url_api, headers=headers_api, timeout=10)
            if resp.status_code == 200:
                respuesta = resp.json().get("response", [])
                print(f"Obtenidos {len(respuesta)} juegos de API-Sports NBA (v1)")
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
                            "deporte": "nba",
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
    # Complemento: Utilizar la API de balldontlie para obtener juegos y estadísticas reales
    try:
        url_bd = f"{balldontlie_base_url}/games"
        params = {"start_date": today, "end_date": today, "per_page": 100}
        resp_bd = requests.get(url_bd, params=params, timeout=10)
        if resp_bd.status_code == 200:
            games = resp_bd.json().get("data", [])
            print(f"Obtenidos {len(games)} juegos de balldontlie para NBA")
            url_stats = f"{balldontlie_base_url}/stats"
            params_stats = {"start_date": today, "end_date": today, "per_page": 100}
            resp_stats = requests.get(url_stats, params=params_stats, timeout=10)
            stats = []
            if resp_stats.status_code == 200:
                stats = resp_stats.json().get("data", [])
                print(f"Obtenidas {len(stats)} estadísticas de balldontlie para NBA")
            else:
                print("Error en balldontlie (NBA - stats):", resp_stats.status_code, resp_stats.text)
            for game in games:
                try:
                    game_id = game.get("id")
                    game_stats = [s for s in stats if s.get("game", {}).get("id") == game_id]
                    datos.append({
                        "nombre_partido": game["home_team"]["full_name"] + " vs " + game["visitor_team"]["full_name"],
                        "liga": "NBA",
                        "deporte": "nba",
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
                        "deporte": "mlb",
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
                        "deporte": "nhl",
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
    print(f"Total de registros obtenidos de todas las fuentes: {len(datos)}")
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
# Funciones para parametrización adicional con datos reales
# -------------------------------
def agregar_parametros_adicionales(registro):
    # Para cada deporte, se hace una llamada real al endpoint de estadísticas de la temporada.
    # Los siguientes ejemplos usan endpoints de API-Sports con valores de ejemplo para {league_id}, {team_id} y {season}.
    deporte = registro.get("deporte", "").lower()
    headers_api = {"x-apisports-key": api_sports_key}
    if deporte == "futbol":
        # Ejemplo: consultar estadísticas para el equipo local
        team_id = registro.get("team_local_id", football_team_id)  # Debe estar presente en el fixture; se usa un valor por defecto
        url_stats = f"https://v3.football.api-sports.io/teams/statistics?league={football_league_id}&team={team_id}&season={football_season}"
        try:
            resp = requests.get(url_stats, headers=headers_api, timeout=10)
            if resp.status_code == 200:
                stats = resp.json().get("response", {}).get("fixtures", {})
                registro["param_1"] = stats.get("goals", {}).get("for", None)    # goles anotados local promedio
                registro["param_2"] = stats.get("goals", {}).get("against", None)  # goles permitidos local promedio
                registro["racha_vict_local"] = stats.get("streak", None)           # racha de victorias local
                registro["ultimos_5_local"] = stats.get("form", "-----")           # Ultimos 5 resultados (string, se debe parsear)
                registro["param_3"] = stats.get("goals", {}).get("for", None)        # Se usa igual para visitante (puedes ajustar)
                registro["param_4"] = stats.get("goals", {}).get("against", None)
                registro["racha_vict_visit"] = stats.get("streak", None)
                registro["ultimos_5_visitante"] = stats.get("form", "-----")
            else:
                print("Error en estadísticas fútbol:", resp.status_code, resp.text)
                registro["param_1"] = registro["param_2"] = registro["param_3"] = registro["param_4"] = 0
                registro["racha_vict_local"] = registro["racha_vict_visit"] = 0
                registro["ultimos_5_local"] = registro["ultimos_5_visitante"] = [0]*5
        except Exception as e:
            print("Excepción en estadísticas fútbol:", e)
            registro["param_1"] = registro["param_2"] = registro["param_3"] = registro["param_4"] = 0
            registro["racha_vict_local"] = registro["racha_vict_visit"] = 0
            registro["ultimos_5_local"] = registro["ultimos_5_visitante"] = [0]*5
        registro["alineacion_estado"] = 1  # Se podría obtener de otro endpoint
    elif deporte == "nba":
        team_id = registro.get("team_local_id", nba_team_id)
        url_stats = f"https://v2.nba.api-sports.io/teams/statistics?league={nba_league_id}&team={team_id}&season={nba_season}"
        try:
            resp = requests.get(url_stats, headers=headers_api, timeout=10)
            if resp.status_code == 200:
                stats = resp.json().get("response", {}).get("games", {})
                registro["param_1"] = stats.get("points", {}).get("for", None)    # puntos anotados local promedio
                registro["param_2"] = stats.get("points", {}).get("against", None)  # puntos permitidos local promedio
                registro["racha_vict_local"] = stats.get("streak", None)
                registro["ultimos_5_local"] = stats.get("form", "-----")
            else:
                print("Error en estadísticas NBA (local):", resp.status_code, resp.text)
                registro["param_1"] = registro["param_2"] = 0
                registro["racha_vict_local"] = 0
                registro["ultimos_5_local"] = [0]*5
        except Exception as e:
            print("Excepción en estadísticas NBA (local):", e)
            registro["param_1"] = registro["param_2"] = 0
            registro["racha_vict_local"] = 0
            registro["ultimos_5_local"] = [0]*5
        # Para el visitante, se asume lo mismo (en la práctica harías una llamada con team_id del visitante)
        registro["param_3"] = registro["param_1"]
        registro["param_4"] = registro["param_2"]
        registro["racha_vict_visit"] = registro["racha_vict_local"]
        registro["ultimos_5_visitante"] = registro["ultimos_5_local"]
        registro["alineacion_estado"] = 1
    elif deporte == "mlb":
        team_id = registro.get("team_local_id", mlb_team_id)
        url_stats = f"https://v1.baseball.api-sports.io/teams/statistics?league={mlb_league_id}&team={team_id}&season={mlb_season}"
        try:
            resp = requests.get(url_stats, headers=headers_api, timeout=10)
            if resp.status_code == 200:
                stats = resp.json().get("response", {}).get("games", {})
                registro["param_1"] = stats.get("runs", {}).get("for", None)      # carreras anotadas local promedio
                registro["param_2"] = stats.get("runs", {}).get("against", None)    # carreras permitidas local promedio
                registro["racha_vict_local"] = stats.get("streak", None)
                registro["ultimos_5_local"] = stats.get("form", "-----")
            else:
                print("Error en estadísticas MLB (local):", resp.status_code, resp.text)
                registro["param_1"] = registro["param_2"] = 0
                registro["racha_vict_local"] = 0
                registro["ultimos_5_local"] = [0]*5
        except Exception as e:
            print("Excepción en estadísticas MLB (local):", e)
            registro["param_1"] = registro["param_2"] = 0
            registro["racha_vict_local"] = 0
            registro["ultimos_5_local"] = [0]*5
        registro["param_3"] = registro["param_1"]
        registro["param_4"] = registro["param_2"]
        registro["racha_vict_visit"] = registro["racha_vict_local"]
        registro["ultimos_5_visitante"] = registro["ultimos_5_local"]
        registro["alineacion_estado"] = 1
    elif deporte == "nhl":
        team_id = registro.get("team_local_id", nhl_team_id)
        url_stats = f"https://v1.hockey.api-sports.io/teams/statistics?league={nhl_league_id}&team={team_id}&season={nhl_season}"
        try:
            resp = requests.get(url_stats, headers=headers_api, timeout=10)
            if resp.status_code == 200:
                stats = resp.json().get("response", {}).get("games", {})
                registro["param_1"] = stats.get("goals", {}).get("for", None)      # goles anotados local promedio
                registro["param_2"] = stats.get("goals", {}).get("against", None)    # goles permitidos local promedio
                registro["racha_vict_local"] = stats.get("streak", None)
                registro["ultimos_5_local"] = stats.get("form", "-----")
            else:
                print("Error en estadísticas NHL (local):", resp.status_code, resp.text)
                registro["param_1"] = registro["param_2"] = 0
                registro["racha_vict_local"] = 0
                registro["ultimos_5_local"] = [0]*5
        except Exception as e:
            print("Excepción en estadísticas NHL (local):", e)
            registro["param_1"] = registro["param_2"] = 0
            registro["racha_vict_local"] = 0
            registro["ultimos_5_local"] = [0]*5
        registro["param_3"] = registro["param_1"]
        registro["param_4"] = registro["param_2"]
        registro["racha_vict_visit"] = registro["racha_vict_local"]
        registro["ultimos_5_visitante"] = registro["ultimos_5_local"]
        registro["alineacion_estado"] = 1
    else:
        registro["param_1"] = registro["param_2"] = registro["param_3"] = registro["param_4"] = 0.0
        registro["param_5"] = registro["param_6"] = 0
        registro["ultimos_5_local"] = [0]*5
        registro["ultimos_5_visitante"] = [0]*5
        registro["alineacion_estado"] = 1
    return registro

# -------------------------------
# Funciones para Entrenamiento y Predicción de Todos los Deportes
# -------------------------------
def obtener_datos_actualizados():
    datos = []
    datos.extend(obtener_datos_futbol())
    datos.extend(obtener_datos_nba())
    for dep in ["MLB", "NHL"]:
        datos.extend(obtener_datos_deporte_api(dep))
    print(f"Total de registros obtenidos de todas las fuentes: {len(datos)}")
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

def obtener_datos_entrenamiento():
    datos_actuales = obtener_datos_actualizados()
    print(f"Número de registros obtenidos para entrenamiento: {len(datos_actuales)}")
    # Agregar parámetros adicionales reales para cada deporte
    datos_ext = [agregar_parametros_adicionales(d) for d in datos_actuales]
    simulated_data = datos_ext * 10  # Aumenta la cantidad de registros
    for d in simulated_data:
        d["resultado"] = random.choice([0, 1, 2])
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
                "resultado": int(d["resultado"]),
                "param_1": float(d.get("param_1", 0.0)),
                "param_2": float(d.get("param_2", 0.0)),
                "param_3": float(d.get("param_3", 0.0)),
                "param_4": float(d.get("param_4", 0.0)),
                "param_5": int(d.get("racha_vict_local", 0)),
                "param_6": int(d.get("racha_vict_visit", 0)),
                "param_7": float(sum(d.get("ultimos_5_local", [0]*5)))/5,
                "param_8": float(sum(d.get("ultimos_5_visitante", [0]*5)))/5,
                "param_9": int(d.get("alineacion_estado", 1))
            }
            training_rows.append(row)
        except Exception as e:
            print("Error construyendo datos de entrenamiento:", e)
    df_train = pd.DataFrame(training_rows)
    if df_train.empty:
        print("El DataFrame de entrenamiento está vacío.")
    elif df_train["resultado"].nunique() < 2:
        print("La columna 'resultado' tiene una única clase:", df_train["resultado"].unique())
    else:
        print("Datos de entrenamiento listos. Distribución de clases:", df_train["resultado"].value_counts().to_dict())
    return df_train

def entrenar_modelo():
    df = obtener_datos_entrenamiento()
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
    # Se amplían las opciones de predicción: 
    # Además de la predicción básica, se añade una predicción de "over/under" y "handicap".
    opciones = {
        0: ("Gana Local", 0.75, "Alta anotación", 0.72, "Local con ventaja", 0.71),
        1: ("Empate", 0.70, "Baja anotación", 0.70, "Ninguna ventaja", 0.70),
        2: ("Gana Visitante", 0.75, "Alta anotación", 0.72, "Visitante con ventaja", 0.71)
    }
    # Retornamos una tupla con 3 predicciones: (prediccion, confianza), (over/under, confianza), (handicap, confianza)
    base = opciones.get(pred, ("Indefinido", 0.0, "Sin datos", 0.0, "Sin ventaja", 0.0))
    return base[0:2]  # Para mantener la estructura original (puedes ajustar para usar todas las predicciones)

# -------------------------------
# Funciones para Actualizar Datos y Procesar Predicciones
# -------------------------------
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
    try:
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
                "pronostico_2": "Menos de 2.5 goles",  # Estos campos se pueden ampliar
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

# -------------------------------
# Funciones para integración de datos NBA desde balldontlie
# -------------------------------
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
            print("Error obteniendo equipos de balldontlie:", response.status_code, response.text)
    except Exception as e:
        print("Excepción al obtener equipos de balldontlie:", e)
    
    # Insertar juegos del día de balldontlie
    try:
        today = get_today()
        response = requests.get(f"{balldontlie_base_url}/games?start_date={today}&end_date={today}", timeout=10)
        if response.status_code == 200:
            juegos = response.json().get("data", [])
            print(f"Obtenidos {len(juegos)} juegos del día de balldontlie")
            for juego in juegos:
                data_game = {
                    "nombre_partido": juego["home_team"]["full_name"] + " vs " + juego["visitor_team"]["full_name"],
                    "liga": "NBA",
                    "deporte": "nba",
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
    
    # Insertar estadísticas de juegos del día como registros en predicciones
    try:
        today = get_today()
        response = requests.get(f"{balldontlie_base_url}/stats?start_date={today}&end_date={today}&per_page=100", timeout=10)
        if response.status_code == 200:
            stats = response.json().get("data", [])
            print(f"Obtenidas {len(stats)} estadísticas de balldontlie")
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
            print("Error en stats de balldontlie:", response.status_code, response.text)
    except Exception as e:
        print("Excepción en stats de balldontlie:", e)

# -------------------------------
# Función principal del pipeline
# -------------------------------
def ejecutar_pipeline():
    try:
        actualizar_datos_partidos()
        insertar_nba_balldontlie()
        procesar_predicciones()
        print("Pipeline ejecutado correctamente.")
    except Exception as e:
        print("Error global en el pipeline:", e)

if __name__ == "__main__":
    ejecutar_pipeline()
