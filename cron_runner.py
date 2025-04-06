import time
import requests
import os

# Cargar variables desde entorno
PREDICTION_URL = os.getenv("PREDICTION_URL", "https://sports-prediction-api-2a0q.onrender.com/api/v1/predicciones")
API_KEY = os.getenv("API_KEY")  # si usas autenticación opcional

def main():
    while True:
        try:
            headers = {"Authorization": f"Bearer {API_KEY}"} if API_KEY else {}
            response = requests.post(PREDICTION_URL, headers=headers)
            print(f"Predicción enviada. Código: {response.status_code} - {response.text}")
        except Exception as e:
            print(f"Error al enviar predicción: {e}")
        
        time.sleep(600)  # Esperar 10 minutos (600 segundos)

if __name__ == "__main__":
    main()
