# Usa una imagen base de Python 3.11 (slim)
FROM python:3.11-slim

# Instalar dependencias del sistema necesarias
RUN apt-get update && apt-get install -y \
    build-essential \
    libssl-dev \
    libffi-dev \
    libxgboost-dev \
    && rm -rf /var/lib/apt/lists/*

# Establecer directorio de trabajo
WORKDIR /app

# Copiar archivos del proyecto
COPY . .

# Instalar dependencias de Python
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Exponer el puerto (Render asigna el puerto mediante la variable de entorno PORT)
ENV PORT=10000
EXPOSE ${PORT}

# Comando para ejecutar la aplicación utilizando uvicorn,
# se usa la variable PORT para binding dinámico (Render inyecta $PORT)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "10000"]
