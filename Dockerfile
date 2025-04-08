# Usa una imagen base de Python 3.11-slim
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

# Copiar todos los archivos al contenedor (incluyendo el modelo)
COPY . .

# Instalar dependencias de Python
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Render asigna un puerto a través de una variable de entorno (por defecto usamos 10000)
ENV PORT=10000
EXPOSE ${PORT}

# Ejecutar la aplicación
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "10000"]
