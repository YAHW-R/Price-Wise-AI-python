# Servicio de IA Price-Wise

El **Servicio de IA** es un microservicio de alto rendimiento construido con Python y FastAPI, diseñado para analizar datos de productos de comercio electrónico en tiempo real. Utiliza **LightGBM** (Gradient Boosting de Microsoft) para detectar posibles estafas y evaluar métricas de relación calidad-precio.

## 🧠 Funcionalidades Core

- **Ingeniería de Características:** Calcula métricas avanzadas como `discount_percentage` (porcentaje de descuento) y `trust_score` (basado en calificaciones y cantidad de reseñas).
- **Inferencia de ML:** Procesa los datos de los productos a través de modelos LightGBM entrenados para predecir probabilidades de estafa y puntuaciones de valor.
- **Interfaz API:** Proporciona una interfaz RESTful para que el Broker solicite el análisis de productos.

## 🛠 Stack Tecnológico

- **Framework:** FastAPI
- **Motor de ML:** LightGBM
- **Procesamiento de Datos:** Pandas, Scikit-learn
- **Servidor:** Uvicorn

## 🚀 Primeros Pasos

### Requisitos Previos

- Python 3.11+
- pip

### Instalación

1.  Navegue al directorio del servicio de IA:
    ```bash
    cd AI-python
    ```
2.  Instale las dependencias:
    ```bash
    pip install -r requirements.txt
    ```

### Ejecución del Servicio

Inicie el servidor FastAPI:
```bash
uvicorn app.main:app --port 5001 --reload
```
El servicio estará disponible en `http://localhost:5001`.

## 📡 Endpoints de la API

### `POST /api/v1/analyze`
Analiza un producto o un lote de productos.

**Cuerpo de la Solicitud:**
```json
{
  "product_id": "string",
  "title": "string",
  "current_price": 100.0,
  "original_price": 150.0,
  "rating": 4.5,
  "reviews_count": 120,
  "shop": "Nombre de la Tienda",
  ...
}
```

**Respuesta:**
```json
{
  "scam_probability": 0.05,
  "value_score": 0.85,
  "features_used": {
     "discount_percentage": 33.3,
     "trust_score": 21.5
  }
}
```

## 🏋️ Entrenamiento del Modelo

Para reentrenar los modelos con nuevos datos:
1. Asegúrese de que `train_data.csv` esté actualizado.
2. Ejecute el script de entrenamiento:
   ```bash
   python train_model.py
   ```
Esto actualizará los pesos en el directorio `model_weights/`.

## 📁 Estructura

- `app/`: Aplicación FastAPI, endpoints y servicios.
- `model_weights/`: Modelos LightGBM serializados y codificadores.
- `notebooks/`: Notebooks de investigación y desarrollo.
- `train_model.py`: Script para el entrenamiento y evaluación del modelo.
