from fastapi import FastAPI
from app.api.endpoints import router as api_router

app = FastAPI(title="Ecommerce ML Service", version="1.0.0")

@app.get("/")
async def root():
    return {"message": "Ecommerce ML Analyzer API is running"}

# Incluir los endpoints de análisis
app.include_router(api_router, prefix="/api/v1")
