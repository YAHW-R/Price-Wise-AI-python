from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Union
from app.services.inference import engine

router = APIRouter()

# Schema basado en el Product en Go (GEMINI.md 3.1)
class Product(BaseModel):
    product_id: str
    job_id: str
    title: str
    current_price: float
    original_price: float
    rating: float
    reviews_count: int
    availability: bool
    shop: str
    image_url: str
    product_url: str

@router.post("/analyze")
async def analyze_products(products: Union[List[Product], Product]):
    """
    Recibe el JSON del producto (o lista) y devuelve el análisis de ML
    conservando los datos originales más los campos scam_probability y value_score.
    """
    try:
        # Convertir a lista si es un solo producto
        if isinstance(products, Product):
            products_list = [products.dict()]
        else:
            products_list = [p.dict() for p in products]
        
        # Realizar análisis por lote
        analysis_results = engine.analyze_batch(products_list)
        
        if analysis_results is None:
            raise HTTPException(
                status_code=500, 
                detail="Los modelos no están entrenados o no se encontraron."
            )
            
        # Si se envió un solo producto, devolver un solo objeto, sino la lista
        if isinstance(products, Product):
            return analysis_results[0]
            
        return analysis_results
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
