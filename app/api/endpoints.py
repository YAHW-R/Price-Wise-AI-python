from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.services.features import calculate_features
from app.services.inference import predict_scam_probability, calculate_value_score

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
    product_url: str

@router.post("/analyze")
async def analyze_product(product: Product):
    """
    Recibe el JSON del producto y devuelve el análisis de ML.
    """
    try:
        product_dict = product.dict()
        
        # 1. Feature Engineering
        features = calculate_features(product_dict)
        
        # 2. Inferencia (Mock por ahora)
        scam_prob = predict_scam_probability(features)
        value_score = calculate_value_score(features)
        
        return {
            "scam_probability": round(scam_prob, 2),
            "value_score": round(value_score, 2),
            "features_used": {
                "discount_percentage": round(features["discount_percentage"], 2),
                "trust_score": round(features["trust_score"], 2)
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
