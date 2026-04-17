import random

# Por ahora simulamos el modelo LightGBM
# Más adelante aquí cargaríamos el .pkl de model_weights/

def predict_scam_probability(features: dict) -> float:
    """Simula la probabilidad de estafa basada en las características."""
    # Simular una lógica (si el descuento es muy alto, mayor riesgo)
    if features.get("discount_percentage", 0) > 0.8:
        return random.uniform(0.7, 0.95)
    return random.uniform(0.01, 0.3)

def calculate_value_score(features: dict) -> float:
    """Simula el puntaje de valor calidad/precio."""
    # Mayor descuento y mayor confianza, mayor puntaje
    discount = features.get("discount_percentage", 0)
    trust = features.get("trust_score", 0)
    
    score = (discount * 0.5) + (min(trust / 10.0, 1.0) * 0.5)
    return min(max(score, 0.0), 1.0)
