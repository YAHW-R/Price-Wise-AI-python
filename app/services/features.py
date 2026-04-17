import math

def calculate_features(product_dict: dict) -> dict:
    """
    Calcula nuevas variables para el análisis del producto:
    - discount_percentage: Diferencia porcentual entre original y actual.
    - trust_score: Basado en rating y número de reviews.
    """
    current_price = product_dict.get("current_price", 0)
    original_price = product_dict.get("original_price", current_price)
    rating = product_dict.get("rating", 0)
    reviews_count = product_dict.get("reviews_count", 0)

    # Evitar división por cero
    if original_price > 0:
        discount_percentage = (original_price - current_price) / original_price
    else:
        discount_percentage = 0.0

    # Trust score: Rating * log(ReviewsCount + 1)
    trust_score = rating * math.log1p(reviews_count)

    return {
        "discount_percentage": discount_percentage,
        "trust_score": trust_score
    }
