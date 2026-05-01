import pandas as pd
import math
import joblib
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

from config import Config

# Lista de marcas sospechosas para heurística
SUSPICIOUS_BRANDS = ["abibas", "suny", "panasonig", "nokla", "rebeok", "adidass"]

class FeatureEngineer:
    def __init__(self, model_path: str = None):
        if model_path is None:
            self.model_path = Config.MODEL_PATH
        else:
            self.model_path = model_path
        self.tfidf = None
        self.shop_encoder = LabelEncoder()
        
    def _extract_heuristics(self, df: pd.DataFrame) -> pd.DataFrame:
        # 1. Variables de Texto
        df['title_length'] = df['title'].str.len()
        df['title_uppercase_ratio'] = df['title'].apply(
            lambda x: sum(1 for c in x if c.isupper()) / len(x) if len(x) > 0 else 0
        )
        df['title_has_brand_typo'] = df['title'].apply(
            lambda x: 1 if any(brand in x.lower() for brand in SUSPICIOUS_BRANDS) else 0
        )
        
        # 3. Variables Numéricas y Booleanas
        # Evitar división por cero en discount_ratio
        df['discount_ratio'] = (df['original_price'] - df['current_price']) / df['original_price'].replace(0, 1)
        df['interaction_score'] = df.apply(
            lambda x: x['rating'] * math.log1p(x['reviews_count']), axis=1
        )
        df['availability_int'] = df['availability'].astype(int)
        
        return df

    def transform(self, products_list: list, is_training=False) -> tuple[pd.DataFrame, pd.DataFrame]:
        df = pd.DataFrame(products_list)
        
        # Guardar datos originales para la respuesta final
        original_data = df.copy()
        
        # Aplicar heurísticas y derivadas
        df = self._extract_heuristics(df)
        
        # TF-IDF (Capa B)
        titles = df['title'].fillna("")
        tfidf_path = os.path.join(self.model_path, "tfidf.pkl")
        
        if is_training:
            self.tfidf = TfidfVectorizer(max_features=50, stop_words='english')
            tfidf_matrix = self.tfidf.fit_transform(titles)
            joblib.dump(self.tfidf, tfidf_path)
        else:
            if self.tfidf is None:
                if os.path.exists(tfidf_path):
                    self.tfidf = joblib.load(tfidf_path)
                else:
                    # Fallback por si no existe el modelo
                    self.tfidf = TfidfVectorizer(max_features=50, stop_words='english')
                    self.tfidf.fit(titles)
            tfidf_matrix = self.tfidf.transform(titles)
        
        tfidf_df = pd.DataFrame(
            tfidf_matrix.toarray(), 
            columns=[f"tfidf_{i}" for i in range(tfidf_matrix.shape[1])]
        )
        
        # Categorización de Tienda (Shop)
        shop_encoder_path = os.path.join(self.model_path, "shop_encoder.pkl")
        if is_training:
            df['shop_encoded'] = self.shop_encoder.fit_transform(df['shop'])
            joblib.dump(self.shop_encoder, shop_encoder_path)
        else:
            if not hasattr(self.shop_encoder, 'classes_'):
                if os.path.exists(shop_encoder_path):
                    self.shop_encoder = joblib.load(shop_encoder_path)
                else:
                    self.shop_encoder.fit(df['shop'])
            
            # Manejo de tiendas nuevas (no vistas en entrenamiento)
            known_shops = set(self.shop_encoder.classes_)
            df['shop_encoded'] = df['shop'].apply(
                lambda x: self.shop_encoder.transform([x])[0] if x in known_shops else -1
            )

        # Unir todo y descartar "Basura" (ID, JobID, ImageURL, ProductURL, RawData)
        cols_to_keep = [
            'current_price', 'original_price', 'rating', 'reviews_count',
            'title_length', 'title_uppercase_ratio', 'title_has_brand_typo',
            'discount_ratio', 'interaction_score', 'availability_int', 'shop_encoded'
        ]
        
        final_df = pd.concat([df[cols_to_keep], tfidf_df], axis=1)
        return final_df, original_data
