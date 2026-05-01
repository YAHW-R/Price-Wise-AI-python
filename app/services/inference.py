import joblib
import os
import pandas as pd
from app.services.features import FeatureEngineer

class AnalysisEngine:
    def __init__(self, model_path="model_weights/"):
        self.model_path = model_path
        self.fe = FeatureEngineer(model_path)
        self.scam_model = None
        self.value_model = None
        self.is_loaded = False
        self.load_models()

    def load_models(self):
        """Carga los modelos desde el disco a la memoria (ROM simulada)"""
        scam_path = os.path.join(self.model_path, "scam_model.lgb")
        value_path = os.path.join(self.model_path, "value_model.lgb")
        
        if os.path.exists(scam_path) and os.path.exists(value_path):
            try:
                self.scam_model = joblib.load(scam_path)
                self.value_model = joblib.load(value_path)
                self.is_loaded = True
                print("Modelos LightGBM cargados correctamente.")
            except Exception as e:
                print(f"Error cargando modelos: {e}")
        else:
            print(f"Modelos no encontrados en {self.model_path}. Por favor, ejecuta train_model.py primero.")

    def analyze_batch(self, products_list: list):
        """Realiza el análisis de una lista de productos"""
        if not self.is_loaded:
            # Intentar cargar de nuevo por si se generaron después de iniciar la app
            self.load_models()
            if not self.is_loaded:
                return None

        # 1. Normalización y Feature Engineering
        X, original_df = self.fe.transform(products_list, is_training=False)
        
        # 2. Inferencia
        scam_probs = self.scam_model.predict(X)
        value_scores = self.value_model.predict(X)
        
        # 3. Formatear resultados (Datos originales + campos nuevos)
        results = []
        for i, row in original_df.iterrows():
            product_res = row.to_dict()
            product_res["scam_probability"] = float(scam_probs[i])
            product_res["value_score"] = float(value_scores[i])
            results.append(product_res)
            
        return results

# Singleton para uso en toda la aplicación
engine = AnalysisEngine()
