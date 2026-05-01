import pandas as pd
import lightgbm as lgb
import numpy as np
import joblib
import os
from app.services.features import FeatureEngineer

# Crear carpeta de modelos si no existe
os.makedirs("model_weights", exist_ok=True)

def load_and_clean_data(file_path="train_data.csv"):
    df = pd.read_csv(file_path)
    
    # Renombrar columnas para que coincidan con lo que espera FeatureEngineer
    column_mapping = {
        'Title': 'title',
        'CurrentPrice': 'current_price',
        'OriginalPrice': 'original_price',
        'Rating': 'rating',
        'ReviewsCount': 'reviews_count',
        'Availability': 'availability',
        'Shop': 'shop'
    }
    df = df.rename(columns=column_mapping)
    
    # Seleccionar solo las columnas necesarias para características y etiquetas
    # Ignoramos reasoning_scam, reasoning_value y otros campos de análisis/metadatos
    cols_to_keep = list(column_mapping.values()) + ['scam_probability', 'value_score']
    df = df[cols_to_keep]
    
    # Limpieza básica
    # Rellenar precios faltantes con 0 si es necesario, o eliminar
    df['current_price'] = pd.to_numeric(df['current_price'], errors='coerce').fillna(0)
    df['original_price'] = pd.to_numeric(df['original_price'], errors='coerce').fillna(df['current_price'])
    df['rating'] = pd.to_numeric(df['rating'], errors='coerce').fillna(0)
    df['reviews_count'] = pd.to_numeric(df['reviews_count'], errors='coerce').fillna(0)
    df['availability'] = df['availability'].fillna(False)
    
    # Eliminar filas donde las etiquetas sean NaN
    df = df.dropna(subset=['scam_probability', 'value_score'])
    
    return df

def train():
    print("Cargando datos de train_data.csv...")
    df = load_and_clean_data("train_data.csv")
    
    fe = FeatureEngineer()
    
    # Convertir a lista de diccionarios para FeatureEngineer.transform
    data_list = df.to_dict('records')
    
    print("Transformando características...")
    X, _ = fe.transform(data_list, is_training=True)
    
    y_scam = df['scam_probability'].values
    y_value = df['value_score'].values
    
    # Entrenamiento Scam (Usamos Regressor porque es una probabilidad continua)
    print("Entrenando modelo de Scam (Regresión)...")
    scam_model = lgb.LGBMRegressor(objective='regression', n_estimators=100, verbose=-1)
    scam_model.fit(X, y_scam, categorical_feature=['shop_encoded'])
    joblib.dump(scam_model, "model_weights/scam_model.lgb")
    
    # Entrenamiento Value (Regresión)
    print("Entrenando modelo de Value (Regresión)...")
    value_model = lgb.LGBMRegressor(objective='regression', n_estimators=100, verbose=-1)
    value_model.fit(X, y_value, categorical_feature=['shop_encoded'])
    joblib.dump(value_model, "model_weights/value_model.lgb")
    
    print("¡Entrenamiento completado! Archivos guardados en model_weights/")

if __name__ == "__main__":
    train()
