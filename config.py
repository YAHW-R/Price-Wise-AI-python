import os

class Config: 
    MODEL_PATH = os.environ.get('MODEL_PATH', 'model_weights/')