import os
import sys
import pandas as pd
from src.logger import logging
from src.exception import CustomException
from src.utils import load_object
from src.components.explainability import explain_instance

class PredictPipeline:
    def __init__(self):
        self.model_path = os.path.join("artifacts", "model.pkl")
        self.preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")

    def predict(self, features: pd.DataFrame):
        try:
            preprocessor = load_object(self.preprocessor_path)
            model = load_object(self.model_path)
            
            # Ensure features are transformed before prediction
            data_transformed = preprocessor.transform(features)
            probs = model.predict_proba(data_transformed)[:, 1]
            return (probs >= 0.3).astype(int), probs
        except Exception as e:
            raise CustomException(e, sys)

    def explain(self, features: pd.DataFrame, top_n: int = 5):
        try:
            preprocessor = load_object(self.preprocessor_path)
            model = load_object(self.model_path)
            # Pass the preprocessor to the explain_instance function
            return explain_instance(model, preprocessor, features, top_n=top_n)
        except Exception as e:
            raise CustomException(e, sys)

class CustomData:
    def __init__(self, account_length, total_day_minutes, total_eve_minutes, 
                 total_night_minutes, total_intl_minutes, customer_service_calls, 
                 number_vmail_messages, total_day_calls, total_eve_calls, 
                 total_night_calls, total_intl_calls, international_plan, 
                 voice_mail_plan, area_code):
        
        # Keys here MUST match your training column names exactly
        self.data = {
            "Account length": [account_length],
            "Total day minutes": [total_day_minutes],
            "Total eve minutes": [total_eve_minutes],
            "Total night minutes": [total_night_minutes],
            "Total intl minutes": [total_intl_minutes],
            "Customer service calls": [customer_service_calls],
            "Number vmail messages": [number_vmail_messages],
            "Total day calls": [total_day_calls],
            "Total eve calls": [total_eve_calls],
            "Total night calls": [total_night_calls],
            "Total intl calls": [total_intl_calls],
            "International plan": [str(international_plan).title()],
            "Voice mail plan": [str(voice_mail_plan).title()],
            "Area code": [area_code]
        }

    def get_data_as_dataframe(self):
        return pd.DataFrame(self.data)