import os
import sys
import pandas as pd
import joblib
from src.exception import CustomException
from src.utils import load_object
# Ensure your explain_instance is imported correctly
from src.components.explainability import explain_instance

class PredictPipeline:
    def __init__(self):
        self.model_path = os.path.join("artifacts", "model.pkl")
        self.preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")

    def predict(self, features):
        try:
            preprocessor = load_object(self.preprocessor_path)
            model = load_object(self.model_path)
            data_scaled = preprocessor.transform(features)
            probs = model.predict_proba(data_scaled)[:, 1]
            return (probs >= 0.3).astype(int), probs
        except Exception as e:
            raise CustomException(e, sys)

    def explain(self, features, top_n=5):
        try:
            preprocessor = load_object(self.preprocessor_path)
            model = load_object(self.model_path)
            return explain_instance(model, preprocessor, features, top_n=top_n)
        except Exception as e:
            raise CustomException(e, sys)

class CustomData:
    def __init__(self, account_length, total_day_minutes, total_eve_minutes, total_night_minutes, 
                 total_intl_minutes, customer_service_calls, number_vmail_messages, 
                 total_day_calls, total_eve_calls, total_night_calls, total_intl_calls, 
                 international_plan, voice_mail_plan, area_code):
        
        self.data = {
            "account_length": [account_length],
            "total_day_minutes": [total_day_minutes],
            "total_eve_minutes": [total_eve_minutes],
            "total_night_minutes": [total_night_minutes],
            "total_intl_minutes": [total_intl_minutes],
            "customer_service_calls": [customer_service_calls],
            "number_vmail_messages": [number_vmail_messages],
            "total_day_calls": [total_day_calls],
            "total_eve_calls": [total_eve_calls],
            "total_night_calls": [total_night_calls],
            "total_intl_calls": [total_intl_calls],
            "international_plan": [international_plan],
            "voice_mail_plan": [voice_mail_plan],
            "area_code": [area_code]
        }

    def get_data_as_dataframe(self):
        return pd.DataFrame(self.data)