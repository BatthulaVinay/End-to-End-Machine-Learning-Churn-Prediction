import streamlit as st
import pandas as pd
from src.pipeline.predict_pipeline import PredictPipeline, CustomData

# Page Config
st.set_page_config(page_title="Customer Churn Predictor", layout="centered")

@st.cache_resource
def get_predictor():
    return PredictPipeline()

predictor = get_predictor()

st.title("📊 Customer Churn Prediction")
st.markdown("Enter customer details to predict churn.")

with st.form("churn_form"):
    account_length = st.number_input("Account Length", min_value=1)
    total_day_minutes = st.number_input("Total Day Minutes", min_value=0.0)
    total_eve_minutes = st.number_input("Total Evening Minutes", min_value=0.0)
    total_night_minutes = st.number_input("Total Night Minutes", min_value=0.0)
    total_intl_minutes = st.number_input("Total International Minutes", min_value=0.0)
    customer_service_calls = st.number_input("Customer Service Calls", min_value=0)
    number_vmail_messages = st.number_input("Voicemail Messages", min_value=0)
    total_day_calls = st.number_input("Total Day Calls", min_value=0)
    total_eve_calls = st.number_input("Total Evening Calls", min_value=0)
    total_night_calls = st.number_input("Total Night Calls", min_value=0)
    total_intl_calls = st.number_input("Total International Calls", min_value=0)
    international_plan = st.selectbox("International Plan", ["yes", "no"])
    voice_mail_plan = st.selectbox("Voice Mail Plan", ["yes", "no"])
    area_code = st.selectbox("Area Code", [408, 415, 510])
    
    submit = st.form_submit_button("Predict Churn")

if submit:
    # Prepare data using your existing pipeline logic
    data = CustomData(
        account_length=account_length,
        total_day_minutes=total_day_minutes,
        total_eve_minutes=total_eve_minutes,
        total_night_minutes=total_night_minutes,
        total_intl_minutes=total_intl_minutes,
        customer_service_calls=customer_service_calls,
        number_vmail_messages=number_vmail_messages,
        total_day_calls=total_day_calls,
        total_eve_calls=total_eve_calls,
        total_night_calls=total_night_calls,
        total_intl_calls=total_intl_calls,
        international_plan=international_plan,
        voice_mail_plan=voice_mail_plan,
        area_code=area_code
    )
    
    df = data.get_data_as_dataframe()
    
    # Predict directly
    predictions, probabilities = predictor.predict(df)
    explanation = predictor.explain(df, top_n=5)
    
    # Display Results
    if predictions[0] == 1:
        st.error("🚨 Customer is likely to churn")
    else:
        st.success("✅ Customer is unlikely to churn")
        
    st.write(f"**Churn Probability:** {round(float(probabilities[0]), 4)}")
    st.write("### Explanation")
    st.json(explanation)