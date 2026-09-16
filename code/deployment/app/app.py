"""
Stage 3: Deployment — Web App
==============================
Streamlit app with input fields for car attributes, a prediction
button, and an area that displays the predicted price returned by
the model API.
"""

import os
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://api:8000")

st.set_page_config(page_title="Used Car Price Prediction", page_icon="🚗")
st.title("🚗 Used Car Price Prediction")
st.write("Enter the details of the car below to get an estimated selling price.")

with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    with col1:
        brand = st.text_input("Brand", value="Maruti")
        car_age = st.number_input("Car age (years)", min_value=0, max_value=40, value=5)
        km_driven = st.number_input("Kilometers driven", min_value=0, value=45000, step=1000)
        fuel = st.selectbox("Fuel type", ["Petrol", "Diesel", "CNG", "LPG", "Electric"])
    with col2:
        seller_type = st.selectbox("Seller type", ["Individual", "Dealer", "Trustmark Dealer"])
        transmission = st.selectbox("Transmission", ["Manual", "Automatic"])

    submitted = st.form_submit_button("Predict price")

if submitted:
    payload = {
        "km_driven": km_driven,
        "car_age": car_age,
        "fuel": fuel,
        "seller_type": seller_type,
        "transmission": transmission,
        "brand": brand,
    }
    try:
        response = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
        response.raise_for_status()
        prediction = response.json()["predicted_price"]
        st.success(f"Estimated selling price: **₹{prediction:,.0f}**")
    except requests.exceptions.RequestException as e:
        st.error(f"Could not reach the prediction API: {e}")
