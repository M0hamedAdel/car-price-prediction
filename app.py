import streamlit as st
import pickle
import numpy as np

model = pickle.load(open('car_price_model.pkl', 'rb'))

st.title("Car Price Prediction")

year = st.number_input("Year", 2000, 2025)
present_price = st.number_input("Present Price")
kms_driven = st.number_input("Kilometers Driven")

fuel = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG"])
seller = st.selectbox("Seller Type", ["Dealer", "Individual"])
transmission = st.selectbox("Transmission", ["Manual", "Automatic"])

owner = st.selectbox("Owner", [0, 1, 2, 3])

fuel_diesel = 1 if fuel == "Diesel" else 0
fuel_petrol = 1 if fuel == "Petrol" else 0

seller_individual = 1 if seller == "Individual" else 0

trans_manual = 1 if transmission == "Manual" else 0

features = np.array([[

    year,
    present_price,
    kms_driven,
    owner,
    fuel_diesel,
    fuel_petrol,
    seller_individual,
    trans_manual

]])

if st.button("Predict Price"):

    prediction = model.predict(features)

    st.success(f"Estimated Selling Price: ₹ {prediction[0]:,.2f}")