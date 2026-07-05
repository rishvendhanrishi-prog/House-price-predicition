import streamlit as st
import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing

model = joblib.load("model.pkl")
housing = fetch_california_housing()
df = pd.DataFrame(housing.data, columns=housing.feature_names)

st.set_page_config(page_title="House Price Prediction", page_icon="🏠",layout="wide")

st.title("House Price Prediction")
st.write("Enter the house details below.")


MedInc = st.number_input("Median Income", value=5.0)
HouseAge = st.number_input("House Age", value=20.0)
AveRooms = st.number_input("Average Rooms", value=5.0)
AveBedrms = st.number_input("Average Bedrooms", value=1.0)
Population = st.number_input("Population", value=1000.0)
AveOccup = st.number_input("Average Occupancy", value=3.0)
Latitude = st.number_input("Latitude", value=34.0)
Longitude = st.number_input("Longitude", value=-118.0)

if st.button("Predict Price"):

    data = np.array([[MedInc,HouseAge,AveRooms,AveBedrms,Population,AveOccup,Latitude,Longitude]])

    prediction = model.predict(data)
    df = pd.DataFrame(housing.data, columns=housing.feature_names)
    st.bar_chart(df["MedInc"])
    st.subheader("Dataset Preview")
    st.dataframe(df.head())


    st.success(f"Estimated House Price: ${prediction[0]*100000:,.2f}")