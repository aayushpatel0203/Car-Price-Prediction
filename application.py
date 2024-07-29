import streamlit as st
import pickle
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# Load the model and data with error handling
try:
    model = pickle.load(open('LinearRegressionModel.pkl', 'rb'))
    car = pd.read_csv('Cleaned_Car_data.csv')
except (FileNotFoundError, IOError):
    st.error("Required files are not found. Please ensure the model and data files are present.")
    st.stop()

# Create a title for the app
st.title("Car Price Prediction")

# Create select boxes for company, car model, year, and fuel type
companies = sorted(car['company'].unique())
car_models = sorted(car['name'].unique())
years = sorted(car['year'].unique(), reverse=True)
fuel_types = car['fuel_type'].unique()

company = st.selectbox("Select Company", companies)

# Filter car models based on the selected company
filtered_car_models = car[car['company'] == company]['name'].unique()
car_model = st.selectbox("Select Car Model", filtered_car_models)

year = st.selectbox("Select Year", years)
fuel_type = st.selectbox("Select Fuel Type", fuel_types)

# Create a slider for driven kilometers
driven = st.slider("Kilometers Driven", min_value=0, max_value=100000, step=1000)

# Create a button to make prediction
if st.button("Predict"):
    # Make prediction using the model
    try:
        prediction = model.predict(pd.DataFrame(columns=['name', 'company', 'year', 'kms_driven', 'fuel_type'],
                                                data=np.array([car_model, company, year, driven, fuel_type]).reshape(1, 5)))
        
        # Format the prediction with commas
        formatted_prediction = f"{prediction[0]:,.2f}"
        
        # Display the prediction
        st.write(f"Predicted Price: {formatted_prediction}")
    except Exception as e:
        st.error(f"Error in making prediction: {e}")