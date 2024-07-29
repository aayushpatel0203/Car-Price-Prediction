import streamlit as st
import pickle
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# Load the model and data with error handling
@st.cache(allow_output_mutation=True)
def load_model_and_data():
    try:
        model = pickle.load(open('LinearRegressionModel.pkl', 'rb'))
        car = pd.read_csv('Cleaned_Car_data.csv')
        return model, car
    except (FileNotFoundError, IOError):
        st.error("Required files are not found. Please ensure the model and data files are present.")
        st.stop()

# Create a title for the app
def create_title():
    st.title("Car Price Prediction")

# Create select boxes for company, car model, year, and fuel type
def create_select_boxes(car):
    companies = sorted(car['company'].unique())
    car_models = sorted(car['name'].unique())
    years = sorted(car['year'].unique(), reverse=True)
    fuel_types = car['fuel_type'].unique()

    company = st.selectbox("Select Company", companies)
    filtered_car_models = car[car['company'] == company]['name'].unique()
    car_model = st.selectbox("Select Car Model", filtered_car_models)
    year = st.selectbox("Select Year", years)
    fuel_type = st.selectbox("Select Fuel Type", fuel_types)
    return company, car_model, year, fuel_type

# Create a slider for driven kilometers
def create_slider():
    driven = st.slider("Kilometers Driven", min_value=0, max_value=100000, step=1000)
    return driven

# Create a button to make prediction
def create_button(model, company, car_model, year, driven, fuel_type):
    if st.button("Predict"):
        try:
            prediction = model.predict(pd.DataFrame(columns=['name', 'company', 'year', 'kms_driven', 'fuel_type'],
                                                    data=np.array([car_model, company, year, driven, fuel_type]).reshape(1, 5)))
            formatted_prediction = f"{prediction[0]:,.2f}"
            st.write(f"Predicted Price: {formatted_prediction}")
        except Exception as e:
            st.error(f"Error in making prediction: {e}")

def main():
    model, car = load_model_and_data()
    create_title()
    company, car_model, year, fuel_type = create_select_boxes(car)
    driven = create_slider()
    create_button(model, company, car_model, year, driven, fuel_type)

if __name__ == "__main__":
    main()