# model.py
import joblib
import numpy as np

def load_model():
    # Load the pre-trained model from file
    model = joblib.load(r"D:\Internpe Internship Project\Car_Price_Prediction\model.py")  # Adjust path to your actual model file
    return model

def predict_price(model, features):
    # Pre-process features if necessary
    processed_features = preprocess_features(features)
    prediction = model.predict(processed_features)
    return prediction[0]

def preprocess_features(features):
    # Implement any preprocessing steps here if necessary
    # For example, encoding categorical variables
    # This function should match the preprocessing steps used during model training
    return features
