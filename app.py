import streamlit as st
import tensorflow as tf
import pandas as pd
import numpy as np
import pickle
import os
import datetime

# Page Configuration
st.set_page_config(
    page_title="Uber Fare Prediction",
    page_icon="🚖",
    layout="centered"
)

# Relative Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "Uber_fare_model.h5")
ENCODER_PATH = os.path.join(BASE_DIR, "encoder.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")
FEATURE_ORDER_PATH = os.path.join(BASE_DIR, "feature_order.pkl")

# Distance Calculation Functions (Vectorised for single rows)
def calculate_distances(lon1, lat1, lon2, lat2):
    #  Haversine Distance
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    haversine_km = 6367 * c

    return float(haversine_km)
    

# Cache Model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


# Cache Preprocessing Objects - FIX APPLIED HERE
@st.cache_resource
def load_preprocessing():
    with open(ENCODER_PATH, "rb") as file:
        encoder = pickle.load(file)

    with open(SCALER_PATH, "rb") as file:
        scaler = pickle.load(file)

    # FIX: Using the correct defined constant path variable instead of raw string
    with open(FEATURE_ORDER_PATH, "rb") as file:
        feature_order = pickle.load(file)
    
    return encoder, scaler, feature_order


# Load everything once
model = load_model()
encoder, scaler, feature_order = load_preprocessing()



# UI Header
st.title("🚖 Uber Fare Prediction System")
st.write("Enter trip details below to calculate the estimated fare.")

st.divider()

# User Inputs Block
st.subheader("📍 Location Coordinates (NYC Boundaries)")
col1, col2 = st.columns(2)
with col1:
    pickup_latitude = st.number_input("Pickup Latitude", min_value=40.0, max_value=42.0, value=40.7383, format="%.5f")
    pickup_longitude = st.number_input("Pickup Longitude", min_value=-74.0, max_value=-72.0, value=-73.9998, format="%.5f")
with col2:
    dropoff_latitude = st.number_input("Dropoff Latitude", min_value=40.0, max_value=42.0, value=40.7232, format="%.5f")
    dropoff_longitude = st.number_input("Dropoff Longitude", min_value=-74.0, max_value=-72.0, value=-73.9995, format="%.5f")

st.divider()

st.subheader("⏰ Date, Time & Passengers")
col3, col4 = st.columns(2)
with col3:
    user_date = st.date_input("Ride Date", datetime.date(2013, 5, 7))
with col4:
    user_time = st.time_input("Ride Time", datetime.time(19, 52))

passenger_count = st.slider("Passenger Count", min_value=1, max_value=6, value=1)

st.divider()


# Predict Button
if st.button("💰 Estimate Fare", use_container_width=True):

    # 1. Feature Extraction from Date & Time
    hour = user_time.hour
    month = user_date.month
    day_of_week = user_date.weekday() # Monday=0, Sunday=6

    # 2. MATCHING YOUR IPYNB LOGIC: Create 'is_weekend' feature
    is_weekend = 1 if day_of_week >= 5 else 0

    # 3. MATCHING YOUR IPYNB LOGIC (Handling exact bins 0-5, 5-12, 12-17, 17-21, 21-24)
    if 0 <= hour < 5 or 21 <= hour < 24:
        time_slot = 'Night'
    elif 5 <= hour < 12:
        time_slot = 'Morning'
    elif 12 <= hour < 17:
        time_slot = 'Afternoon'
    else:
        time_slot = 'Evening'

    # 4. Calculate Spatial Distance Features
    distance_km = calculate_distances(
        pickup_longitude, pickup_latitude, 
        dropoff_longitude, dropoff_latitude
    )

    # 5. Continuous Features DataFrame
    continuous_data = pd.DataFrame({
        "is_weekend": [is_weekend],
        "pickup_longitude": [pickup_longitude],
        "pickup_latitude": [pickup_latitude],
        "dropoff_longitude": [dropoff_longitude],
        "dropoff_latitude": [dropoff_latitude],
        "distance_km": [distance_km]
    })

    # 6. Categorical Features Structure mapping
    categorical_cols = ['hour', 'day_of_week', 'month', 'passenger_count', 'time_slots']
    
    categorical_data = pd.DataFrame({
        'hour': [hour],
        'day_of_week': [day_of_week],
        'month': [month],
        'passenger_count': [passenger_count],
        'time_slots': [time_slot]
    })

    # Apply Master Multi-Column One-Hot Encoder
    encoded_array = encoder.transform(categorical_data)
    encoded_df = pd.DataFrame(
        encoded_array, 
        columns=encoder.get_feature_names_out(categorical_cols)
    )

    # 7. Combine Remaining Continuous Features and Encoded Features matrix
    final_input_data = pd.concat(
        [continuous_data.reset_index(drop=True), 
         encoded_df.reset_index(drop=True)], 
        axis=1
    )

    final_input_data = final_input_data[feature_order]


    # 8. Scale Features using Train Scaler
    input_scaled = scaler.transform(final_input_data)

    # 9. Model Prediction
    scaled_prediction = model.predict(input_scaled, verbose=0)
    real_fare = float(scaled_prediction[0][0])
    
    # NYC Minimum base fare boundary rule check
    if real_fare < 2.50: 
        real_fare = 2.50  

    # Display Output UI - FIX: Changed distance_km[0] to simple distance_km
    st.subheader("Estimation Result")
    
    st.success(
        f"💵 **Estimated Uber Fare: ${real_fare:.2f}**\n\n"
        f"Calculated Route Distance: **{distance_km:.2f} km**"
    )

    st.metric(
        label="Trip Price (USD)",
        value=f"${real_fare:.2f}",
        delta=f"{distance_km:.2f} km Trip"
    )
