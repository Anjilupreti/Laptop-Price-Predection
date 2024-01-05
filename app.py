import streamlit as st
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load the pipeline from the pickle file
pipe = pickle.load(open('pipe.pkl', 'rb'))

# Load any necessary data for input features
df = pd.read_pickle('df.pkl')

st.title("Laptop Price Prediction App")

# Collect user inputs
company = st.selectbox('Brand', df['Company'].unique())
type = st.selectbox('Type', df['TypeName'].unique())
ram = st.selectbox('RAM (in GB)', [2, 4, 6, 8, 12, 16, 24, 32, 64])
weight = st.number_input('Weight of the Laptop')
touchscreen = st.selectbox('Touchscreen', ['No', 'Yes'])
ips = st.selectbox('IPS', ['No', 'Yes'])
screen_size = st.number_input('Screen Size')
resolution = st.selectbox('Screen Resolution', ['1920x1080', '1366x768', '1600x900', '3840x2160', '3200x1800', '2880x1800', '2560x1600', '2560x1440', '2304x1440'])
cpu = st.selectbox('CPU', df['Cpu brand'].unique())
hdd = st.selectbox('HDD (in GB)', [0, 128, 256, 512, 1024, 2048])
ssd = st.selectbox('SSD (in GB)', [0, 8, 128, 256, 512, 1024])
gpu = st.selectbox('GPU', df['Gpu brand'].unique())
os = st.selectbox('OS', df['os'].unique())

if st.button('Predict Price', key='predict_button'):
    # Encode categorical variables
    label_encoder = LabelEncoder()

    company = label_encoder.fit_transform([company])[0]
    type = label_encoder.fit_transform([type])[0]
    cpu = label_encoder.fit_transform([cpu])[0]
    gpu = label_encoder.fit_transform([gpu])[0]
    os = label_encoder.fit_transform([os])[0]

    # Convert binary categorical features to numerical representation
    touchscreen = 1 if touchscreen == 'Yes' else 0
    ips = 1 if ips == 'Yes' else 0

    # Calculate PPI
    X_res, Y_res = map(float, resolution.split('x'))

    # Check if screen_size is zero before performing the division
    if screen_size == 0:
        st.warning("Screen size cannot be zero. Please enter a valid screen size.")
        st.stop()

    # Explicitly convert values to float before calculation
    X_res, Y_res, screen_size = map(float, [X_res, Y_res, screen_size])

    # Check for NaN values in the query array
    if np.isnan([X_res, Y_res, screen_size]).any():
        st.warning("NaN values detected in the input data. Please check your input values.")
        st.stop()

    # Calculate PPI
    ppi = ((X_res ** 2) + (Y_res ** 2)) ** 0.5 / screen_size

    # Create the query array
    query = np.array([company, type, ram, weight, touchscreen, ips, ppi, cpu, hdd, ssd, gpu, os]).reshape(1, -1)

    # Print and inspect the query array
    print("Query array:", query)

    try:
        # Make the prediction
        predicted_price = pipe.predict(query)[0]
        st.success(f"The predicted price of this configuration is ${predicted_price:.2f}")
    except Exception as e:
        st.error(f"An error occurred: {e}")
        print(f"Exception details: {e}")
