import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load the model and vectorizer
model = pickle.load(open('furniture_model.pkl', 'rb'))
tfidf = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))

st.set_page_config(page_title="Furniture Sales Predictor", layout="centered")
st.title("E-commerce Furniture Sales Predictor: Developed by Unni R")
st.write("Enter the details below to forecast potential sales volume.")

# Sidebar for inputs
with st.form("prediction_form"):
    title = st.text_input("Product Title", placeholder="e.g., Luxury Velvet Sofa Set")
    price = st.number_input("Price (USD)", min_value=1.0, value=150.0)
    orig_price = st.number_input("Original Price (USD)", min_value=1.0, value=200.0)
    
    submit = st.form_submit_button("Predict Sales")

if submit:
    # 1. Text Processing
    title_tfidf = tfidf.transform([title]).toarray()
    
    # 2. Manual Feature Engineering (Matching your 156 features)
    price_log = np.log1p(price)
    discount_pct = (orig_price - price) / orig_price if orig_price > 0 else 0
    is_set = 1 if any(x in title.lower() for x in ['set', 'pcs', 'pack', 'piece']) else 0
    price_title_ratio = len(title) / price_log
    
    # Constructing the numeric vector (Ensure order matches training)
    # [price_log, discount_pct, shipping_cost(0), has_orig_price(1), price_title_ratio, is_set]
    numeric_features = np.array([[price_log, discount_pct, 0.0, 1, price_title_ratio, is_set]])
    
    # 3. Final Input Assembly
    final_input = np.hstack((numeric_features, title_tfidf))
    
    # 4. Prediction
    prediction_log = model.predict(final_input)
    prediction = np.expm1(prediction_log)
    
    st.snow()
    st.success(f"### Predicted Total Sales: **{int(prediction[0])} Units**")
