import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Real Estate Price Predictor",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
    <style>
    /* Main container styling */
    .main {
        background-color: #f8fafc;
    }
    
    /* Header styling */
    .main-title {
        font-size: 2.3rem;
        font-weight: 800;
        background: linear-gradient(90deg, #4f46e5, #06b6d4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        color: #64748b;
        font-size: 1.05rem;
        margin-bottom: 2rem;
    }

    /* Result Card */
    .metric-card {
        background: linear-gradient(135deg, #4f46e5 0%, #3b82f6 50%, #06b6d4 100%);
        padding: 24px;
        border-radius: 16px;
        color: white;
        box-shadow: 0 10px 25px -5px rgba(59, 130, 246, 0.4);
        text-align: center;
        margin-top: 1.5rem;
    }
    .metric-title {
        font-size: 1.1rem;
        font-weight: 500;
        opacity: 0.9;
        margin-bottom: 0.5rem;
    }
    .metric-value {
        font-size: 2.6rem;
        font-weight: 800;
        letter-spacing: -0.5px;
    }
    
    /* Custom button style */
    .stButton > button {
        width: 100%;
        background: linear-gradient(90deg, #4f46e5, #06b6d4);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 1.5rem;
        font-size: 1.1rem;
        font-weight: 600;
        transition: all 0.2s ease-in-out;
    }
    .stButton > button:hover {
        opacity: 0.95;
        transform: translateY(-1px);
    }
    </style>
""", unsafe_allow_html=True)

# Load model
@st.cache_resource
def load_model():
    with open("Linear.pkl", "rb") as file:
        model = pickle.load(file)
    return model

try:
    model = load_model()
except FileNotFoundError:
    st.error("⚠️ `Linear.pkl` file not found in the current directory. Please place it in the same folder as `app.py`.")
    st.stop()

# Header
st.markdown('<div class="main-title">🏡 Smart Home Valuation</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Estimate property market values accurately using trained linear regression.</div>', unsafe_allow_html=True)

# Form Layout
with st.form("prediction_form"):
    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown("### 📐 **Property Dimensions**")
        square_footage = st.number_input(
            "Square Footage (sq ft)", 
            min_value=200, 
            max_value=15000, 
            value=2200, 
            step=50
        )
        lot_size = st.number_input(
            "Lot Size (sq ft)", 
            min_value=500, 
            max_value=50000, 
            value=5000, 
            step=100
        )
        garage_size = st.number_input(
            "Garage Capacity (Number of cars)", 
            min_value=0, 
            max_value=6, 
            value=2, 
            step=1
        )
        year_built = st.number_input(
            "Year Built", 
            min_value=1850, 
            max_value=2030, 
            value=2015, 
            step=1
        )

    with col2:
        st.markdown("### 🛋️ **Interior & Quality**")
        num_bedrooms = st.slider(
            "Number of Bedrooms", 
            min_value=1, 
            max_value=10, 
            value=3, 
            step=1
        )
        num_bathrooms = st.slider(
            "Number of Bathrooms", 
            min_value=1.0, 
            max_value=8.0, 
            value=2.0, 
            step=0.5
        )
        neighborhood_quality = st.select_slider(
            "Neighborhood Quality Rating (1 = Low, 10 = Prime)", 
            options=list(range(1, 11)), 
            value=7
        )

    submit_btn = st.form_submit_button("Calculate Estimated Value")

# Prediction logic
if submit_btn:
    features = pd.DataFrame([{
        "Square_Footage": square_footage,
        "Num_Bedrooms": num_bedrooms,
        "Num_Bathrooms": num_bathrooms,
        "Year_Built": year_built,
        "Lot_Size": lot_size,
        "Garage_Size": garage_size,
        "Neighborhood_Quality": neighborhood_quality
    }])

    prediction = model.predict(features)[0]
    formatted_price = f"${prediction:,.2f}" if prediction >= 0 else "$0.00"

    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">ESTIMATED PROPERTY VALUATION</div>
            <div class="metric-value">{formatted_price}</div>
        </div>
    """, unsafe_allow_html=True)

    with st.expander("🔍 View Raw Input Vector"):
        st.dataframe(features, use_container_width=True)
