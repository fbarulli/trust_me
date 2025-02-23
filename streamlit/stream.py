import os
import sys
APP_DIR = os.path.dirname(os.path.abspath(__file__)) # streamlit_app directory
PARENT_DIR = os.path.dirname(APP_DIR) # trust-me-data-analysis directory - not needed for this specific case
sys.path.insert(0, APP_DIR) # Add streamlit_app directory to path

print("Streamlit Script Working Directory:", os.getcwd())
print("Python Path (inside stream.py - after path modification):", sys.path) # Check PATH after modification

import streamlit as st
import pandas as pd
from PIL import Image

import company_section
import general_section

# --- Custom Styling ---
st.markdown(
    """
    <style>
    /* Force white background */
    html, body, [data-testid="stAppViewContainer"], .stApp {
        background-color: white !important;
        color: black !important;
    }

    /* Ensure all text is visible */
    .stText, .stTitle, .stHeader, .stMarkdown, .stDataFrame, .stTable {
        color: black !important;
    }

    /* Sidebar */
    .css-1d391kg {
        background-color: #0e76a8 !important;
        color: white !important;
    }

    .css-1d391kg h1 {
        color: white !important;
    }

    /* Radio Buttons Styling */
    .st-bx input[type="radio"]:checked + div[data-baseweb="radio"] > div {
        background-color: #0e76a8 !important;
        color: #ffffff !important;
    }

    /* Card-Like Content Box */
    .content-box {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- App Structure ---
st.title("🚀 Trust Me - Data Analysis")

# Sidebar Navigation
selected_section = st.sidebar.radio(
    "📌 Choose a Section:",
    ["Intro", "Scraping", "Company", "Category", "General"],
    index=0
)

# Main Content Area
if selected_section == "Intro":
    st.write(f"🚀 Intro Section Selected. (Content for Intro will go here)")
elif selected_section == "Scraping":
    st.write(f"🚀 Scraping Section Selected. (Content for Scraping will go here)")
else:
    st.write(f"🚀 {selected_section} Section Selected. (Not yet customized)")