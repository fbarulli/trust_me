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
import category_section # ADD THIS LINE - Import category_section at the top


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
st.title("🚀 Trust Me - Supply Chain -  "  "  Customer Satisfaction")

# Sidebar Navigation
selected_section = st.sidebar.radio(
    "📌 Choose a Section:",
    ["Intro", "Scraping", "Company", "Category", "General"],
    index=0
)

# Main Content Area
if selected_section == "Intro":
    st.markdown(f"""Join us as we investigate how [Truspilot](https://www.trustpilot.com/) reviews can be used to analyze customer satisfaction.<br><br>
Our group consists of: Felix, Kjell, and Fabian, as we take different approaches to classify customer sentiment and ratings.<br><br>
Felix : single company reviews <br>
Kjell : single category <br>
Fabian : all companies/categories""", unsafe_allow_html=True)

    # --- Load and Display CSV ---
    csv_file_path = os.path.join(PARENT_DIR, 'streamlit', 'trustpilot_reviews_1000.csv') # Construct path to CSV
    try:
        df = pd.read_csv(csv_file_path)
        #st.header("Sample of Trustpilot Reviews Data:")
        random_sample_df = df[['cust_review_text']].sample(n=5)
        st.dataframe(random_sample_df.reset_index(drop=True), use_container_width=True)
    except FileNotFoundError:
        st.error(f"Error: CSV file not found at: {csv_file_path}")
    except Exception as e:
        st.error(f"An error occurred while loading the CSV: {e}")


elif selected_section == "Scraping":
    st.write(f"During our initial scrape, ")


elif selected_section == "Company":
    company_section.show_company_section() # Call function from company_section.py
elif selected_section == "General":
    general_section.show_general_section() # Call function from general_section.py
elif selected_section == "Category":
    category_section.show_category_section() # Corrected line: Call function from category_section.py (already imported at the top)

else:
    st.write(f"🚀 {selected_section} Section Selected. (Not yet customized)")