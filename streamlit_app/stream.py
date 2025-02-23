
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
import category_section
import scraping_section


# --- Define CSV file path at the beginning ---
CSV_FILE_NAME = 'trustpilot_reviews_1000.csv' # Define the CSV file name
CSV_FILE_PATH = os.path.join('.', CSV_FILE_NAME) # Relative path from the project root (where stream.py is assumed to be run)
print(f"CSV File Path (stream.py): {CSV_FILE_PATH}") # Print path for debugging


# --- Load CSV Data (Load it once at the top) ---
try:
    df = pd.read_csv(CSV_FILE_PATH)
    DATA_LOADED = True # Flag to indicate data loaded successfully
except FileNotFoundError:
    st.error(f"Error: CSV file not found at: {CSV_FILE_PATH}")
    DATA_LOADED = False
except Exception as e:
    st.error(f"An error occurred while loading the CSV: {e}")
    DATA_LOADED = False


# --- Custom Styling ---
st.markdown(
    """
    <style>
    /* ... (Your CSS Styling - no changes needed) ... */
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

    if DATA_LOADED: # Only proceed if data was loaded successfully
        #st.header("Sample of Trustpilot Reviews Data:") # REMOVED - No header for sample table
        random_sample_df = df[['cust_review_text']].sample(n=5)
        st.dataframe(random_sample_df.reset_index(drop=True), use_container_width=True)

        st.subheader("DataFrame Dimensions") # Add a subheader for clarity # ADDED
        st.write(f"DataFrame Shape: Rows = {df.shape[0]}, Columns = {df.shape[1]}") # Display df.shape # ADDED


elif selected_section == "Scraping":
    scraping_section.show_scraping_section(df, DATA_LOADED) # Call function from scraping_section.py, passing df and DATA_LOADED


elif selected_section == "Company":
    company_section.show_company_section() # Call function from company_section.py
elif selected_section == "General":
    general_section.show_general_section() # Call function from general_section.py
elif selected_section == "Category":
    category_section.show_category_section() # Corrected line: Call function from category_section.py (already imported at the top)

else:
    st.write(f"🚀 {selected_section} Section Selected. (Not yet customized)")