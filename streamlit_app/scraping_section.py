
# stream.py
import os
import sys
import streamlit as st
import pandas as pd
from PIL import Image
import logging
from typing import Optional

# Import section modules
import company_section
import general_section
import category_section
import scraping_section

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(module)s - %(message)s')
logger = logging.getLogger(__name__)

APP_DIR: str = os.path.dirname(os.path.abspath(__file__)) # streamlit_app directory
PARENT_DIR: str = os.path.dirname(APP_DIR) # trust-me-data-analysis directory - not needed for this specific case
sys.path.insert(0, APP_DIR) # Add streamlit_app directory to path

# --- Define CSV file path at the beginning ---
CSV_FILE_NAME: str = 'trustpilot_reviews_1000.csv' # Define the CSV file name
CSV_FILE_PATH: str = os.path.join(APP_DIR, CSV_FILE_NAME) # Corrected path to be relative to the script's directory
logger.debug(f"CSV File Path (stream.py): {CSV_FILE_PATH}")


def load_csv_data(csv_path: str) -> Optional[pd.DataFrame]:
    """Loads CSV data into a pandas DataFrame.

    Args:
        csv_path: Path to the CSV file.

    Returns:
        DataFrame if loaded successfully, None otherwise.
    """
    logger.info(f"load_csv_data: Starting for csv_path='{csv_path}'")
    if not isinstance(csv_path, str):
        logger.error(f"load_csv_data: csv_path is not a string: {type(csv_path)}")
        raise TypeError(f"csv_path must be a string, got {type(csv_path)}")

    try:
        logger.debug("load_csv_data: Attempting to load CSV data...")
        df: pd.DataFrame = pd.read_csv(csv_path)
        logger.info("load_csv_data: CSV data loaded successfully.")
        return df
    except FileNotFoundError:
        logger.error(f"load_csv_data: CSV file not found at: {csv_path}")
        st.error(f"Error: CSV file not found at: {csv_path}")
        return None
    except Exception as e:
        logger.exception(f"load_csv_data: An error occurred while loading the CSV: {e}")
        st.error(f"An error occurred while loading the CSV: {e}")
        return None
    finally:
        logger.info(f"load_csv_data: Finished for csv_path='{csv_path}'")


def display_intro_section(df: Optional[pd.DataFrame]) -> None:
    """Displays the introduction section of the Streamlit app.

    Args:
        df: Optional DataFrame to display sample data from.
    """
    logger.info("display_intro_section: Starting")
    st.markdown(f"""Join us as we investigate how [Truspilot](https://www.trustpilot.com/) reviews can be used to analyze customer satisfaction.<br><br>
Our group consists of: Felix, Kjell, and Fabian, as we take different approaches to classify customer sentiment and ratings.<br><br>
Felix : single company reviews <br>
Kjell : single category <br>
Fabian : all companies/categories""", unsafe_allow_html=True)

    if df is not None: # Proceed only if DataFrame is valid
        logger.debug("display_intro_section: DataFrame is loaded, displaying sample.")
        try:
            random_sample_df: pd.DataFrame = df[['cust_review_text']].sample(n=5)
            st.dataframe(random_sample_df.reset_index(drop=True), use_container_width=True)

            st.subheader("DataFrame Dimensions") # Add a subheader for clarity
            st.write(f"DataFrame Shape: Rows = {df.shape[0]}, Columns = {df.shape[1]}") # Display df.shape
            logger.debug("display_intro_section: Sample DataFrame and dimensions displayed.")
        except Exception as e:
            logger.exception("display_intro_section: Error displaying sample DataFrame or dimensions.")
            st.error("Error displaying data sample. Check logs for details.")
    else:
        logger.warning("display_intro_section: DataFrame is None, cannot display sample.")
        st.warning("Warning: Dataframe not loaded, intro section might be incomplete.")
    logger.info("display_intro_section: Finished")


def main() -> None:
    """Main application function."""
    logger.info("main: Application starting")
    logger.debug(f"main: Streamlit Script Working Directory: {os.getcwd()}")
    logger.debug(f"main: Python Path: {sys.path}")

    # --- Load CSV Data ---
    df: Optional[pd.DataFrame] = load_csv_data(CSV_FILE_PATH)
    DATA_LOADED: bool = df is not None # Flag to indicate data loaded successfully

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
    selected_section: str = st.sidebar.radio(
        "📌 Choose a Section:",
        ["Intro", "Scraping", "Company", "Category", "General"],
        index=0,
        key="main_sidebar_radio_stream" # Ensure unique key
    )
    logger.debug(f"main: Selected section: {selected_section}")

    # Main Content Area
    if selected_section == "Intro":
        display_intro_section(df)

    elif selected_section == "Scraping":
        logger.info("main: Navigating to Scraping section...")
        scraping_section.show_scraping_section() # Call function from scraping_section.py

    elif selected_section == "Company":
        logger.info("main: Navigating to Company section...")
        company_section.show_company_section() # Call function from company_section.py
    elif selected_section == "General":
        logger.info("main: Navigating to General section...")
        general_section.show_general_section() # Call function from general_section.py
    elif selected_section == "Category":
        logger.info("main: Navigating to Category section...")
        category_section.show_category_section() # Call function from category_section.py

    else:
        st.write(f"🚀 {selected_section} Section Selected. (Not yet customized)")
        logger.info(f"main: Selected section '{selected_section}' - Not yet customized")
    logger.info("main: Application finished")


if __name__ == "__main__":
    main()