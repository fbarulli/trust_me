# scraping_section.py
import streamlit as st
import pandas as pd
import os
import logging
from typing import Optional

# Set up logging for this module
logger = logging.getLogger(__name__)

# --- CORRECTED FILE NAME and PATH ---
SCRAPED_CSV_FILE_NAME: str = 'trustpilot_reviews.csv'
SCRAPED_CSV_FILE_PATH: str = os.path.join('..', 'fabian', 'EDA', SCRAPED_CSV_FILE_NAME) # Corrected relative path
# --- END CORRECTIONS ---


def show_scraping_section() -> None:
    """Displays the scraping section in the Streamlit app, including details about the scraping process and displaying the shape of an independently loaded scraped dataset."""
    logger.info("show_scraping_section: Starting")  # Log function start
    logger.debug(f"show_scraping_section: Constructed CSV path: {SCRAPED_CSV_FILE_PATH}") # Log the constructed path

    st.header("🌐 Scraping Process")
    st.write(f"During our initial scrape we collected:")

    #st.subheader("Shape of Independently Loaded Scraped Data") # New subheader

    # --- Load a DIFFERENT CSV within scraping_section.py ---
    logger.info(f"show_scraping_section: Attempting to load scraped data from: {SCRAPED_CSV_FILE_PATH}")
    scraped_df: Optional[pd.DataFrame] = None  # Initialize scraped_df with a default value and type hint
    SCRAPED_DATA_LOADED: bool = False # Initialize flag

    try:
        scraped_df = pd.read_csv(SCRAPED_CSV_FILE_PATH) # Load the different CSV
        SCRAPED_DATA_LOADED = True
        logger.info(f"show_scraping_section: Successfully loaded scraped data from: {SCRAPED_CSV_FILE_PATH}")
    except FileNotFoundError:
        st.error(f"Error: Different CSV file not found at: {SCRAPED_CSV_FILE_PATH}")
        SCRAPED_DATA_LOADED = False
        scraped_df = None # Set to None in case of error
        logger.warning(f"show_scraping_section: File not found error for scraped data CSV: {SCRAPED_CSV_FILE_PATH}")
    except Exception as e:
        st.error(f"An error occurred while loading the different CSV: {e}")
        SCRAPED_DATA_LOADED = False
        scraped_df = None # Set to None in case of error
        logger.exception(f"show_scraping_section: An error occurred while loading scraped CSV: {e}", exc_info=True) # Log exception with traceback


    if SCRAPED_DATA_LOADED:
        if scraped_df is not None: # Check if DataFrame is valid before accessing shape
            shape_code_scraped_df: str = f"""

  shape {scraped_df.shape}
"""
            st.code(shape_code_scraped_df, language="plaintext")
            logger.debug(f"show_scraping_section: Displayed shape of scraped DataFrame: {scraped_df.shape}")

            if "company" in scraped_df.columns: # Check if 'company' column exists
                unique_companies_count = scraped_df["company"].nunique()
                st.write(f"And we found reviews for **{unique_companies_count}** unique companies.")
                logger.debug(f"show_scraping_section: Displayed unique company count: {unique_companies_count}")
            else:
                st.warning("Column 'company' not found in the scraped data. Cannot display unique company count.")
                logger.warning("show_scraping_section: Column 'company' not found in scraped data.")

            # --- Display DataFrame Columns instead of Scraping Methodology ---
            st.subheader("Extracted Data Fields:")
            if scraped_df.columns.tolist(): # Check if column list is not empty
                columns_list = scraped_df.columns.tolist()
                st.markdown(" - " + "\n - ".join(columns_list)) # Format as bullet points
                logger.debug(f"show_scraping_section: Displayed DataFrame columns: {columns_list}")
            else:
                st.warning("Could not retrieve column names from the scraped data.")
                logger.warning("show_scraping_section: Could not retrieve column names from scraped_df.")
            # --- END Column Display Section ---


        else:
            st.warning("Scraped data loaded flag is True, but DataFrame is unexpectedly None.") # Defensive check
            logger.warning("show_scraping_section: SCRAPED_DATA_LOADED is True, but scraped_df is None.")

    else:
        st.warning("Independent scraped data loading failed. DataFrame shape is not available for the scraped data.")
        logger.warning("show_scraping_section: Independent scraped data loading failed.")


    #st.subheader("Data Sources")
    #st.write("- **Trustpilot Website:** [https://www.trustpilot.com/](https://www.trustpilot.com/) - The primary source for customer reviews.")

    st.subheader("Scraping Methodology") # Keep the subheader "Scraping Methodology"
    st.write("Our scraper was designed to (general description remains):") # Keep general description
    st.markdown("""
    - **Target specific categories and/or companies** (depending on the scraping script and parameters used).
    - **Extract key information from each review, including:**
        - Review text
        - Rating/Stars given
        - Date of review
        - Company/Category information (where available)
    - **Handle pagination** to collect reviews across multiple pages.
    - **Implement rate limiting and error handling** to respect Trustpilot's terms of service and ensure robust scraping.
    """)
    logger.info("show_scraping_section: Finished") # Log function finish


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG) # Configure basic logging if running standalone
    show_scraping_section() # Test function