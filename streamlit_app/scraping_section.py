

import streamlit as st
import pandas as pd
import os

def show_scraping_section(df, DATA_LOADED):
    st.header("🌐 Scraping Process")
    st.write(f"During our initial scrape, we collected customer reviews from [Trustpilot](https://www.trustpilot.com/).")
    st.write("We utilized a Python-based scraper (code available in the 'scraper' directory of this repository) to gather publicly available review data.")

    if DATA_LOADED:
        st.subheader("DataFrame Shape from Loaded Data (from stream.py)") # Clarify which DataFrame shape is shown
        shape_code_loaded_df = f"""

  shape {df.shape}
"""
        st.code(shape_code_loaded_df, language="plaintext")
    else:
        st.warning("Data loading failed in stream.py. DataFrame shape is not available for the loaded data.")

    # --- Load a DIFFERENT CSV within scraping_section.py ---
    st.subheader("Shape of Independently Loaded Scraped Data") # New subheader

    # Define path for the DIFFERENT CSV file
    SCRAPED_CSV_FILE_NAME = 'another_trustpilot_reviews.csv'  # Name of the different CSV file
    SCRAPED_CSV_FILE_PATH = os.path.join('.', SCRAPED_CSV_FILE_NAME) # Assuming it's in the project root

    try:
        scraped_df = pd.read_csv(SCRAPED_CSV_FILE_PATH) # Load the different CSV
        SCRAPED_DATA_LOADED = True
    except FileNotFoundError:
        st.error(f"Error: Different CSV file not found at: {SCRAPED_CSV_FILE_PATH}")
        SCRAPED_DATA_LOADED = False
        scraped_df = None # Set to None in case of error
    except Exception as e:
        st.error(f"An error occurred while loading the different CSV: {e}")
        SCRAPED_DATA_LOADED = False
        scraped_df = None # Set to None in case of error


    if SCRAPED_DATA_LOADED:
        shape_code_scraped_df = f"""

  shape {scraped_df.shape}
"""
        st.code(shape_code_scraped_df, language="plaintext")
    else:
        st.warning("Independent scraped data loading failed. DataFrame shape is not available for the scraped data.")


    st.subheader("Data Sources")
    st.write("- **Trustpilot Website:** [https://www.trustpilot.com/](https://www.trustpilot.com/) - The primary source for customer reviews.")

    st.subheader("Scraping Methodology")
    st.write("Our scraper was designed to:")
    st.markdown("""
    - **Target specific categories and/or companies** (depending on the scraping script and parameters used).
    - **Extract key information** from each review, including:
        - Review text
        - Rating/Stars given
        - Date of review
        - Company/Category information (where available)
    - **Handle pagination** to collect reviews across multiple pages.
    - **Implement rate limiting and error handling** to respect Trustpilot's terms of service and ensure robust scraping.
    """)