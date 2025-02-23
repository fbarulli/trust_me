

import streamlit as st
import pandas as pd # Import pandas if needed in this section
import os # Import os if needed in this section

# --- CSV file path is NOT defined here ---
# This module receives the DataFrame 'df' as an argument, it does not load the CSV file directly.
# The CSV loading is handled in stream.py

def show_scraping_section(df, DATA_LOADED): # Pass df and DATA_LOADED as arguments
    st.header("🌐 Scraping Process") # Add a header for the section
    st.write(f"During our initial scrape, we collected customer reviews from [Trustpilot](https://www.trustpilot.com/).") # General description
    st.write("We utilized a Python-based scraper (code available in the 'scraper' directory of this repository) to gather publicly available review data.") # Mention scraper and location

    if DATA_LOADED: # Conditionally display df.shape if data is loaded
        st.subheader("DataFrame Shape from Scraped Data") # Subheader for DataFrame Shape
        # Display df.shape as code block using st.code()
        shape_code = f"""

  shape {df.shape}
"""
        st.code(shape_code, language="plaintext") # Display df.shape as code
    else:
        st.warning("Data loading failed. DataFrame shape is not available.") # Warning if data loading failed

    st.subheader("Data Sources") # Subheader for data sources
    st.write("- **Trustpilot Website:** [https://www.trustpilot.com/](https://www.trustpilot.com/) - The primary source for customer reviews.") # Link to Trustpilot

    st.subheader("Scraping Methodology") # Subheader for methodology
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
    """) # Details about the scraping process