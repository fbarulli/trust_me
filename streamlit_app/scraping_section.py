# scraping_section.py
import streamlit as st
import pandas as pd
import os
import logging
from typing import Optional
from general_section import display_resized_image # Changed import to general_section

logger = logging.getLogger(__name__)

APP_DIR: str = os.path.dirname(os.path.abspath(__file__))

SCRAPED_CSV_FILE_NAME: str = 'trustpilot_reviews.csv'
SCRAPED_CSV_FILE_PATH: str = os.path.join('..', 'fabian', 'EDA', SCRAPED_CSV_FILE_NAME)


def show_scraping_section() -> None:
    logger.info("show_scraping_section: Starting")
    logger.debug(f"show_scraping_section: Constructed CSV path: {SCRAPED_CSV_FILE_PATH}")

    st.header("🌐 Initial Scrape")
    st.write(f"During our initial scrape we collected:")


    hardcoded_shape = (140124, 9)
    shape_code_scraped_df: str = f"""
  shape {hardcoded_shape}
"""
    st.code(shape_code_scraped_df, language="plaintext")
    logger.debug(f"show_scraping_section: Displayed hardcoded shape: {hardcoded_shape}")
    st.write(f"And we found reviews for **1041** unique companies.") # Hardcoded value here
    st.subheader("Extracted Data Fields:")
    st.markdown("""
- `review_title`
- `cust_name`
- `cust_location`
- `cust_reviews`
- `cust_rating`
- `cust_review_text`
- `seller_response`
- `date_experience`
- `company`""")
    logger.debug(f"show_scraping_section: Displayed placeholder columns.")


    st.subheader("Scraping Methodology")
    st.write("Our scraper was designed to:")
    st.markdown("""
    - Initially, scrape up to 20 pages of reviews per company, grouping ratings into two, 5-4 and 3-1.""")
    st.info("Which resulted in:")

    image_path: str = os.path.join(APP_DIR, 'Images', 'countplot_cust_rating.png')
    display_resized_image(image_path, "Customer Rating Distribution from Initial Scrape")


    logger.info("show_scraping_section: Finished")


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    show_scraping_section()