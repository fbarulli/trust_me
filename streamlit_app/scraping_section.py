# scraping_section.py
import streamlit as st
import pandas as pd
import os
import logging
from typing import Optional
from category_section import display_resized_image

logger = logging.getLogger(__name__)

APP_DIR: str = os.path.dirname(os.path.abspath(__file__))

SCRAPED_CSV_FILE_NAME: str = 'trustpilot_reviews.csv'
SCRAPED_CSV_FILE_PATH: str = os.path.join('..', 'fabian', 'EDA', SCRAPED_CSV_FILE_NAME)


def show_scraping_section() -> None:
    logger.info("show_scraping_section: Starting")
    logger.debug(f"show_scraping_section: Constructed CSV path: {SCRAPED_CSV_FILE_PATH}")

    st.header("🌐 Initial Scrape")
    st.write(f"During our initial scrape we collected:")

    logger.info(f"show_scraping_section: Attempting to load scraped data from: {SCRAPED_CSV_FILE_PATH}")
    scraped_df: Optional[pd.DataFrame] = None
    SCRAPED_DATA_LOADED: bool = False

    try:
        scraped_df = pd.read_csv(SCRAPED_CSV_FILE_PATH)
        SCRAPED_DATA_LOADED = True
        logger.info(f"show_scraping_section: Successfully loaded scraped data from: {SCRAPED_CSV_FILE_PATH}")
    except FileNotFoundError:
        logger.warning(f"show_scraping_section: File not found for scraped data CSV at: {SCRAPED_CSV_FILE_PATH}. Proceeding with hardcoded data.")
        SCRAPED_DATA_LOADED = False
        scraped_df = None
        logger.warning(f"show_scraping_section: File not found error for scraped data CSV: {SCRAPED_CSV_FILE_PATH}")
    except Exception as e:
        st.error(f"An error occurred while loading the different CSV: {e}")
        SCRAPED_DATA_LOADED = False
        scraped_df = None
        logger.exception(f"show_scraping_section: An error occurred while loading scraped CSV: {e}", exc_info=True)


    SCRAPED_DATA_LOADED = True
    hardcoded_shape = (140124, 9)
    shape_code_scraped_df: str = f"""
  shape {hardcoded_shape}
"""
    st.code(shape_code_scraped_df, language="plaintext")
    logger.debug(f"show_scraping_section: Displayed hardcoded shape: {hardcoded_shape}")
    st.write(f"And we found reviews for **[Number of Companies -  To be implemented if needed]** unique companies.")
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


    if SCRAPED_DATA_LOADED and scraped_df is not None:
        if scraped_df is not None:
            if "company" in scraped_df.columns:
                unique_companies_count = scraped_df["company"].nunique()
                logger.debug(f"show_scraping_section: Displayed unique company count: {unique_companies_count}")
            else:
                logger.warning("show_scraping_section: Column 'company' not found in scraped data.")

            if scraped_df.columns.tolist():
                columns_list = scraped_df.columns.tolist()
                logger.debug(f"show_scraping_section: Displayed DataFrame columns: {columns_list}")
            else:
                logger.warning("show_scraping_section: Could not retrieve column names from scraped_df.")


        else:
            logger.warning("show_scraping_section: SCRAPED_DATA_LOADED is True, but scraped_df is None.")

    elif not SCRAPED_DATA_LOADED:
        st.warning("Independent scraped data loading failed. DataFrame shape is not available for the scraped data.")
        logger.warning("show_scraping_section: Independent scraped data loading failed.")



    st.subheader("Scraping Methodology")
    st.write("Our scraper was designed to:")
    st.markdown("""
    - Initially, scrape up to 20 pages of reviews per company, grouping ratings into two, 5-4 and 3-1.""")
    st.info("Which resulted in:")

    image_path: str = os.path.join(APP_DIR, 'Images', 'countplot_cust_rating.png')
    display_resized_image(image_path, "Customer Rating Distribution from Initial Scrape")

    st.header("🌐 Subsequent Scrape")
    st.markdown("""
    - There was a need for a more balanced dataset, therefore an approach with this in mind was made. Resulting in:""")
    image_path_2: str = os.path.join(APP_DIR, 'Images', 'countplot_scrape_2.png')
    display_resized_image(image_path_2, "Customer Rating Distribution from Subsequent Scrape")


    logger.info("show_scraping_section: Finished")


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    show_scraping_section()