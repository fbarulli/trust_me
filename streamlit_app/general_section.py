# general_section.py
import streamlit as st
import pandas as pd
import os
from PIL import Image
import logging
from typing import Optional, List, Tuple

# Set up logging for this module
logger = logging.getLogger(__name__)

APP_DIR: str = os.path.dirname(os.path.abspath(__file__))  # general_section.py's directory (same as stream.py)
PARENT_DIR: str = os.path.dirname(APP_DIR)  # Parent directory of streamlit_app (trust-me-data-analysis)
IMAGES_DIR: str = os.path.join(APP_DIR, 'Images')  # Define IMAGES_DIR globally
SCRAPE_1_DIR: str = os.path.join(IMAGES_DIR, 'fabian_img', 'ngram_sentiment', 'scrape_1')  # Initial scrape images
SCRAPE_2_DIR: str = os.path.join(IMAGES_DIR, 'fabian_img', 'ngram_sentiment', 'scrape_2')  # Subsequent scrape images


def display_resized_image(image_path: str, caption: str, max_width: int = 800) -> None:
    """Displays an image using Streamlit at its original size (or doubled, if specified).

    Args:
        image_path: Path to the image file.
        caption: Caption to display below the image.
        max_width: Maximum width parameter (used as a fallback).
    """
    logger.info(f"display_resized_image: Starting for image_path='{image_path}', caption='{caption}', max_width={max_width}")
    if not isinstance(image_path, str):
        logger.error(f"display_resized_image: image_path is not a string: {type(image_path)}")
        raise TypeError(f"image_path must be a string, got {type(image_path)}")
    if not isinstance(caption, str):
        logger.error(f"display_resized_image: caption is not a string: {type(caption)}")
        raise TypeError(f"caption must be a string, got {type(caption)}")
    if not isinstance(max_width, int):
        logger.error(f"display_resized_image: max_width is not an int: {type(max_width)}")
        raise TypeError(f"max_width must be an int, got {type(max_width)}")

    if os.path.exists(image_path):
        try:
            img = Image.open(image_path)
            original_width, original_height = img.size  # Get original dimensions
            new_width = original_width  # Use full original width (2x from previous half size)
            new_width = min(new_width, max_width * 2)  # Allow 2x max_width for "2x bigger"
            st.image(img, caption=caption, width=new_width)  # Display at new width
            logger.debug(f"display_resized_image: Displayed image with original_width={original_width}, new_width={new_width}")
        except Exception as e:
            logger.exception(f"display_resized_image: Error processing image '{image_path}': {e}")
            st.error(f"Error displaying image. Check logs for details.")
    else:
        st.warning(f"⚠️ Image not found: {image_path}")
        logger.warning(f"display_resized_image: Image not found: {image_path}")
    logger.info(f"display_resized_image: Finished for image_path='{image_path}'")


def display_dataframe_section(section_name: str, csv_path: str) -> None:
    """Displays the Dataframe subsection for a given section, including the plot.

    Args:
        section_name: The name of the section (for headers and image paths if needed).
        csv_path: Path to the CSV file for the dataframe.
    """
    logger.debug("display_dataframe_section: Displaying Dataframe subsection")
    st.markdown("""
    <style>
    .content-box {
        width: 1400px;  /* Default Streamlit width is ~700px, so 2x is 1400px */
        max-width: 100%;  /* Ensure it fits within the viewport */
        margin: 0 auto;  /* Center it */
    }
    </style>
    <div class="content-box">""", unsafe_allow_html=True)

    logger.debug(f"display_dataframe_section: DataFrame loading is disabled in this version.")
    if os.path.exists(csv_path):
        try:
            pd.read_csv(csv_path)  # Just read the CSV, don’t need to assign to df_rev
            logger.debug("display_dataframe_section: DataFrame loaded (not displayed).")
        except Exception as e:
            logger.exception(f"display_dataframe_section: Error loading DataFrame from '{csv_path}': {e}")
            st.error(f"Error loading DataFrame in background. Check logs for details.")
    else:
        st.warning(f"⚠️ CSV file not found: {csv_path}")
        logger.warning(f"display_dataframe_section: CSV file not found: {csv_path}")

    st.markdown("### 📝 Description")
    st.markdown("""
    - In this scenario, I decided to scrape a balanced dataset.\n
    - Main reason for doing this was because I would have to modify training to accommodate an unbalanced dataset.\n
    - I decided it would be better to work with a balanced dataset.\n
    - To my understanding, the dataset is naturally biased, so there's no unbiased way to collect data from this website.\n
    - The final dataset had a shape of (60875, 2), with 12175 reviews per rating, from all 138 categories available on the website.
    """)

    image_path: str = os.path.join(IMAGES_DIR, 'countplot_scrape_2.png')
    display_resized_image(image_path, "Customer Rating Distribution from Subsequent Scrape", max_width=800)

    st.markdown('</div>', unsafe_allow_html=True)


def display_eda_section(section_name: str, eda_image_paths: Optional[List[Tuple[str, str]]] = None, sub_section: str = None) -> None:
    """Displays the EDA subsection for a given section, with Scrape Comparison plots.

    Args:
        section_name: The name of the section (for headers and potentially image paths).
        eda_image_paths: Optional list of tuples, where each tuple is (title, image_path) for EDA images.
        sub_section: Specific EDA subsection to display (not used here, kept for compatibility).
    """
    logger.debug("display_eda_section: Displaying EDA subsection")
    st.markdown('<div class="content-box">', unsafe_allow_html=True)
    st.subheader("📊 Exploratory Data Analysis")

    # Scrape Comparison Plots (side-by-side)
    if os.path.exists(SCRAPE_1_DIR) and os.path.exists(SCRAPE_2_DIR):
        scrape_1_images = [(f"Initial Scrape: {os.path.splitext(img_file)[0].replace('_', ' ').title()}", 
                            os.path.join(SCRAPE_1_DIR, img_file)) 
                           for img_file in os.listdir(SCRAPE_1_DIR) 
                           if img_file.lower().endswith(('.png', '.jpg', '.jpeg'))]
        scrape_2_images = [(f"Subsequent Scrape: {os.path.splitext(img_file)[0].replace('_', ' ').title()}", 
                            os.path.join(SCRAPE_2_DIR, img_file)) 
                           for img_file in os.listdir(SCRAPE_2_DIR) 
                           if img_file.lower().endswith(('.png', '.jpg', '.jpeg'))]

        scrape_1_images.sort(key=lambda x: x[1].split(os.sep)[-1])
        scrape_2_images.sort(key=lambda x: x[1].split(os.sep)[-1])

        if scrape_1_images or scrape_2_images:
            with st.expander("📷 Scrape Comparison Plots (Click to Expand)"):
                max_pairs = min(len(scrape_1_images), len(scrape_2_images))
                if max_pairs == 0:
                    st.info("No matching images found for comparison.")
                else:
                    for i in range(max_pairs):
                        col1, col2 = st.columns(2)
                        with col1:
                            display_resized_image(scrape_1_images[i][1], scrape_1_images[i][0])
                        with col2:
                            display_resized_image(scrape_2_images[i][1], scrape_2_images[i][0])
                st.write("As we can see, most words appear with good and bad sentiment, making this a difficult task.")
        else:
            st.info("No scrape comparison images found in the specified directories.")
    else:
        if not os.path.exists(SCRAPE_1_DIR):
            st.warning(f"⚠️ Initial scrape directory not found: {SCRAPE_1_DIR}")
        if not os.path.exists(SCRAPE_2_DIR):
            st.warning(f"⚠️ Subsequent scrape directory not found: {SCRAPE_2_DIR}")

    st.markdown('</div>', unsafe_allow_html=True)


def display_preprocessing_section(preprocessing_steps: Optional[List[str]] = None) -> None:
    """Displays the Preprocessing subsection."""
    logger.debug("display_preprocessing_section: Displaying Preprocessing subsection")
    st.markdown('<div class="content-box">', unsafe_allow_html=True)
    st.subheader("🛠 Data Preprocessing")
    st.write("The steps of data preprocessing will be described here later.")

    if preprocessing_steps:
        st.markdown("""
        {}
        """.format('\n'.join([f"- **Step {i+1}**: {step}" for i, step in enumerate(preprocessing_steps)])))
    else:
        st.markdown("""
        - **Step 1**: ...
        - **Step 2**: ...
        - **Step 3**: ...
        - **Step 4**: ...
        """)

    st.markdown('</div>', unsafe_allow_html=True)


def display_models_section(model_image_paths: Optional[List[Tuple[str, str]]] = None) -> None:
    """Displays the Models subsection."""
    logger.debug("display_models_section: Displaying Models subsection")
    st.markdown('<div class="content-box">', unsafe_allow_html=True)
    st.subheader("🤖 Modeling")

    if model_image_paths:
        with st.expander("📷 Model Results (Click to Expand)"):
            for title, img_path in model_image_paths:
                logger.debug(f"display_models_section: Trying to display image: {img_path}")
                display_resized_image(img_path, title)
    else:
        st.info("No model result images provided for this section.")

    st.markdown("### 📝 Description")
    st.write("These images showcase the model results.")
    st.markdown('</div>', unsafe_allow_html=True)


def show_general_section(section_name: str = "analysis", subsections: Optional[List[str]] = None, section_key_prefix: str = "general_section") -> None:
    """Displays a general analysis section in the Streamlit app, allowing for customizable subsections.

    Args:
        section_name: The name of the section to display (e.g., "sports", "finance").
        subsections: List of subsections to include (e.g., ["Dataframe", "EDA", "Preprocessing", "Models"]).
                     Defaults to simplified list if None.
        section_key_prefix: Prefix for Streamlit keys to avoid collisions if multiple general sections are used.
    """
    logger.info(f"show_general_section: Starting for section_name='{section_name}'")
    if not isinstance(section_name, str):
        logger.error(f"show_general_section: section_name is not a string: {type(section_name)}")
        raise TypeError(f"section_name must be a string, got {type(section_name)}")

    if subsections is None:
        subsections = ["Dataframe", "EDA", "Preprocessing", "Models"]

    logger.debug(f"show_general_section: Showing section: {section_name}")
    st.header(f"📊 General Analysis")

    subsection_choice: str = st.radio(
        f"🔹 Choose a {section_name.capitalize()} Section:",
        subsections,
        index=0,
        key=f"{section_key_prefix}_subsection_{section_name}"
    )
    logger.debug(f"show_general_section: Selected subsection: {subsection_choice}")

    if subsection_choice == "Dataframe":
        csv_file_path = 'trustpilot_reviews_1000.csv'
        display_dataframe_section(section_name, csv_file_path)

    elif subsection_choice == "EDA":
        display_eda_section(section_name)

    elif subsection_choice == "Preprocessing":
        preprocessing_steps_list = ["Data Cleaning", "Feature Engineering", "Normalization/Scaling", "Handling Missing Values"]
        display_preprocessing_section(preprocessing_steps_list)

    elif subsection_choice == "Models":
        model_images = [
            ("Logistic Regression Results", os.path.join(IMAGES_DIR, "LR Ergebnis.png")),
            ("XGBoost Results", os.path.join(IMAGES_DIR, "XGBoost Ergebnis.png")),
        ]
        display_models_section(model_images)

    logger.info(f"show_general_section: Finished for section_name='{section_name}'")


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    st.set_page_config(layout="wide")
    show_general_section(section_name="General")