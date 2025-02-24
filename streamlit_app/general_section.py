# general_section.py
import streamlit as st
import pandas as pd
import os
from PIL import Image
import logging
from typing import Optional, List, Tuple

# Set up logging for this module
logger = logging.getLogger(__name__)

APP_DIR: str = os.path.dirname(os.path.abspath(__file__)) # general_section.py's directory (same as stream.py)
PARENT_DIR: str = os.path.dirname(APP_DIR) # Parent directory of streamlit_app (trust-me-data-analysis)
IMAGES_DIR: str = os.path.join(APP_DIR, 'Images') # Define IMAGES_DIR globally


def display_resized_image(image_path: str, caption: str, max_width: int = 800) -> None:
    """Resizes and displays an image using Streamlit.

    Args:
        image_path: Path to the image file.
        caption: Caption to display below the image.
        max_width: Maximum width of the resized image.
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
            img.thumbnail((max_width, max_width))  # Resize image
            st.image(img, caption=caption, use_container_width=True)
            logger.debug(f"display_resized_image: Displayed image: {image_path}")
        except Exception as e:
            logger.exception(f"display_resized_image: Error processing image '{image_path}': {e}")
            st.error(f"Error displaying image. Check logs for details.")
    else:
        st.warning(f"⚠️ Image not found: {image_path}")
        logger.warning(f"display_resized_image: Image not found: {image_path}")
    logger.info(f"display_resized_image: Finished for image_path='{image_path}'")


def display_dataframe_section(section_name: str, csv_path: str, df_info_img_path: Optional[str] = None) -> None:
    """Displays the Dataframe subsection for a given section.

    Args:
        section_name: The name of the section (for headers and image paths if needed).
        csv_path: Path to the CSV file for the dataframe.
        df_info_img_path: Optional path to the DataFrame info image.
    """
    logger.debug("display_dataframe_section: Displaying Dataframe subsection")
    st.markdown('<div class="content-box">', unsafe_allow_html=True)
    st.subheader("📂 Data Overview")

    # Load DataFrame
    logger.debug(f"display_dataframe_section: Loading DataFrame from: {csv_path}")

    if os.path.exists(csv_path):
        try:
            df_rev: pd.DataFrame = pd.read_csv(csv_path)
            st.write("Displaying first **10** rows for better performance:")
            st.dataframe(df_rev.head(10))  # Display only first 10 rows
            logger.debug("display_dataframe_section: DataFrame loaded and displayed.")
        except Exception as e:
            logger.exception(f"display_dataframe_section: Error loading or displaying DataFrame from '{csv_path}': {e}")
            st.error(f"Error loading DataFrame. Check logs for details.")

    else:
        st.warning(f"⚠️ CSV file not found: {csv_path}")
        logger.warning(f"display_dataframe_section: CSV file not found: {csv_path}")

    # DF Info Image (Optional)
    if df_info_img_path:
        with st.expander("📷 DF Info (Click to Expand)"):
            if os.path.exists(df_info_img_path):
                display_resized_image(df_info_img_path, "DF Info")
            else:
                st.warning(f"⚠️ DF Info image not found: {df_info_img_path}")
                logger.warning(f"display_dataframe_section: DF Info image not found: {df_info_img_path}")


    st.markdown("### 📝 Description")
    st.write("This section provides an overview of the dataset.")
    st.markdown('</div>', unsafe_allow_html=True)


def display_eda_section(section_name: str, eda_image_paths: Optional[List[Tuple[str, str]]] = None) -> None:
    """Displays the EDA subsection for a given section.

    Args:
        section_name: The name of the section (for headers and potentially image paths).
        eda_image_paths: Optional list of tuples, where each tuple is (title, image_path) for EDA images.
    """
    logger.debug("display_eda_section: Displaying EDA subsection")
    st.markdown('<div class="content-box">', unsafe_allow_html=True)
    st.subheader("📊 Exploratory Data Analysis")

    if eda_image_paths:
        with st.expander("📷 Exploratory Data Analysis (Click to Expand)"):
            for title, img_path in eda_image_paths:
                display_resized_image(img_path, title)
    else:
        st.info("No EDA images provided for this section.")


    st.markdown("### 📝 Analysis")
    st.write("EDA visualizations provide insights into data distributions and trends.")
    st.markdown('</div>', unsafe_allow_html=True)


def display_preprocessing_section(preprocessing_steps: Optional[List[str]] = None) -> None:
    """Displays the Preprocessing subsection.

    Args:
        preprocessing_steps: Optional list of preprocessing steps to display as a bullet list.
    """
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
        """) # Generic steps if none provided

    st.markdown('</div>', unsafe_allow_html=True)


def display_models_section(model_image_paths: Optional[List[Tuple[str, str]]] = None) -> None:
    """Displays the Models subsection.

    Args:
        model_image_paths: Optional list of tuples, where each tuple is (title, image_path) for model result images.
    """
    logger.debug("display_models_section: Displaying Models subsection")
    st.markdown('<div class="content-box">', unsafe_allow_html=True)
    st.subheader("🤖 Modeling")

    if model_image_paths:
        with st.expander("📷 Model Results (Click to Expand)"):
            for title, img_path in model_image_paths:
                # --- Debugging: Print image path before displaying ---
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
                     Defaults to ["Dataframe", "EDA", "Preprocessing", "Models"] if None.
        section_key_prefix: Prefix for Streamlit keys to avoid collisions if multiple general sections are used.
    """
    logger.info(f"show_general_section: Starting for section_name='{section_name}'")
    if not isinstance(section_name, str):
        logger.error(f"show_general_section: section_name is not a string: {type(section_name)}")
        raise TypeError(f"section_name must be a string, got {type(section_name)}")

    if subsections is None:
        subsections = ["Dataframe", "EDA", "Preprocessing", "Models"]

    logger.debug(f"show_general_section: Showing section: {section_name}")
    st.header(f"📊 {section_name.capitalize()} Analysis") # Dynamic header

    # Sub-Selection for Section
    subsection_choice: str = st.radio( # Renamed variable to be more general
        f"🔹 Choose a {section_name.capitalize()} Section:", # Dynamic label
        subsections,
        index=0,
        key=f"{section_key_prefix}_subsection_{section_name}" # Ensure unique key, using prefix
    )
    logger.debug(f"show_general_section: Selected subsection: {subsection_choice}")

    if subsection_choice == "Dataframe":
        # Example usage - you'll need to customize paths based on section_name
        csv_file_path = 'trustpilot_reviews_1000.csv' # General path - adjust as needed
        df_info_image_path = os.path.join(IMAGES_DIR, f"df info {section_name.lower()}.png") # Example image path
        display_dataframe_section(section_name, csv_file_path, df_info_image_path)

    elif subsection_choice == "EDA":
        # Example EDA image paths - customize based on section_name
        eda_images = [
            ("Rating distribution", os.path.join(IMAGES_DIR, f"Rating distribution {section_name.lower()}.png")),
            ("Sentiment distribution", os.path.join(IMAGES_DIR, f"Sentiment {section_name.lower()}.png")),
            ("Top 10 Word Ranking", os.path.join(IMAGES_DIR, "Word ranking top 10.png")), # General image
            ("Top 10 Negative Word Ranking", os.path.join(IMAGES_DIR, "Word Ranking neg 10.png")), # General image
        ]
        display_eda_section(section_name, eda_images)

    elif subsection_choice == "Preprocessing":
        preprocessing_steps_list = ["Data Cleaning", "Feature Engineering", "Normalization/Scaling", "Handling Missing Values"] # Example steps
        display_preprocessing_section(preprocessing_steps_list)

    elif subsection_choice == "Models":
        # Example model image paths - customize based on section_name
        model_images = [
            ("Logistic Regression Results", os.path.join(IMAGES_DIR, "LR Ergebnis.png")), # General images
            ("XGBoost Results", os.path.join(IMAGES_DIR, "XGBoost Ergebnis.png")),      # General images
        ]
        display_models_section(model_images)

    logger.info(f"show_general_section: Finished for section_name='{section_name}'")


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG) # Configure basic logging if running standalone
    st.set_page_config(layout="wide") # Optional: Set page layout if running standalone
    show_general_section(section_name="Sports") # Example usage for "Sports" section
    show_general_section(section_name="Finance", section_key_prefix="finance_section") # Example usage for "Finance" section, with unique key prefix