

# category_section.py
import streamlit as st
import pandas as pd
import os
from PIL import Image
import logging
from typing import Optional, List, Tuple

# Set up logging for this module
logger = logging.getLogger(__name__)

APP_DIR: str = os.path.dirname(os.path.abspath(__file__)) # category_section.py's directory (same as stream.py)
PARENT_DIR: str = os.path.dirname(APP_DIR) # Parent directory of streamlit_app (trust-me-data-analysis)


# Function to resize and display images
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


def show_category_section(category_name: str = "sports") -> None:
    """Displays the category analysis section in the Streamlit app.

    Args:
        category_name: The name of the category to analyze (default: "sports").
    """
    logger.info(f"show_category_section: Starting for category_name='{category_name}'")
    if not isinstance(category_name, str):
        logger.error(f"show_category_section: category_name is not a string: {type(category_name)}")
        raise TypeError(f"category_name must be a string, got {type(category_name)}")

    # Update image paths to use APP_DIR
    IMAGES_DIR: str = os.path.join(APP_DIR, 'Images')

    logger.debug(f"show_category_section: Showing category section: {category_name}")
    st.header(f"📊 {category_name.capitalize()} Category Analysis") # Dynamic header

    # Sub-Selection for Category
    category_subsection: str = st.radio( # Renamed variable to avoid shadowing section name
        f"🔹 Choose a {category_name.capitalize()} Category Section:", # Dynamic label
        ["Dataframe", "EDA", "Preprocessing", "Models"],
        index=0,
        key=f"category_subsection_{category_name}" # Ensure unique key
    )
    logger.debug(f"show_category_section: Selected category subsection: {category_subsection}")

    # --- Dataframe Section ---
    if category_subsection == "Dataframe":
        logger.debug("show_category_section: Displaying Dataframe subsection")
        st.markdown('<div class="content-box">', unsafe_allow_html=True)
        st.subheader("📂 Data Overview")

        # Load DataFrame - Path updated to project root relative path
        df_path: str = os.path.join('trustpilot_reviews_1000.csv') # Adjusted relative path - project root
        logger.debug(f"show_category_section: Loading DataFrame from: {df_path}")

        if os.path.exists(df_path):
            try:
                df_rev: pd.DataFrame = pd.read_csv(df_path)
                st.write("Displaying first **10** rows for better performance:")
                st.dataframe(df_rev.head(10))  # Display only first 10 rows
                logger.debug("show_category_section: DataFrame loaded and displayed.")
            except Exception as e:
                logger.exception(f"show_category_section: Error loading or displaying DataFrame from '{df_path}': {e}")
                st.error(f"Error loading DataFrame. Check logs for details.")

        else:
            st.warning(f"⚠️ CSV file not found: {df_path}")
            logger.warning(f"show_category_section: CSV file not found: {df_path}")

        # DF Info Image
        with st.expander("📷 DF Info (Click to Expand)"):
            # Relative image path, assuming Images folder is within streamlit_app directory
            df_info_path: str = os.path.join(IMAGES_DIR, f"df info {category_name}.png") # Relative image path - CORRECTED PATH
            display_resized_image(df_info_path, "DF Info")

        st.markdown("### 📝 Description")
        st.write("This section provides an overview of the dataset.")
        st.markdown('</div>', unsafe_allow_html=True)

    # --- EDA Section ---
    elif category_subsection == "EDA":
        logger.debug("show_category_section: Displaying EDA subsection")
        st.markdown('<div class="content-box">', unsafe_allow_html=True)
        st.subheader("📊 Exploratory Data Analysis")

        # EDA Images
        eda_images: List[Tuple[str, str]] = [
            ("Rating distribution", os.path.join(IMAGES_DIR, f"Rating distribution {category_name}.png")), # Relative paths - CORRECTED PATH
            ("Sentiment distribution", os.path.join(IMAGES_DIR, f"Sentiment {category_name}.png")),       # Relative paths - CORRECTED PATH
            ("Top 10 Word Ranking", os.path.join(IMAGES_DIR, "Word ranking top 10.png")),     # Relative path - CORRECTED PATH
            ("Top 10 Negative Word Ranking", os.path.join(IMAGES_DIR, "Word Ranking neg 10.png")), # Relative path - CORRECTED PATH
        ]

        with st.expander("📷 Exploratory Data Analysis (Click to Expand)"):
            for title, img_path in eda_images:
                display_resized_image(img_path, title)

        st.markdown("### 📝 Analysis")
        st.write("EDA visualizations provide insights into data distributions and trends.")
        st.markdown('</div>', unsafe_allow_html=True)

    # --- Preprocessing Section ---
    elif category_subsection == "Preprocessing":
        logger.debug("show_category_section: Displaying Preprocessing subsection")
        st.markdown('<div class="content-box">', unsafe_allow_html=True)
        st.subheader("🛠 Data Preprocessing")
        st.write("The steps of data preprocessing will be described here later.")

        st.markdown("""
        - **Step 1**: Data Cleaning
        - **Step 2**: Feature Engineering
        - **Step 3**: Normalization/Scaling
        - **Step 4**: Handling Missing Values
        """)
        st.markdown('</div>', unsafe_allow_html=True)

    # --- Models Section ---
    elif category_subsection == "Models":
        logger.debug("show_category_section: Displaying Models subsection")
        st.markdown('<div class="content-box">', unsafe_allow_html=True)
        st.subheader("🤖 Modeling")

        # Model Results Images
        model_images: List[Tuple[str, str]] = [
            ("Logistic Regression Results", os.path.join(IMAGES_DIR, "LR Ergebnis.png")), # Relative paths - CORRECTED PATH
            ("XGBoost Results", os.path.join(IMAGES_DIR, "XGBoost Ergebnis.png")),      # Relative paths - CORRECTED PATH
        ]

        with st.expander("📷 Model Results (Click to Expand)"):
            for title, img_path in model_images:
                # --- Debugging: Print image path before displaying ---
                logger.debug(f"show_category_section: Trying to display image: {img_path}")
                display_resized_image(img_path, title)

        st.markdown("### 📝 Description")
        st.write("These images showcase the model results.")
        st.markdown('</div>', unsafe_allow_html=True)
    logger.info(f"show_category_section: Finished for category_name='{category_name}'")


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG) # Configure basic logging if running standalone
    st.set_page_config(layout="wide") # Optional: Set page layout if running standalone
    show_category_section() # You can run it directly to test the section