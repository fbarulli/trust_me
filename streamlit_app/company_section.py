# company_section.py
import streamlit as st
import pandas as pd
import os
from PIL import Image
from typing import Optional, List, Tuple
import logging

# Set up logging for this module
logger = logging.getLogger(__name__)

PP_DIR: str = os.path.dirname(os.path.abspath(__file__))  # category_section.py's directory
PARENT_DIR: str = os.path.dirname(APP_DIR)  # Parent directory

# Function to resize and display images
def display_resized_image(image_path: str, caption: str, max_width: int = 800) -> None:
    """Resizes and displays an image using Streamlit."""
    if os.path.exists(image_path):
        try:
            img = Image.open(image_path)
            img.thumbnail((max_width, max_width))
            st.image(img, caption=caption, use_container_width=True)
        except Exception as e:
            logger.exception(f"Error processing image '{image_path}': {e}")
            st.error(f"Error displaying image. Check logs for details.")
    else:
        st.warning(f"⚠️ Image not found: {image_path}")

def show_company_section(company_name: str = "Nike") -> None:
    """Displays the company analysis section in the Streamlit app."""
    logger.info("show_company_section: Starting")
    IMAGES_DIR: str = os.path.join(APP_DIR, 'Images')

    st.header(f"📊 {company_name.capitalize()} - A Case Study")

    st.write("🚀 As one of the biggest sports brands the company Nike was chosen to be the representative of the category"
             "Sports.")

    company_subsection: str = st.radio(
        f"🔹 Choose a subsectgion for the Analysis of the company Nike:",
        ["Dataframe", "EDA", "Preprocessing", "Models"],
        index=0,
        key=f"company_subsection_{company_name}"
    )

    # --- Dataframe Section ---
    if company_subsection == "Dataframe":
        st.markdown('<div class="content-box">', unsafe_allow_html=True)
        st.subheader("📂 Data Overview")

        df_path: str = os.path.join(APP_DIR, 'data_nike_raw.csv')
        if os.path.exists(df_path):
            df_rev: pd.DataFrame = pd.read_csv(df_path)
            st.write("Displaying first **10** rows for better performance:")
            st.dataframe(df_rev.head(10))
            st.dataframe(df_rev.info())
            st.dataframe(df_rev.describe())
        else:
            st.warning(f"⚠️ CSV file not found: {df_path}")

        # DF Info Image
        with st.expander("📷 DF Info (Click to Expand)"):
            df_info_path: str = os.path.join(IMAGES_DIR, f"df info {company_name}.png")
            display_resized_image(df_info_path, "DF Info")

        st.markdown("### 📝 Description")
        st.markdown("""
            - data was scraped from Nike's Trustpilot website ('https://www.trustpilot.com/review/www.nike.com')
            - to date there are 10k+ reviews; average star rating: 1.7 (qualified as 'bad')
            - most prominent category is 1-star (73%), followed by 5-stars (16%),
             4- and 2-stars (both 4%) and finally 3-stars (3%)
            - small, imbalanced data set
            - for practical reasons: only ca. 5200 reviews were used for further analysis. (cf. Section 'EDA')
            - not all of the reviews in Enlish language (~1.6% 'Other') --> standard scraping protocol used for retrieving
            the data of the sports category to be modified
            """)
        st.markdown('</div>', unsafe_allow_html=True)

    # --- EDA Section ---
    elif company_subsection == "EDA":
        st.markdown('<div class="content-box">', unsafe_allow_html=True)
        st.subheader("📊 Exploratory Data Analysis")

        eda_images: List[Tuple[str, str]] = [
            ("Rating distribution", os.path.join(IMAGES_DIR, "countplot_ratings_nike_02.png")),
            ("Sentiment distribution", os.path.join(IMAGES_DIR, "sentiment_analysis_nike_02.png")),
            ("Word Cloud 1 - all ratings", os.path.join(IMAGES_DIR, "word_cloud_nike_01.png")),
            ("Word Cloud 2 - Nike ratings", os.path.join(IMAGES_DIR, "word_cloud_nike_02.png")),
        ]

        with st.expander("📷 Exploratory Data Analysis (Click to Expand)"):
            for title, img_path in eda_images:
                display_resized_image(img_path, title)

        st.markdown("### 📝 Key Insights from EDA")
        st.markdown("""
            - **Rating Distribution:** A strong bias towards 1-star reviews, 5-star reviews as major minor category,
              mitigation via **SMOTE (Synthetic Minority Over-sampling Technique)**.
            - **Sentiment Analysis:** Using **TextBlob**, strong correlation between sentiment polarity and ratings.
            - **Word Clouds:**
              - while trying to scrape all of the 10k+ Nike reviews, a contamination of the data with other reviews can be
              observed **'tesco scammed'** (Word Cloud 1) --> reduction to only 5,200 Nike reviews
              - **Word Cloud 1:** most sold product by far **'shoe'**, issues with **'customer service', '(delivery) time',
              'refund' and 'quality' (?)**, only some praise the quality or satisfaction as **'good'**
            """)
        st.markdown('</div>', unsafe_allow_html=True)

    # --- Preprocessing Section ---
    elif company_subsection == "Preprocessing":
        st.markdown('<div class="content-box">', unsafe_allow_html=True)
        st.subheader("🛠 Data Preprocessing")
        st.markdown("""
                Before training our models, we applied several preprocessing techniques:
                1. **Text Cleaning:** Converting to lowercase, removing special characters, stopwords, and punctuation.
                2. **Tokenization & Lemmatization:** Breaking text into meaningful units and converting words to their root forms.
                3. **Feature Extraction:**
                   - Traditional approaches: **TF-IDF (Term Frequency-Inverse Document Frequency)**
                """)
        st.markdown('</div>', unsafe_allow_html=True)

    # --- Models Section ---
    elif company_subsection == "Models":
        st.markdown('<div class="content-box">', unsafe_allow_html=True)
        st.subheader("🤖 Modeling")

        model_images: List[Tuple[str, str]] = [
            ("Logistic Regression Results", os.path.join(IMAGES_DIR, "class_rep_nike_lr.png")),
            ("Logistic Regression Confusion Matrix", os.path.join(IMAGES_DIR, "cm_nike_lr.png")),
            ("SVM Results", os.path.join(IMAGES_DIR, "class_rep_nike_svm.png")),
            ("SVM Confusion Matrix", os.path.join(IMAGES_DIR, "cm_nike_svm.png")),
            ("Random Forest Results", os.path.join(IMAGES_DIR, "class_rep_nike_rf.png")),
            ("Random Forest Confusion Matrix", os.path.join(IMAGES_DIR, "cm_nike_rf.png")),
            ("XGBoost Results", os.path.join(IMAGES_DIR, "class_rep_nike_xgb.png")),
            ("XGBoost Confusion Matrix", os.path.join(IMAGES_DIR, "cm_nike_xgb.png")),
            ("Summary Model Performance", os.path.join(IMAGES_DIR, "comparison_nike_models.jpeg")),
        ]

        with st.expander("📷 Model Results (Click to Expand)"):
            for title, img_path in model_images:
                display_resized_image(img_path, title)

        st.markdown("### 📝 Model Experimentation & Evaluation")
        st.markdown("""
                Models trained and evaluated:
                **Machine Learning:**
                1. **Logistic Regression (LR):** Simple baseline model, reasonable accuracy (93%).
                2. **Support Vector Machine (SVM):** Improved accuracy and recall (95%)
                3. **Ensemble Methods:**
                   - **Random Forest (RF):** Best overall performance and consistency (accuracy: 98%)
                   - **XGBoost:** Slightly less performant than the RF model (accuracy: 95%) but in general a little more robust
                     than the SVM model; decrease in performance might be due to different preprocessing approach (use of
                     LabelEncoder was necessary)
                **Deep Learning:**
                4. **BERT-based Transformer Model):** No evaluation possible due to time constraints and
                   hardware limitations
                """)
        st.markdown('</div>', unsafe_allow_html=True)

    logger.info("show_company_section: Finished")

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG) # Configure basic logging if running standalone
    show_company_section() # Test function