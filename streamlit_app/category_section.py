# category_section.py 
import streamlit as st
import pandas as pd
import os
from PIL import Image
import logging
from typing import Optional, List, Tuple

# Set up logging for this module
logger = logging.getLogger(__name__)

APP_DIR: str = os.path.dirname(os.path.abspath(__file__))  # category_section.py's directory
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

def show_category_section(category_name: str = "sports") -> None:
    """Displays the category analysis section in the Streamlit app."""
    IMAGES_DIR: str = os.path.join(APP_DIR, 'Images')

    st.header(f"📊 {category_name.capitalize()} Category Analysis") 

    category_subsection: str = st.radio(
        f"🔹 Choose a {category_name.capitalize()} Category Section:",
        ["Dataframe", "EDA", "Preprocessing", "Models"],
        index=0,
        key=f"category_subsection_{category_name}"
    )

    # --- Dataframe Section ---
    if category_subsection == "Dataframe":
        st.markdown('<div class="content-box">', unsafe_allow_html=True)
        st.subheader("📂 Data Overview")

        df_path: str = os.path.join(APP_DIR, 'trustpilot_reviews_1000.csv')
        if os.path.exists(df_path):
            df_rev: pd.DataFrame = pd.read_csv(df_path)
            st.write("Displaying first **10** rows for better performance:")
            st.dataframe(df_rev.head(10))  
        else:
            st.warning(f"⚠️ CSV file not found: {df_path}")

        # DF Info Image
        with st.expander("📷 DF Info (Click to Expand)"):
            df_info_path: str = os.path.join(IMAGES_DIR, f"df info {category_name}.png")
            display_resized_image(df_info_path, "DF Info")

        st.markdown("### 📝 Description")
        st.markdown("""
        - **Structured Data is Crucial:** A well-structured dataset is essential for effective machine learning and predictive modeling.
        - **Trustpilot Review Extraction:** The dataset was initially gathered by scraping Trustpilot's website.
        - **Data Cleaning:** The raw dataset required extensive cleaning to remove inconsistencies, missing values, and noise.
        - **Company & Review Collection:**
          - Trustpilot categories were analyzed to identify relevant **sports companies**.
          - Companies were scraped from the first three pages of Trustpilot search results.
          - Reviews were extracted from each company’s first page, capturing **ratings, review text, timestamps, and metadata**.
        - **Balancing the Dataset:** Initial EDA revealed a heavy **imbalance toward 5-star reviews**.
          - To correct this, additional reviews were scraped, **expanding the dataset to 1,000 reviews per company**.
          - This process increased the dataset size to **34,923 reviews**, making it more representative.
        """)

        st.markdown('</div>', unsafe_allow_html=True)

    # --- EDA Section ---
    elif category_subsection == "EDA":
        st.markdown('<div class="content-box">', unsafe_allow_html=True)
        st.subheader("📊 Exploratory Data Analysis")

        eda_images: List[Tuple[str, str]] = [
            ("Rating distribution", os.path.join(IMAGES_DIR, f"Rating distribution {category_name}.png")),
            ("Sentiment distribution", os.path.join(IMAGES_DIR, f"Sentiment {category_name}.png")),
            ("Top 10 Word Ranking", os.path.join(IMAGES_DIR, "Word ranking top 10.png")),
            ("Top 10 Negative Word Ranking", os.path.join(IMAGES_DIR, "Word Ranking neg 10.png")),
        ]

        with st.expander("📷 Exploratory Data Analysis (Click to Expand)"):
            for title, img_path in eda_images:
                display_resized_image(img_path, title)

        st.markdown("### 📝 Key Insights from EDA")
        st.markdown("""
        - **Rating Distribution:** A strong bias towards 5-star reviews, requiring mitigation via **SMOTE (Synthetic Minority Over-sampling Technique)**.
        - **Sentiment Analysis:** Using **TextBlob**, we confirmed a strong correlation between sentiment polarity and ratings.
        - **Word Rankings:**
          - Positive reviews frequently mentioned words like **'great,' 'fast,' and 'recommend.'**
          - Negative reviews focused on terms such as **'delay,' 'order,' and 'customer service.'**
        """)

        st.markdown('</div>', unsafe_allow_html=True)

    # --- Preprocessing Section ---
    elif category_subsection == "Preprocessing":
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
    elif category_subsection == "Models":
        st.markdown('<div class="content-box">', unsafe_allow_html=True)
        st.subheader("🤖 Modeling")

        model_images: List[Tuple[str, str]] = [
            ("Logistic Regression Results", os.path.join(IMAGES_DIR, "LR Ergebnis.png")),
            ("XGBoost Results", os.path.join(IMAGES_DIR, "XGBoost Ergebnis.png")),
        ]

        with st.expander("📷 Model Results (Click to Expand)"):
            for title, img_path in model_images:
                display_resized_image(img_path, title)

        st.markdown("### 📝 Model Experimentation & Evaluation")
        st.markdown("""
        We experimented with multiple approaches:
        1. **Logistic Regression & SVM:** Simple but effective baseline models.
        2. **Ensemble Methods (Random Forest, XGBoost):** Provided higher accuracy due to better feature learning.
        3. **Deep Learning (BERT-based Transformer Model):** Captured contextual meaning but was constrained by hardware limitations.

        Among these, **XGBoost performed the best**, balancing accuracy and computational efficiency.

        To evaluate our models, we used **accuracy, precision, recall, and F1-score**:
        - **XGBoost achieved 94% accuracy** and performed best in balancing precision and recall.
        """)

        st.markdown('</div>', unsafe_allow_html=True)

    logger.info(f"show_category_section: Finished for category_name='{category_name}'")

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    st.set_page_config(layout="wide")
    show_category_section()
