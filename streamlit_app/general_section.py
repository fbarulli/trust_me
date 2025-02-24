# general_section.py
import streamlit as st
import pandas as pd
import os
from PIL import Image
import logging
from typing import Optional, List, Tuple
from sentiment_model import predict_sentiment  # Import from separate module

# Set up logging for this module
logger = logging.getLogger(__name__)

APP_DIR: str = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR: str = os.path.dirname(APP_DIR)
IMAGES_DIR: str = os.path.join(APP_DIR, 'Images')
SCRAPE_1_DIR: str = os.path.join(IMAGES_DIR, 'fabian_img', 'ngram_sentiment', 'scrape_1')
SCRAPE_2_DIR: str = os.path.join(IMAGES_DIR, 'fabian_img', 'ngram_sentiment', 'scrape_2')

# [Existing functions like display_resized_image, display_dataframe_section, etc. remain unchanged]

def display_sentiment_prediction_section():
    """Displays the Sentiment Prediction subsection."""
    logger.debug("display_sentiment_prediction_section: Displaying Sentiment Prediction subsection")
    st.markdown('<div class="content-box">', unsafe_allow_html=True)
    st.subheader("🔍 Sentiment Prediction")

    # User input
    user_input = st.text_area("Enter your text here:", height=150, placeholder="Type your review or comment...")
    
    if st.button("Predict Sentiment"):
        if user_input.strip():
            try:
                # Define the model path relative to APP_DIR
                model_path = os.path.join(APP_DIR, "model", "fabian_model", "final_model.pt")
                
                # Predict sentiment using the imported function
                result = predict_sentiment(user_input, model_path=model_path)
                
                # Display results
                st.markdown("### Prediction Results")
                st.write(f"**Sentiment:** {result['sentiment']} (Class {result['predicted_class']})")
                st.write("**Probabilities:**")
                for label, prob in result['probabilities'].items():
                    st.write(f"- {label}: {prob:.4f}")
            except FileNotFoundError as e:
                st.error(f"Model file not found: {e}")
            except Exception as e:
                logger.exception(f"Error during sentiment prediction: {e}")
                st.error("An error occurred during prediction. Check logs for details.")
        else:
            st.warning("Please enter some text to analyze.")
    
    st.markdown('</div>', unsafe_allow_html=True)

# [Rest of the file including show_general_section remains unchanged]

def show_general_section(section_name: str = "analysis", subsections: Optional[List[str]] = None, section_key_prefix: str = "general_section") -> None:
    logger.info(f"show_general_section: Starting for section_name='{section_name}'")
    if not isinstance(section_name, str):
        logger.error(f"show_general_section: section_name is not a string: {type(section_name)}")
        raise TypeError(f"section_name must be a string, got {type(section_name)}")

    if subsections is None:
        subsections = ["Dataframe", "EDA", "Preprocessing", "Models", "Sentiment Prediction"]

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

    elif subsection_choice == "Sentiment Prediction":
        display_sentiment_prediction_section()

    logger.info(f"show_general_section: Finished for section_name='{section_name}'")

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    st.set_page_config(layout="wide")
    show_general_section(section_name="General")