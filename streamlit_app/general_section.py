# general_section.py
import streamlit as st
import pandas as pd
import os
from PIL import Image
import logging
from typing import Optional, List, Tuple
from sentiment_model import predict_sentiment
import time

# Set up logging for this module
logger = logging.getLogger(__name__)

APP_DIR: str = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR: str = os.path.dirname(APP_DIR)
IMAGES_DIR: str = os.path.join(APP_DIR, 'Images')
SCRAPE_1_DIR: str = os.path.join(IMAGES_DIR, 'fabian_img', 'ngram_sentiment', 'scrape_1')
SCRAPE_2_DIR: str = os.path.join(IMAGES_DIR, 'fabian_img', 'ngram_sentiment', 'scrape_2')

# Define all functions before show_general_section
def display_resized_image(image_path: str, caption: str, max_width: int = 800) -> None:
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
            original_width, original_height = img.size
            new_width = original_width
            new_width = min(new_width, max_width * 2)
            st.image(img, caption=caption, width=new_width)
            logger.debug(f"display_resized_image: Displayed image with original_width={original_width}, new_width={new_width}")
        except Exception as e:
            logger.exception(f"display_resized_image: Error processing image '{image_path}': {e}")
            st.error(f"Error displaying image. Check logs for details.")
    else:
        st.warning(f"⚠️ Image not found: {image_path}")
        logger.warning(f"display_resized_image: Image not found: {image_path}")
    logger.info(f"display_resized_image: Finished for image_path='{image_path}'")

def display_dataframe_section(section_name: str, csv_path: str) -> None:
    logger.debug("display_dataframe_section: Displaying Dataframe subsection")
    st.markdown("""
    <style>
    .content-box {
        width: 1400px;
        max-width: 100%;
        margin: 0 auto;
    }
    </style>
    <div class="content-box">""", unsafe_allow_html=True)

    logger.debug(f"display_dataframe_section: DataFrame loading is disabled in this version.")
    if os.path.exists(csv_path):
        try:
            pd.read_csv(csv_path)
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
    logger.debug("display_eda_section: Displaying EDA subsection")
    st.markdown('<div class="content-box">', unsafe_allow_html=True)
    st.subheader("📊 Exploratory Data Analysis")

    if os.path.exists(SCRAPE_2_DIR):
        scrape_2_images = [(f"Subsequent Scrape: {os.path.splitext(img_file)[0].replace('_', ' ').title()}", 
                            os.path.join(SCRAPE_2_DIR, img_file)) 
                           for img_file in os.listdir(SCRAPE_2_DIR) 
                           if img_file.lower().endswith(('.png', '.jpg', '.jpeg'))]

        scrape_2_images.sort(key=lambda x: x[1].split(os.sep)[-1])

        if scrape_2_images:
            with st.expander("📷 Scrape Comparison Plots (Click to Expand)"):
                st.write("We can see here both the unprocessed df vs the processed df")
                for title, img_path in scrape_2_images:
                    display_resized_image(img_path, title)
                st.write("As we can see, most words appear with good and bad sentiment, making this a difficult task.")
        else:
            st.info("No scrape comparison images found in the specified directory.")
    else:
        st.warning(f"⚠️ Subsequent scrape directory not found: {SCRAPE_2_DIR}")

    st.markdown('</div>', unsafe_allow_html=True)

def display_models_section(model_image_paths: Optional[List[Tuple[str, str]]] = None) -> None:
    """Displays the Models subsection."""
    logger.debug("display_models_section: Displaying Models subsection")
    st.markdown('<div class="content-box">', unsafe_allow_html=True)
    st.subheader("🤖 Modeling")

    # Define the new image path
    default_image = [("Model Training Results", os.path.join(IMAGES_DIR, "fabian_img", "model_training_res.png"))]

    # Use provided model_image_paths if available, otherwise use the default
    images_to_display = model_image_paths if model_image_paths is not None else default_image

    if images_to_display:
        with st.expander("📷 Model Results (Click to Expand)"):
            st.write("I was not able to fully optimize performance, nor was I able to fully understand why")
            for title, img_path in images_to_display:
                logger.debug(f"display_models_section: Trying to display image: {img_path}")
                display_resized_image(img_path, title)
    else:
        st.info("No model result images provided for this section.")

    st.markdown("### 📝 Description")
    st.write("These images showcase the model results.")
    st.markdown('</div>', unsafe_allow_html=True)

def display_sentiment_prediction_section():
    """Displays the Sentiment Prediction subsection with a progress bar."""
    logger.debug("display_sentiment_prediction_section: Displaying Sentiment Prediction subsection")
    st.markdown('<div class="content-box">', unsafe_allow_html=True)
    st.subheader("🔍 Sentiment Prediction")

    # User input
    user_input = st.text_area("Enter your text here:", height=150, placeholder="Type your review or comment...")
    
    if st.button("Predict Sentiment"):
        if user_input.strip():
            try:
                # Define the model path
                model_path = os.path.join(APP_DIR, "model", "fabian_model", "final_model.pt")
                
                # Create a progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()
                status_text.text("Loading model and predicting sentiment...")

                # Simulate progress
                for i in range(100):
                    time.sleep(0.01)
                    progress_bar.progress(i + 1)
                
                # Predict sentiment
                result = predict_sentiment(user_input, model_path=model_path)
                
                # Clear progress bar and update status
                progress_bar.progress(100)
                status_text.text("Prediction complete!")

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

def display_conclusion_section():
    """Displays the Conclusion subsection."""
    logger.debug("display_conclusion_section: Displaying Conclusion subsection")
    st.markdown('<div class="content-box">', unsafe_allow_html=True)
    st.subheader("📌 Conclusion")

    st.markdown("""
- **Models**:  
I started with traditional ML, testing several boosting models, SVM, and Linear Regression, and Attention-based Deep Learning models. From there, I chose the best-performing ML models for baselines and for Optuna trials, and the same for Deep Learning. Some of the DL models used were:  
*{LoRA on big BERT, Roberta, Electra, BERT, SA-BERT}*

- **Preprocessing**:  
After several trials and experimentation with preprocessing and unprocessed data, performance never really improved.
There were many instances of single words that potentially could have been used in some other way, but I mainly focused on:  
*{Standardizing dates, contractions, websites, and special characters. Non-alphanumeric word removal, words with a less than 10 count}*

- **Feature Engineering**:  
Several techniques were used to boost performance:  
*{CountV, TFIDF, stop word removal, sentiment intensity, text feature extraction, text summarization, Topic modeling, NGRAM, NER, POS, VADER, Embeddings, data augmentation, embedding fine-tuning with whole word masking and Span Masking, InfoNCELoss}*  
were tried only to, in most cases, decrease performance. No feature engineering improved ML score, and the only feature that boosted performance significantly was embeddings for BERT.

- **Training**:  
With the initial dataset, I was able to get up to 80%, accuracy usign ML models. With the same dataset, deep learning models scored 60% at best. However with the second dataset, 
performed 40% worse than the first dataset, and deep learning model performance did not change.                 

- **Conclusion**:  
In conclusion, the majority (90%) of words only appear less than 10 times in the dataset. In theory, there are a lot of specific mentions of products from all categories—be that services, experiences, products, etc.—that I was not able to fully account for. Even when using pre-trained models with sentiment awareness and feature engineering, a better score was not possible to achieve.  
All in all, to effectively tackle this problem, a specific model for each category seems to yield the best result.
    """)

    st.markdown('</div>', unsafe_allow_html=True)

def show_general_section(section_name: str = "analysis", subsections: Optional[List[str]] = None, section_key_prefix: str = "general_section") -> None:
    logger.info(f"show_general_section: Starting for section_name='{section_name}'")
    if not isinstance(section_name, str):
        logger.error(f"show_general_section: section_name is not a string: {type(section_name)}")
        raise TypeError(f"section_name must be a string, got {type(section_name)}")

    if subsections is None:
        subsections = ["Dataframe", "EDA", "Models", "Sentiment Prediction", "Conclusion"]

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

    elif subsection_choice == "Models":
        display_models_section()

    elif subsection_choice == "Sentiment Prediction":
        display_sentiment_prediction_section()

    elif subsection_choice == "Conclusion":
        display_conclusion_section()

    logger.info(f"show_general_section: Finished for section_name='{section_name}'")

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    st.set_page_config(layout="wide")
    show_general_section(section_name="General")