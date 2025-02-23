


import streamlit as st
import pandas as pd
import os
from PIL import Image

# Function to resize and display images
def display_resized_image(image_path, caption, max_width=800):
    if os.path.exists(image_path):
        img = Image.open(image_path)
        img.thumbnail((max_width, max_width))  # Resize image
        st.image(img, caption=caption, use_container_width=True)
    else:
        st.warning(f"⚠️ Image not found: {image_path}")

def show_category_section():
    st.header("📊 Category Analysis")

    # Sub-Selection for Category
    category_section = st.radio(
        "🔹 Choose a Category Section:",
        ["Dataframe", "EDA", "Preprocessing", "Models"],
        index=0
    )

    # --- Dataframe Section ---
    if category_section == "Dataframe":
        st.markdown('<div class="content-box">', unsafe_allow_html=True)
        st.subheader("📂 Data Overview")

        # Load DataFrame
        df_path = "trustpilot_reviews_1000.csv" # Path updated: relative to streamlit/
        if os.path.exists(df_path):
            df_rev = pd.read_csv(df_path)
            st.write("Displaying first **10** rows for better performance:")
            st.dataframe(df_rev.head(10))  # Display only first 10 rows
        else:
            st.warning(f"⚠️ CSV file not found: {df_path}")

        # DF Info Image
        with st.expander("📷 DF Info (Click to Expand)"):
            df_info_path = "Images/df info sports.png" # Path updated: relative to streamlit/
            display_resized_image(df_info_path, "DF Info")

        st.markdown("### 📝 Description")
        st.write("This section provides an overview of the dataset.")
        st.markdown('</div>', unsafe_allow_html=True)

    # --- EDA Section ---
    elif category_section == "EDA":
        st.markdown('<div class="content-box">', unsafe_allow_html=True)
        st.subheader("📊 Exploratory Data Analysis")

        # EDA Images
        eda_images = [
            ("Rating distribution", "Images/Rating distribution sports.png"), # Path updated
            ("Sentiment distribution", "Images/Sentiment sports.png"),       # Path updated
            ("Top 10 Word Ranking", "Images/Word ranking top 10.png"),     # Path updated
            ("Top 10 Negative Word Ranking", "Images/Word Ranking neg 10.png"), # Path updated
        ]

        with st.expander("📷 Exploratory Data Analysis (Click to Expand)"):
            for title, img_path in eda_images:
                display_resized_image(img_path, title)

        st.markdown("### 📝 Analysis")
        st.write("EDA visualizations provide insights into data distributions and trends.")
        st.markdown('</div>', unsafe_allow_html=True)

    # --- Preprocessing Section ---
    elif category_section == "Preprocessing":
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
    elif category_section == "Models":
        st.markdown('<div class="content-box">', unsafe_allow_html=True)
        st.subheader("🤖 Modeling")

        # Model Results Images
        model_images = [
            ("Logistic Regression Results", "Images/LR Ergebnis.png"), # Path updated
            ("XGBoost Results", "Images/XGBoost Ergebnis.png"),      # Path updated
        ]

        with st.expander("📷 Model Results (Click to Expand)"):
            for title, img_path in model_images:
                display_resized_image(img_path, title)

        st.markdown("### 📝 Description")
        st.write("These images showcase the model results.")
        st.markdown('</div>', unsafe_allow_html=True)