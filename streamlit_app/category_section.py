# category_section.py
import streamlit as st
import pandas as pd
import os
from PIL import Image

APP_DIR = os.path.dirname(os.path.abspath(__file__)) # category_section.py's directory (same as stream.py)
PARENT_DIR = os.path.dirname(APP_DIR) # Parent directory of streamlit_app (trust-me-data-analysis)


# Function to resize and display images
def display_resized_image(image_path, caption, max_width=800):
    if os.path.exists(image_path):
        img = Image.open(image_path)
        img.thumbnail((max_width, max_width))  # Resize image
        st.image(img, caption=caption, use_container_width=True)
    else:
        st.warning(f"⚠️ Image not found: {image_path}")

def show_category_section(category_name="sports"): # Parameterized category name, default to "sports"
    st.header(f"📊 {category_name.capitalize()} Category Analysis") # Dynamic header

    # Sub-Selection for Category
    category_subsection = st.radio( # Renamed variable to avoid shadowing section name
        f"🔹 Choose a {category_name.capitalize()} Category Section:", # Dynamic label
        ["Dataframe", "EDA", "Preprocessing", "Models"],
        index=0
    )

    # --- Dataframe Section ---
    if category_subsection == "Dataframe":
        st.markdown('<div class="content-box">', unsafe_allow_html=True)
        st.subheader("📂 Data Overview")

        # Load DataFrame - Path updated to project root relative path
        df_path = os.path.join('trustpilot_reviews_1000.csv') # Adjusted relative path - project root

        if os.path.exists(df_path):
            df_rev = pd.read_csv(df_path)
            st.write("Displaying first **10** rows for better performance:")
            st.dataframe(df_rev.head(10))  # Display only first 10 rows
        else:
            st.warning(f"⚠️ CSV file not found: {df_path}")

        # DF Info Image
        with st.expander("📷 DF Info (Click to Expand)"):
            # Relative image path, assuming Images folder is within streamlit_app directory
            df_info_path = os.path.join('streamlit_app', "Images", f"df info {category_name}.png") # Relative image path
            display_resized_image(df_info_path, "DF Info")

        st.markdown("### 📝 Description")
        st.write("This section provides an overview of the dataset.")
        st.markdown('</div>', unsafe_allow_html=True)

    # --- EDA Section ---
    elif category_subsection == "EDA":
        st.markdown('<div class="content-box">', unsafe_allow_html=True)
        st.subheader("📊 Exploratory Data Analysis")

        # EDA Images
        eda_images = [
            ("Rating distribution", os.path.join('streamlit_app', "Images", f"Rating distribution {category_name}.png")), # Relative paths
            ("Sentiment distribution", os.path.join('streamlit_app', "Images", f"Sentiment {category_name}.png")),       # Relative paths
            ("Top 10 Word Ranking", os.path.join('streamlit_app', "Images", "Word ranking top 10.png")),     # Relative path
            ("Top 10 Negative Word Ranking", os.path.join('streamlit_app', "Images", "Word Ranking neg 10.png")), # Relative path
        ]

        with st.expander("📷 Exploratory Data Analysis (Click to Expand)"):
            for title, img_path in eda_images:
                display_resized_image(img_path, title)

        st.markdown("### 📝 Analysis")
        st.write("EDA visualizations provide insights into data distributions and trends.")
        st.markdown('</div>', unsafe_allow_html=True)

    # --- Preprocessing Section ---
    elif category_subsection == "Preprocessing":
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
        st.markdown('<div class="content-box">', unsafe_allow_html=True)
        st.subheader("🤖 Modeling")

        # Model Results Images
        model_images = [
            ("Logistic Regression Results", os.path.join('streamlit_app', "Images", "LR Ergebnis.png")), # Relative paths
            ("XGBoost Results", os.path.join('streamlit_app', "Images", "XGBoost Ergebnis.png")),      # Relative paths
        ]

        with st.expander("📷 Model Results (Click to Expand)"):
            for title, img_path in model_images:
                display_resized_image(img_path, title)

        st.markdown("### 📝 Description")
        st.write("These images showcase the model results.")
        st.markdown('</div>', unsafe_allow_html=True)