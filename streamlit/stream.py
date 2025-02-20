import streamlit as st
import pandas as pd
import os
from PIL import Image

# --- Custom Styling ---
st.markdown(
    """
    <style>
    /* Force white background */
    html, body, [data-testid="stAppViewContainer"], .stApp {
        background-color: white !important;
        color: black !important;
    }

    /* Ensure all text is visible */
    .stText, .stTitle, .stHeader, .stMarkdown, .stDataFrame, .stTable {
        color: black !important;
    }

    /* Sidebar */
    .css-1d391kg {
        background-color: #0e76a8 !important;
        color: white !important;
    }

    .css-1d391kg h1 {
        color: white !important;
    }

    /* Radio Buttons Styling */
    .st-bx input[type="radio"]:checked + div[data-baseweb="radio"] > div {
        background-color: #0e76a8 !important;
        color: #ffffff !important;
    }

    /* Card-Like Content Box */
    .content-box {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- App Structure ---
st.title("🚀 Trust Me - Data Analysis")

# Sidebar Navigation
selected_section = st.sidebar.radio(
    "📌 Choose a Section:",
    ["Intro", "Scraping", "Company", "Category", "General"],
    index=0
)

# Function to resize and display images
def display_resized_image(image_path, caption, max_width=800):
    if os.path.exists(image_path):
        img = Image.open(image_path)
        img.thumbnail((max_width, max_width))  # Resize image
        st.image(img, caption=caption, use_container_width=True)
    else:
        st.warning(f"⚠️ Image not found: {image_path}")

# Main Content Area
if selected_section == "Category":
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
        df_path = "streamlit/trustpilot_reviews_1000.csv"
        if os.path.exists(df_path):
            df_rev = pd.read_csv(df_path)
            st.write("Displaying first **10** rows for better performance:")
            st.dataframe(df_rev.head(10))  # Display only first 10 rows
        else:
            st.warning(f"⚠️ CSV file not found: {df_path}")

        # DF Info Image
        with st.expander("📷 DF Info (Click to Expand)"):
            df_info_path = "streamlit/Images/df info sports.png"
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
            ("Rating distribution", "streamlit/Images/Rating distribution sports.png"),
            ("Sentiment distribution", "streamlit/Images/Sentiment sports.png"),
            ("Top 10 Word Ranking", "streamlit/Images/Word ranking top 10.png"),
            ("Top 10 Negative Word Ranking", "streamlit/Images/Word Ranking neg 10.png"),
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
            ("Logistic Regression Results", "streamlit/Images/LR Ergebnis.png"),
            ("XGBoost Results", "streamlit/Images/XGBoost Ergebnis.png"),
        ]

        with st.expander("📷 Model Results (Click to Expand)"):
            for title, img_path in model_images:
                display_resized_image(img_path, title)

        st.markdown("### 📝 Description")
        st.write("These images showcase the model results.")
        st.markdown('</div>', unsafe_allow_html=True)

else:
    st.write(f"🚀 {selected_section} Section Selected. (Not yet customized)")
