import streamlit as st

# --- Styling ---
st.markdown(
    """
    <style>
    /* Main background color */
    .stApp {
        background-color: #282c34; /* Dark Gray */
    }

    /* Sidebar button styling (radio buttons, not st.button) */
    .st-bx { /* Targeting the radio button container */
        color: #4a4a4a; /* Dark text for unselected buttons */
    }
    .st-bx input[type="radio"]:checked + div[data-baseweb="radio"] > div {
        background-color: #000000 !important; /* Black background for selected button */
        color: #ffffff !important; /* White text for selected button */
    }
    .st-bx input[type="radio"] + div[data-baseweb="radio"] > div {
       background-color: #ffffff;
       border-color: #000000
    }
    /* All text (except buttons, handled above) */
   .st-b7, .st-bb, .st-bc, .st-bd, .st-be, .st-bf, .st-bg, .st-bh, .st-bi, .st-bj, .st-bk, .st-bl, .st-bm, .st-bn, .st-bo, .st-bp, .st-bq, .st-br, .st-bs, .st-bt, .st-bu, .st-bv, .st-bw, .st-bx, .st-by, .st-bz, .st-c0, .st-c1, .st-c2, .st-c3, .st-c4, .st-c5, .st-c6, .st-c7, .st-c8,  .st-dc {
        color: #ffffff;
     }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- App Structure ---

st.title("Trust Me")

# Sidebar with buttons.  Crucially, we use index=0 to make "Intro" selected by default.
selected_section = st.sidebar.radio(
    "Choose a Section:",
    ["Intro", "Scrapping", "Company", "Category", "General"],
    index=0  # Set "Intro" as the default selection
)

# Main Content Area
if selected_section == "Intro":
    st.header("Intro Content")
    st.header("Header 1 (Large)")
    st.subheader("Header 2 (Medium)")
    st.text("Header 3 (Small)")

    st.markdown("---")
    st.header("Section 1")
    st.text("# Change here")

    st.markdown("---")
    st.header("Section 2")
    st.text("# Change here")

    st.markdown("---")
    st.header("Section 3")
    st.text("# Change here")

    # --- Image and DataFrame placeholders ---
    col1, col2, col3 = st.columns(3) #creating columns
    with col1:
      #st.image("/app/static/your_image.png")  # Uncomment and replace with your image path
      pass #added so that the code runs without errors

    with col2:
      #st.dataframe(df.head())  # Uncomment and replace with your DataFrame
      pass

    with col3:
      pass



elif selected_section == "Scrapping":
    st.header("Scrapping Content")
    st.header("Header 1 (Large)")
    st.subheader("Header 2 (Medium)")
    st.text("Header 3 (Small)")
     # --- Image and DataFrame placeholders ---
    col1, col2, col3 = st.columns(3)
    with col1:
      #st.image("/app/static/your_image.png")  # Uncomment and replace with your image path
      pass

    with col2:
      #st.dataframe(df.head())  # Uncomment and replace with your DataFrame
      pass

    with col3:
      pass


elif selected_section == "Company":
    st.header("Company Content")
    st.header("Header 1 (Large)")
    st.subheader("Header 2 (Medium)")
    st.text("Header 3 (Small)")
     # --- Image and DataFrame placeholders ---
    col1, col2, col3 = st.columns(3)
    with col1:
      #st.image("/app/static/your_image.png")  # Uncomment and replace with your image path
      pass

    with col2:
      #st.dataframe(df.head())  # Uncomment and replace with your DataFrame
      pass

    with col3:
      pass


elif selected_section == "Category":
    st.header("Category Content")
    st.header("Header 1 (Large)")
    st.subheader("Header 2 (Medium)")
    st.text("Header 3 (Small)")
     # --- Image and DataFrame placeholders ---
    col1, col2, col3 = st.columns(3)
    with col1:
      #st.image("/app/static/your_image.png")  # Uncomment and replace with your image path
      pass

    with col2:
      #st.dataframe(df.head())  # Uncomment and replace with your DataFrame
      pass

    with col3:
      pass


elif selected_section == "General":
    st.header("General Content")
    st.header("Header 1 (Large)")
    st.subheader("Header 2 (Medium)")
    st.text("Header 3 (Small)")
     # --- Image and DataFrame placeholders ---
    col1, col2, col3 = st.columns(3)
    with col1:
      #st.image("/app/static/your_image.png")  # Uncomment and replace with your image path
      pass

    with col2:
      #st.dataframe(df.head())  # Uncomment and replace with your DataFrame
      pass

    with col3:
      pass