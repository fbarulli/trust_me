import streamlit as st

st.title("TrustMe - An in-depth survey on Trutpilot on three different levels")  # layers instead of levels?!
st.write("Introduction - Text from Report 1 + Report 2")
if st.checkbox("Display"):
    st.header("The Nike Data Set - A Case Study")
    st.subheader("Brief Summary")
    st.write("As one of the biggest sports brands the company Nike was chosen to be the representative of the category"
             "Sports. Data was scraped from Nike's Trustpilot website ('https://www.trustpilot.com/review/www.nike.com')."
             "To date there are 10k+ reviews. The average star rating is 1.7 (qualified as “bad”)."
             "Obviously, the most prominent category is 1-star (73%), followed by 5-stars (16%),"
             "4- and 2-stars (both 4%) and finally 3-stars (3%). The data set is rather small. For practical reasons only"
             "ca. 5200 reviews were used for further analysis. (cf. Section 'EDA'). The fact that not all of the reviews"
             "(~1.6%) were in English language made it necessary to modify the standard scraping protocol which was used"
             "for retrieving the data of the sports category.")
    st.image("lang_ratings_nike_bar.png")
    st.image("lang_ratings_nike_pie.png")
    st.subheader("EDA")
    st.write("In this section, some key figures which characterize the data set at hand are presented.")
    st.dataframe(data_nike.head(10))  # data set with customer name and company name removed
    st.dataframe(data_nike.info())
    st.dataframe(data_nike.describe())
    st.image("countplot_ratings_nike_02.png")
    st.write("During the preprocessing stage, the same NLP methods were applied to the Nike data set as for the data of"
             "the Sports category. The most frequent words from the customer reviews are shown in a word cloud. Also, the"
             "result of the sentiment analysis is displayed in form form of a box plot.")
    st.image("word_cloud_nike_01.png")
    st.image("word_cloud_nike_02.png")
    st.image("sentiment_analysis_nike_02.png")
    st.subheader("Model Training and Evaluation")
    st.text("The following ML models were trained and evaluated: a simple logistic regression model, an SVM model, a Random"
            "Forest model and an XGBoost model. It was also tried to train a BERT model on the DPL level. However, due to"
            "a lack of a GPU and time contraints this step remained unsuccessful. Of all the ML models, the Random Forest approach"
            "performed best, followed by the SVM and XGBoost Model. But even the metrics for the logistic regression model"
            "were rather good. This might be due to the small size of the data set. Anyway, it seems that the SMOTE oversampling"
            " of the minor categories definitely helped to overcome the initial imbalances in the data.")

# List of common Streamlit elements
"""
st.title()  # displays a title
st.header()  # displays a second title
st.subheader()  # displays a third title
st.markdown()  # displays text in markdown format
st.code()  # displays code
st.image()  # display an image (this function takes a 3-dimensional np.array as argument)
st.write()  # display text or code (equivalent to print on a notebook)
st.dataframe()  # displays a dataframe
# Note: Classic dataframe code should be enclosed in this command. For example, if the base code is df.head(), we write st.dataframe(df.head()) to get the display on Streamlit.
st.button  # creates a button
st.checkbox()  # creates a checkbox to get the display
st.selectbox()  # creates a box with different options to get the selected display
st.slider()  # creates a slider to select a numeric value from a given range
select_slider()  # creates a slider to select a non-numeric value from a given range
"""