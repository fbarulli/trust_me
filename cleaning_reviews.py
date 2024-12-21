import pandas as pd
import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk
import logging

# Download necessary NLTK data files (run once)
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load the dataset
df = pd.read_csv("trustpilot_reviews.csv", on_bad_lines="skip")

# Define a text cleaning function
def clean_text(text):
    # Handle NaN values
    if pd.isnull(text):
        return ""

    try:
        # Convert to lowercase
        text = text.lower()

        # Remove URLs
        text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)

        # Remove HTML tags
        text = re.sub(r"<.*?>", "", text)

        # Remove special characters, punctuation, and numbers
        text = re.sub(r"[^\w\s]", "", text)
        text = re.sub(r"\d+", "", text)

        # Remove non-ASCII characters
        text = text.encode("ascii", "ignore").decode()

        # Tokenize text
        words = word_tokenize(text)

        # Remove stopwords
        stop_words = set(stopwords.words("english"))
        words = [word for word in words if word not in stop_words]

        # Lemmatize words
        lemmatizer = WordNetLemmatizer()
        words = [lemmatizer.lemmatize(word) for word in words]

        # Rejoin words into a single string
        cleaned_text = " ".join(words)

        return cleaned_text

    except Exception as e:
        logging.error(f"Error cleaning text: {e}")
        return ""

# Apply the cleaning function to the review text column
df["cleaned_review_text"] = df["customer_review_text"].apply(clean_text)

# Save the cleaned dataset
df.to_csv("cleaned_trustpilot_reviews.csv", index=False)

print("Review texts cleaned and saved to 'cleaned_trustpilot_reviews.csv'")