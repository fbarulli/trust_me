import pandas as pd
import re
from textblob import TextBlob
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.over_sampling import SMOTE

def preprocess_reviews(review_file, categories_file, companies_file):
    # Load datasets
    df_categories = pd.read_csv(categories_file)
    df_companies = pd.read_csv(companies_file)
    df_reviews = pd.read_csv(review_file)
    
    # Handle missing values
    df_reviews['cust_review_text'] = df_reviews['cust_review_text'].fillna('')
    df_reviews['comment_length'] = df_reviews['cust_review_text'].apply(lambda x: len(str(x)))
    
    # Sentiment analysis
    df_reviews['sentiment'] = df_reviews['cust_review_text'].apply(lambda x: TextBlob(x).sentiment.polarity)
    
    # Convert text to lowercase
    df_reviews['cust_review_text'] = df_reviews['cust_review_text'].str.lower()
    
    # Remove special characters, numbers, and HTML tags
    df_reviews['cust_review_text'] = df_reviews['cust_review_text'].apply(lambda x: re.sub(r'[^a-zA-Z\s]', '', str(x)))
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    df_reviews['cust_review_text'] = df_reviews['cust_review_text'].apply(lambda x: ' '.join([word for word in x.split() if word not in stop_words]))
    
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    df_reviews['cust_review_text'] = df_reviews['cust_review_text'].apply(lambda x: ' '.join([lemmatizer.lemmatize(word) for word in x.split()]))
    df_reviews['cust_review_text'] = df_reviews['cust_review_text'].apply(lambda x: ' '.join([lemmatizer.lemmatize(word, pos=wordnet.VERB) for word in x.split()]))
    
    # Convert ratings to numeric
    df_reviews['cust_rating'] = pd.to_numeric(df_reviews['cust_rating'], errors='coerce')
    
    # Drop missing values
    df_reviews = df_reviews.dropna(subset=['cust_rating', 'cust_review_text'])
    
    # TF-IDF vectorization
    tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
    X = tfidf.fit_transform(df_reviews['cust_review_text'])
    
    # Target variable
    y = df_reviews['cust_rating']
    
    # SMOTE for balancing
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    
    return X_resampled, y_resampled, df_reviews
