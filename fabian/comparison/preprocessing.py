from typing import Union, Optional, Tuple, Dict
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import traceback
import re
import threading
from tqdm.auto import tqdm
from scipy.sparse import hstack
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import lightgbm as lgb
import xgboost as xgb
from textblob import TextBlob
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer




logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def preprocess_reviews(review_data: Union[str, pd.DataFrame], chunk_size: int = 10000) -> pd.DataFrame:
    """
    Preprocesses review text data for sentiment analysis and feature extraction.
    Accepts either a CSV file path or a Pandas DataFrame as input and processes the "text" column.
    Returns only the processed DataFrame, without TF-IDF features or target variable.

    Args:
        review_data (Union[str, pd.DataFrame]): Path to the CSV file or a Pandas DataFrame
                                                 containing review data with a "text" column.
        chunk_size (int): Number of rows to process in each chunk (if loading from CSV).

    Returns:
        pd.DataFrame: Processed DataFrame with 'text', 'comment_length', and 'sentiment' columns.
    """
    logging.info(f"Starting preprocess_reviews function with chunk size: {chunk_size}")

    try:
        logging.info("Loading review dataset.")
        if isinstance(review_data, str):
            logging.info(f"Loading from CSV file: {review_data}")
            review_chunks = pd.read_csv(review_data, chunksize=chunk_size)
        elif isinstance(review_data, pd.DataFrame):
            logging.info("Using DataFrame object as input.")
            review_chunks = [review_data] 
        else:
            raise TypeError("Input review_data must be a CSV file path or a Pandas DataFrame.")

        processed_chunks = []

        lemmatizer = WordNetLemmatizer()
        stop_words = set(stopwords.words('english'))

        for chunk_df in tqdm(review_chunks, desc="Processing review chunks"):
            try:
                if 'text' not in chunk_df.columns:
                    raise ValueError("Input DataFrame or CSV must contain a 'text' column.")

                chunk_df['text'] = chunk_df['text'].astype(str)
                chunk_df['rating'] = pd.to_numeric(chunk_df['rating'], errors='coerce') 

                chunk_df['comment_length'] = chunk_df['text'].str.len()

                chunk_df['sentiment'] = chunk_df['text'].apply(lambda x: TextBlob(x).sentiment.polarity)

                chunk_df['text'] = chunk_df['text'].str.lower()
                chunk_df['text'] = chunk_df['text'].apply(lambda x: re.sub(r'[^a-zA-Z\s]', '', str(x)))

                def remove_stopwords_and_lemmatize(text: str) -> str:
                    words = text.split()
                    words = [word for word in words if word not in stop_words]
                    words = [lemmatizer.lemmatize(word) for word in words]
                    words = [lemmatizer.lemmatize(word, pos=wordnet.VERB) for word in words]
                    return ' '.join(words)

                chunk_df['text'] = chunk_df['text'].apply(remove_stopwords_and_lemmatize)

                chunk_df_cleaned = chunk_df.dropna(subset=['rating', 'text']) 
                processed_chunks.append(chunk_df_cleaned)

            except Exception as e_inner:
                logging.error(f"Error processing chunk: {e_inner}")
                logging.error(traceback.format_exc())

        df_reviews = pd.concat(processed_chunks, ignore_index=True)

        

        logging.info("Finished preprocess_reviews function")
        return df_reviews 

    except FileNotFoundError as fnfe:
        logging.error(f"File not found: {fnfe}")
        logging.error(traceback.format_exc())
        raise
    except TypeError as te:
        logging.error(f"Type error: {te}. Input must be a CSV file path or a Pandas DataFrame.")
        logging.error(traceback.format_exc())
        raise
    except ValueError as ve:
        logging.error(f"Value error: {ve}. Ensure DataFrame or CSV has 'text' column.") 
        logging.error(traceback.format_exc())
        raise
    except Exception as e_outer:
        logging.error(f"An error occurred during preprocessing: {e_outer}")
        logging.error(traceback.format_exc())
        raise



logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def compare_rating_sentiment(df: pd.DataFrame):
    logging.info("Starting compare_rating_sentiment function")

    try:
        average_sentiment_per_rating = df.groupby('rating')['sentiment'].mean().reset_index()
        average_sentiment_per_rating = average_sentiment_per_rating.sort_values(by='rating')

        plt.figure(figsize=(10, 6))

        sns.lineplot(x='rating', y='sentiment', data=average_sentiment_per_rating, marker='o')

        plt.title('Average Sentiment Score vs. Rating')
        plt.xlabel('Rating')
        plt.ylabel('Average Sentiment Score')
        plt.xticks(average_sentiment_per_rating['rating'].unique())
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)

        plt.tight_layout()
        sns.despine()
        plt.show()

        logging.info("Plot 'Average Sentiment Score vs. Rating' generated and displayed.")

    except KeyError as ke:
        logging.error(f"KeyError in compare_rating_sentiment: {ke}. Ensure DataFrame has 'rating' and 'sentiment' columns.")
        logging.error(traceback.format_exc())
        raise
    except Exception as e:
        logging.error(f"Error in compare_rating_sentiment: {e}")
        logging.error(traceback.format_exc())
        raise

    logging.info("Finished compare_rating_sentiment function")







def train_predict_evaluate_rating_lgbm(X_train, y_train, X_val, y_val, y_original) -> pd.DataFrame:
    """
    Trains a LightGBM classifier to predict 'rating'.
    Evaluates the model on a validation set and returns performance metrics in a DataFrame.

    Args:
        X_train: Training features.
        y_train: Training target.
        X_val: Validation features.
        y_val: Validation target.
        y_original: Original target series for class count.

    Returns:
        pd.DataFrame: DataFrame containing validation accuracy, macro F1 score, and macro ROC AUC.
    """
    results_df = pd.DataFrame(columns=['Metric', 'Value', 'Model']) 

    try:
        lgbm_model = lgb.LGBMClassifier(random_state=42, objective='multiclass', num_class=len(y_original.unique()), verbose=-1) 
        lgbm_model.fit(X_train, y_train,
                       eval_set=[(X_val, y_val)],
                       eval_metric='multi_logloss',
                       callbacks=[lgb.callback.log_evaluation(period=0),
                                  lgb.callback.early_stopping(stopping_rounds=20, verbose=-1)])

        y_pred_val = lgbm_model.predict(X_val)

        accuracy_val = accuracy_score(y_val, y_pred_val)
        f1_score_val_macro = f1_score(y_val, y_pred_val, average='macro')
        y_prob_val = lgbm_model.predict_proba(X_val)
        roc_auc_val_macro = roc_auc_score(y_val, y_prob_val, multi_class='ovr', average='macro')


        results_df = pd.concat([results_df, pd.DataFrame([{'Metric': 'Validation Accuracy', 'Value': accuracy_val, 'Model': 'LGBM'}])], ignore_index=True)
        results_df = pd.concat([results_df, pd.DataFrame([{'Metric': 'Validation Macro F1 Score', 'Value': f1_score_val_macro, 'Model': 'LGBM'}])], ignore_index=True)
        results_df = pd.concat([results_df, pd.DataFrame([{'Metric': 'Validation Macro ROC AUC', 'Value': roc_auc_val_macro, 'Model': 'LGBM'}])], ignore_index=True)

        return results_df

    except Exception as e:
        error_df = pd.DataFrame([{'Metric': 'Error', 'Value': str(e), 'Model': 'LGBM'}])
        return error_df


def train_predict_evaluate_rating_xgb(X_train, y_train, X_val, y_val, y_original) -> pd.DataFrame:
    """
    Trains an XGBoost classifier to predict 'rating'.
    Evaluates the model on a validation set and returns performance metrics in a DataFrame.

    Args:
        X_train: Training features.
        y_train: Training target.
        X_val: Validation features.
        y_val: Validation target.
        y_original: Original target series for class count.

    Returns:
        pd.DataFrame: DataFrame containing validation accuracy, macro F1 score, and macro ROC AUC.
    """
    results_df = pd.DataFrame(columns=['Metric', 'Value', 'Model']) 

    try:
        xgb_model = xgb.XGBClassifier(random_state=42, objective='multi:softmax', num_class=len(y_original.unique()), eval_metric='mlogloss', verbose=0) 
        xgb_model.fit(X_train, y_train,
                      eval_set=[(X_val, y_val)],
                      verbose=0) 

        y_pred_val = xgb_model.predict(X_val)

        accuracy_val = accuracy_score(y_val, y_pred_val)
        f1_score_val_macro = f1_score(y_val, y_pred_val, average='macro')

        y_prob_val = xgb_model.predict_proba(X_val)
        roc_auc_val_macro = roc_auc_score(y_val, y_prob_val, multi_class='ovr', average='macro')


        results_df = pd.concat([results_df, pd.DataFrame([{'Metric': 'Validation Accuracy', 'Value': accuracy_val, 'Model': 'XGBoost'}])], ignore_index=True)
        results_df = pd.concat([results_df, pd.DataFrame([{'Metric': 'Validation Macro F1 Score', 'Value': f1_score_val_macro, 'Model': 'XGBoost'}])], ignore_index=True)
        results_df = pd.concat([results_df, pd.DataFrame([{'Metric': 'Validation Macro ROC AUC', 'Value': roc_auc_val_macro, 'Model': 'XGBoost'}])], ignore_index=True)

        return results_df

    except Exception as e:
        error_df = pd.DataFrame([{'Metric': 'Error', 'Value': str(e), 'Model': 'XGBoost'}])
        return error_df


def train_evaluate_concurrently(df: pd.DataFrame) -> pd.DataFrame:
    """
    Trains and evaluates LightGBM and XGBoost classifiers concurrently and returns combined metrics in a DataFrame.

    Args:
        df (pd.DataFrame): DataFrame with 'rating', 'text', 'comment_length', and 'sentiment' columns.

    Returns:
        pd.DataFrame: DataFrame containing validation metrics for both LGBM and XGBoost.
    """
    try:
        X_text = df['text'].astype(str)
        X_num = df[['comment_length', 'sentiment']].astype(float)
        y_original = df['rating'].astype(int)
        y = y_original - 1

        tfidf_vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        X_text_tfidf = tfidf_vectorizer.fit_transform(X_text)

        scaler = StandardScaler()
        X_num_scaled = scaler.fit_transform(X_num)
        X_combined = hstack([X_text_tfidf, X_num_scaled])

        X_train_temp, X_val_test, y_train_temp, y_val_test = train_test_split(
            X_combined, y, test_size=0.4, random_state=42, stratify=y)
        X_val, X_test, y_val, y_test = train_test_split(
            X_val_test, y_val_test, test_size=0.5, random_state=42, stratify=y_val_test)

        X_train = X_train_temp
        y_train = y_train_temp


        lgbm_thread = threading.Thread(target=train_predict_evaluate_rating_lgbm, args=(X_train, y_train, X_val, y_val, y_original))
        xgb_thread = threading.Thread(target=train_predict_evaluate_rating_xgb, args=(X_train, y_train, X_val, y_val, y_original))

        lgbm_thread.start()
        xgb_thread.start()

        lgbm_results = None
        xgb_results = None

        lgbm_thread.join() 
        xgb_thread.join()

        lgbm_results = train_predict_evaluate_rating_lgbm(X_train, y_train, X_val, y_val, y_original) 
        xgb_results = train_predict_evaluate_rating_xgb(X_train, y_train, X_val, y_val, y_original)


        combined_results_df = pd.concat([lgbm_results, xgb_results], ignore_index=True)
        return combined_results_df


    except Exception as e:
        error_df = pd.DataFrame([{'Metric': 'Error', 'Value': str(e), 'Model': 'Concurrent Training'}])
        return error_df
    





def calculate_lexical_diversity_optimized(df_or_path: Union[pd.DataFrame, str], window_size: int = 50, col_name: Optional[str] = None) -> tuple[float, Optional[str]]:
    """
    Calculates Moving Average Type-Token Ratio (MATTR) on a text column
    in a Pandas DataFrame or CSV file.

    Args:
        df_or_path: Pandas DataFrame or path to a CSV file.
        window_size: Window size for MATTR calculation (number of tokens in each window).
        col_name: Optional column name to use for calculation. If None, the function
                  will automatically select the longest text column (object dtype).

    Returns:
        A tuple containing:
        - MATTR score (float) or NaN if no suitable text column is found.
        - Column name used for calculation (str or None if no column was used).
    """

    if isinstance(df_or_path, str):
        try:
            df = pd.read_csv(df_or_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"CSV file not found at: {df_or_path}")
    elif isinstance(df_or_path, pd.DataFrame):
        df = df_or_path
    else:
        raise TypeError("Input must be a Pandas DataFrame or a path to a CSV file.")

    object_cols = df.select_dtypes(include='object').columns

    if col_name:
        if col_name not in df.columns:
            raise ValueError(f"Column name '{col_name}' not found in DataFrame.")
        if df[col_name].dtype != 'object':
            print(f"Warning: Column '{col_name}' is not of 'object' dtype. Proceeding, but ensure it contains text data.")
        longest_text_col = col_name
    else:
        if not object_cols.any():
            return np.nan, None
        if len(object_cols) > 1:
            avg_lengths = df[object_cols].apply(lambda col: col.str.len().mean())
            longest_text_col = avg_lengths.idxmax()
        else:
            longest_text_col = object_cols[0]

    text_series = df[longest_text_col].astype(str)

    def calculate_mattr_series(text_series, window_size):
        """Calculates MATTR for a Pandas Series of text, optimized for vector operations."""
        tokenized_series = text_series.str.lower().str.findall(r'\w+')
        def get_ttr_window(tokens):
            if not tokens:
                return 0.0
            return len(set(tokens)) / len(tokens)
        mattr_scores = []
        for tokens_list in tokenized_series:
            if not tokens_list:
                mattr_scores.append(0.0)
            else:
                window_ttrs = []
                for i in range(0, len(tokens_list) - window_size + 1, 1):
                    window = tokens_list[i:i + window_size]
                    window_ttrs.append(get_ttr_window(window))
                if window_ttrs:
                    mattr_scores.append(np.mean(window_ttrs))
                else:
                    mattr_scores.append(0.0)
        return np.nanmean(mattr_scores)

    mattr_value = calculate_mattr_series(text_series, window_size)

    return mattr_value, longest_text_col







logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def count_words_in_csv(csv_path: str, output_csv_path: str, chunk_size: int = 10000, col_name: Optional[str] = None):
    """
    Counts word frequencies in a specified or automatically detected text column of a CSV file.
    Optimized for large datasets using chunking and vectorization.

    Args:
        csv_path (str): Path to the input CSV file.
        output_csv_path (str): Path to save the output CSV file with word counts.
        chunk_size (int): Number of rows to process in each chunk.
        col_name (Optional[str]):  Optional column name to use. If None, the function
                                  will automatically select the first 'object' dtype column.
    """
    logging.info(f"Starting count_words_in_csv function with chunk size: {chunk_size}, CSV: {csv_path}")

    total_word_counts = pd.Series(dtype='int64')

    try:
        logging.info(f"Reading CSV in chunks from: {csv_path}")
        csv_chunks = pd.read_csv(csv_path, chunksize=chunk_size)

        for chunk_df in tqdm(csv_chunks, desc="Processing CSV chunks"):
            try:
                object_cols = chunk_df.select_dtypes(include='object').columns

                if col_name:
                    if col_name not in chunk_df.columns:
                        raise ValueError(f"Specified column name '{col_name}' not found in CSV.")
                    text_column_name = col_name
                elif not object_cols.empty:
                    text_column_name = object_cols[0]
                    logging.info(f"Automatically selected text column: {text_column_name}")
                else:
                    logging.warning("No 'object' dtype column found in chunk. Skipping chunk word counting.")
                    continue

                text_series = chunk_df[text_column_name].astype(str).str.lower()
                word_series = text_series.str.findall(r'\w+').explode()

                chunk_word_counts = word_series.value_counts()
                total_word_counts = total_word_counts.add(chunk_word_counts, fill_value=0)

            except Exception as chunk_err:
                logging.error(f"Error processing a chunk: {chunk_err}")
                logging.error(traceback.format_exc())

        logging.info("Finished processing all chunks. Creating output DataFrame.")
        output_df = total_word_counts.reset_index()
        output_df.columns = ['word', 'count']
        output_df = output_df.sort_values(by='count', ascending=False)

        logging.info(f"Writing word counts to CSV: {output_csv_path}")
        output_df.to_csv(output_csv_path, index=False)
        logging.info(f"Word count CSV file created at: {output_csv_path}")

    except FileNotFoundError as fnfe:
        logging.error(f"File not found: {fnfe}")
        logging.error(traceback.format_exc())
        raise
    except ValueError as ve:
        logging.error(f"ValueError: {ve}")
        logging.error(traceback.format_exc())
        raise
    except Exception as e:
        logging.error(f"An error occurred in count_words_in_csv: {e}")
        logging.error(traceback.format_exc())
        raise

    logging.info("Finished count_words_in_csv function.")










tqdm.pandas() 

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def filter_words_dataframe(word_count_csv_path: str, object_df_or_path: Union[pd.DataFrame, str],
                           text_col_name: Optional[str] = None, count_threshold: int = 10) -> pd.DataFrame:
    """
    Loads a word count CSV, filters out low-count words from a specified text column
    of another DataFrame (or CSV), and returns the modified DataFrame.

    Args:
        word_count_csv_path (str): Path to CSV file containing 'word' and 'count' columns.
        object_df_or_path (Union[pd.DataFrame, str]): Pandas DataFrame or path to CSV
                                                      containing the text column to filter.
        text_col_name (Optional[str]): Name of the text column (object dtype) to filter in object_df.
                                        If None, the function will select the first 'object' column.
        count_threshold (int): Threshold count below which words are considered low-count and filtered out.

    Returns:
        pd.DataFrame: DataFrame with the specified text column filtered to remove low-count words.
                      Returns the original DataFrame if no suitable text column is found or errors occur.
    """
    logging.info(f"Starting filter_words_dataframe function. Word count CSV: {word_count_csv_path}, Threshold: {count_threshold}")

    try:
        logging.info(f"Loading word count CSV from: {word_count_csv_path}")
        word_counts_df = pd.read_csv(word_count_csv_path)

        if not all(col in word_counts_df.columns for col in ['word', 'count']):
            raise ValueError("Word count CSV must contain columns 'word' and 'count'.")

        low_count_words_series = word_counts_df[word_counts_df['count'] < count_threshold]['word']
        low_count_words_set = set(low_count_words_series)
        logging.info(f"Found {len(low_count_words_set)} low-count words (count < {count_threshold}).")

        if isinstance(object_df_or_path, str):
            logging.info(f"Loading object DataFrame from CSV: {object_df_or_path}")
            object_df = pd.read_csv(object_df_or_path)
        elif isinstance(object_df_or_path, pd.DataFrame):
            object_df = object_df_or_path
        else:
            raise TypeError("object_df_or_path must be a Pandas DataFrame or a path to a CSV file.")

        object_cols = object_df.select_dtypes(include='object').columns

        if text_col_name:
            if text_col_name not in object_df.columns:
                raise ValueError(f"Specified text column '{text_col_name}' not found in object DataFrame.")
            target_text_col = text_col_name
        elif not object_cols.empty:
            target_text_col = object_cols[0]
            logging.info(f"Automatically selected text column for filtering: {target_text_col}")
        else:
            logging.warning("No 'object' dtype column found in object DataFrame. Returning original DataFrame.")
            return object_df

        logging.info(f"Filtering low-count words from column: '{target_text_col}'")

        def remove_low_count_words(text: str) -> str:
            if not isinstance(text, str):
                return text
            words = text.split()
            filtered_words = [word for word in words if word not in low_count_words_set]
            return ' '.join(filtered_words)

        object_df[target_text_col] = object_df[target_text_col].progress_apply(remove_low_count_words) 
        logging.info(f"Low-count words filtering applied to column '{target_text_col}'.")

        logging.info("Finished filter_words_dataframe function.")
        return object_df

    except FileNotFoundError as fnfe:
        logging.error(f"File not found: {fnfe}")
        logging.error(traceback.format_exc())
        raise
    except ValueError as ve:
        logging.error(f"ValueError: {ve}")
        logging.error(traceback.format_exc())
        raise
    except TypeError as te:
        logging.error(f"TypeError: {te}")
        logging.error(traceback.format_exc())
        raise
    except Exception as e:
        logging.error(f"An unexpected error occurred in filter_words_dataframe: {e}")
        logging.error(traceback.format_exc())
        raise