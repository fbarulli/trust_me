import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.svm import LinearSVC
from imblearn.under_sampling import RandomUnderSampler
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet
import numpy as np
import nltk
import logging
import asyncio
from typing import Dict, Tuple, List

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Download NLTK resources (moved outside function for efficiency)
nltk_resources = ['punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger']
for resource in nltk_resources:
    try:
        nltk.data.find(f'tokenizers/{resource}')
    except LookupError:
        nltk.download(resource, quiet=True)
    try:
        nltk.data.find(f'corpora/{resource}')
    except LookupError:
        nltk.download(resource, quiet=True)
    try:
        nltk.data.find(f'taggers/averaged_perceptron_tagger')
    except LookupError:
        nltk.download('averaged_perceptron_tagger', quiet=True)

# Text preprocessing
def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts."""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ, "N": wordnet.NOUN, "V": wordnet.VERB, "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)

def process_text(text: str, lemmatizer: WordNetLemmatizer) -> str:
    """Preprocess text by tokenizing, lowercasing, and lemmatizing."""
    text = str(text).lower()
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    tokens = [lemmatizer.lemmatize(word, get_wordnet_pos(word)) for word in tokens]
    return ' '.join([word for word in tokens if len(word) > 2])

def preprocess_texts(texts: pd.Series, lemmatizer: WordNetLemmatizer) -> List[str]:
    """Preprocess a list of texts."""
    return [process_text(text, lemmatizer) for text in texts]

def vectorize_data(vectorizer, X_train: List[str], X_test: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    """Vectorize training and test data."""
    X_train_vec = vectorizer.fit_transform(X_train).astype(np.float32)
    X_test_vec = vectorizer.transform(X_test).astype(np.float32)
    return X_train_vec, X_test_vec

async def train_and_evaluate_async(
    X_train: np.ndarray,
    y_train: pd.Series,
    X_test: np.ndarray,
    y_test: pd.Series,
    model_name: str,
    model
) -> Dict:
    """Asynchronously train and evaluate a model."""
    logging.info(f"Training {model_name}...")
    pipeline = Pipeline([('classifier', model)])
    try:
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test) if hasattr(model, "predict_proba") else None

        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0) # Handle potential zero division
        roc_auc = roc_auc_score(y_test, y_prob, multi_class='ovr', average='weighted') if y_prob is not None else None
        cm = confusion_matrix(y_test, y_pred)

        metrics = {
            "Model": model_name,
            "ROC AUC": roc_auc,
            "Confusion Matrix": cm,
            "accuracy": report.get("accuracy"),
        }
        for class_label, scores in report.items():
            if class_label not in ["accuracy", "macro avg", "weighted avg"]:
                for metric, value in scores.items():
                    metrics[f"{class_label}_{metric}"] = value
        logging.info(f"Finished training and evaluating {model_name}.")
        return metrics
    except Exception as e:
        logging.error(f"Error training {model_name}: {e}")
        return {"Model": model_name, "Error": str(e)}

async def run_text_classification(
    df: pd.DataFrame,
    text_column: str,
    target_column: str
) -> pd.DataFrame:
    """Main function to run text classification."""
    logging.info("Starting text classification process.")

    df = df.dropna(subset=[text_column, target_column])

    X = df[text_column]
    y = df[target_column] - 1  # Remap class labels to start from 0

    logging.info("Splitting data into training and testing sets.")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    logging.info("Applying undersampling to the training data.")
    rus = RandomUnderSampler(random_state=42)
    X_train_resampled, y_train_resampled = rus.fit_resample(pd.DataFrame(X_train), y_train)
    X_train_resampled = X_train_resampled.iloc[:, 0]

    logging.info("Initializing lemmatizer.")
    lemmatizer = WordNetLemmatizer()

    logging.info("Preprocessing text data.")
    X_train_processed = preprocess_texts(X_train_resampled, lemmatizer)
    X_test_processed = preprocess_texts(X_test, lemmatizer)

    models = {
        "Logistic Regression": LogisticRegression(random_state=42, max_iter=10000),
        "ElasticNet (Logistic)": LogisticRegression(random_state=42, penalty='elasticnet', solver='saga', l1_ratio=0.5, max_iter=10000),
        "Random Forest": RandomForestClassifier(random_state=42),
        "XGBoost": XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='mlogloss'),
        "LightGBM": LGBMClassifier(random_state=42),
        "LinearSVC": LinearSVC(random_state=42, dual=False)
    }

    vectorizers = {
        "CountVectorizer": CountVectorizer(),
        "TfidfVectorizer": TfidfVectorizer()
    }

    all_results = []
    async_tasks = []

    logging.info("Starting model training and evaluation loop.")
    for vectorizer_name, vectorizer in vectorizers.items():
        logging.info(f"Vectorizing data using {vectorizer_name}.")
        X_train_vec, X_test_vec = vectorize_data(vectorizer, X_train_processed, X_test_processed)
        for model_name, model in models.items():
            task = train_and_evaluate_async(
                X_train_vec, y_train_resampled, X_test_vec, y_test, model_name, model
            )
            async_tasks.append((vectorizer_name, model_name, task))

    model_results = await asyncio.gather(*[task for _, _, task in async_tasks])

    logging.info("Processing results.")
    for (vectorizer_name, model_name, _), result in zip(async_tasks, model_results):
        if "Error" in result:
            all_results.append({"Vectorizer": vectorizer_name, "Model": model_name, "Error": result["Error"]})
        else:
            result["Vectorizer"] = vectorizer_name
            all_results.append(result)

    results_df = pd.DataFrame(all_results)
    logging.info("Text classification process completed.")
    return results_df

