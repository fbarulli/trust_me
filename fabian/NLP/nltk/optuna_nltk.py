import optuna
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import re
from scipy.sparse import csr_matrix
import mlflow
import mlflow.sklearn
import emoji
from nltk.tokenize import word_tokenize
import functools

def remove_emojis(text):
  return ''.join(c for c in text if c not in emoji.UNICODE_EMOJI['en'])

class TextPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, use_stem=False, use_lem=False, use_stop=False, use_regex=False, stop_words = None):
        self.use_stem = use_stem
        self.use_lem = use_lem
        self.use_stop = use_stop
        self.use_regex = use_regex
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = stop_words

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        texts = X.copy()
        if self.use_regex:
            texts = texts.str.lower()
            texts = texts.apply(lambda x: re.sub(r'[^a-zA-Z\s]', '', x))
        
        texts = texts.apply(remove_emojis)
        
        def process_text(text):
          words = word_tokenize(text)
          if self.use_stop:
            words = [w for w in words if w not in self.stop_words]
          if self.use_stem:
            words = [self.stemmer.stem(w) for w in words]
          if self.use_lem:
            words = [self.lemmatizer.lemmatize(w) for w in words]
          return " ".join(words)
        
        texts = texts.apply(process_text)
        return texts

def objective(trial, X_train, y_train, X_val, y_val, stop_words):
    preprocessing_config = {
        'use_stem': trial.suggest_categorical('use_stem', [True, False]),
        'use_lem': trial.suggest_categorical('use_lem', [True, False]),
        'use_stop': trial.suggest_categorical('use_stop', [True, False]),
        'use_regex': trial.suggest_categorical('use_regex', [True, False]),
    }
    
    vectorizer_config = {
        'ngram_range': (1, trial.suggest_int('ngram_max', 1, 3)),
        'min_df': trial.suggest_float('min_df', 0.00001, 0.1),
        'max_df': trial.suggest_float('max_df', 0.5, 1.0),
        'max_features': trial.suggest_int('max_features', 1000, 25000)
    }

    gb_config = {
       'n_estimators': trial.suggest_int('n_estimators', 50, 300),
       'learning_rate': trial.suggest_float('learning_rate', 0.01, 1.0, log=True),
       'max_depth': trial.suggest_int('max_depth', 1, 7),
       'min_samples_split': trial.suggest_int('min_samples_split', 2, 150),
       'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 60),
       'validation_fraction': 0.1,
        'n_iter_no_change': 5,
        'tol': 0.01
   }
    
    preprocessor = TextPreprocessor(**preprocessing_config, stop_words = stop_words)
    X_train_processed = preprocessor.fit_transform(X_train)
    X_val_processed = preprocessor.transform(X_val)

    vectorizer = TfidfVectorizer(**vectorizer_config)
    X_train_vectorized = vectorizer.fit_transform(X_train_processed)
    X_val_vectorized = vectorizer.transform(X_val_processed)

    pipeline = Pipeline([
    ('classifier', GradientBoostingClassifier(random_state=42))
    ])

    pipeline.set_params(classifier = GradientBoostingClassifier(**gb_config))
    pipeline.fit(X_train_vectorized, y_train)
    y_pred = pipeline.predict(X_val_vectorized)
    f1 = f1_score(y_val, y_pred, average='weighted')

    mlflow.log_params(preprocessing_config)
    mlflow.log_params(vectorizer_config)
    mlflow.log_params(gb_config)
    mlflow.log_metric("f1_score", f1)

    return f1


def main():
    # Split data
    df = pd.read_csv('text.csv')
    df.drop(['Unnamed: 0', "company", "review"], axis=1, inplace=True)

    #Ensure correct data types
    df['text'] = df['text'].astype(str)
    df['rating'] = df['rating'].astype(int)

    X_temp, X_test, y_temp, y_test = train_test_split(df['text'], df['rating'], test_size=0.2, random_state=42, stratify = df['rating'])
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.2, random_state=42, stratify = y_temp)
    
    stop_words = set(stopwords.words('english'))

    with mlflow.start_run():
        # Run optimization
        study = optuna.create_study(direction='maximize')
        objective_with_data = lambda trial: objective(trial, X_train, y_train, X_val, y_val, stop_words)
        study.optimize(objective_with_data, n_trials = 50, n_jobs=-1)

        print("Best trial:")
        print("  Value: ", study.best_trial.value)
        print("  Params: ")
        for key, value in study.best_trial.params.items():
            print(f"    {key}: {value}")

        best_trial = study.best_trial
        mlflow.log_params(best_trial.params)
        mlflow.log_metric("best_f1_score", best_trial.value)

        preprocessor_best = TextPreprocessor(use_stem = best_trial.params['use_stem'],
                                             use_lem = best_trial.params['use_lem'],
                                             use_stop = best_trial.params['use_stop'],
                                             use_regex = best_trial.params['use_regex'],
                                             stop_words = stop_words)
        
        X_train_processed_best = preprocessor_best.fit_transform(X_train)
        X_val_processed_best = preprocessor_best.transform(X_val)

        vectorizer_best = TfidfVectorizer(ngram_range = (1, best_trial.params['ngram_max']),
                                          min_df = best_trial.params['min_df'],
                                          max_df = best_trial.params['max_df'],
                                          max_features = best_trial.params['max_features'])
        X_train_vectorized_best = vectorizer_best.fit_transform(X_train_processed_best)
        X_val_vectorized_best = vectorizer_best.transform(X_val_processed_best)

        best_pipeline = Pipeline([
            ('classifier', GradientBoostingClassifier(random_state=42,
                                                      **{k:v for k,v in best_trial.params.items() if k not in ['use_stem','use_lem', 'use_stop', 'use_regex', 'ngram_max', 'min_df', 'max_df', 'max_features']}))
            ])

        best_pipeline.fit(X_train_vectorized_best, y_train)
        mlflow.sklearn.log_model(best_pipeline, "best_model")

if __name__ == "__main__":
    main()