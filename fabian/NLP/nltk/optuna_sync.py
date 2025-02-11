import os
import re
import emoji
import mlflow
from mlflow import MlflowClient
import optuna
import logging
import pickle
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, Any
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    recall_score,
    f1_score,
    classification_report
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sns.despine()
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')


def remove_emojis(text: str) -> str:
    emoji_set = set(e['emoji'] for e in emoji.emoji_list(text))
    return ''.join(c for c in text if c not in emoji_set)


class TextPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, use_stem=False, use_lem=False, use_stop=False, use_regex=False, stop_words=None):
        self.use_stem = use_stem
        self.use_lem = use_lem
        self.use_stop = use_stop
        self.use_regex = use_regex
        self.stop_words = stop_words

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Perform transformations here (regex, removing emojis, etc.)
        return X.apply(remove_emojis)


def train_model(X_train, y_train, params: Dict[str, Any]) -> GradientBoostingClassifier:
    model = GradientBoostingClassifier(random_state=42, **params)
    model.fit(X_train, y_train)
    return model


def objective(trial: optuna.Trial, X_train: pd.Series, y_train: pd.Series) -> float:
    # Example of searching over just a couple params
    n_estimators = trial.suggest_int('n_estimators', 50, 300)
    learning_rate = trial.suggest_float('learning_rate', 0.01, 0.2, log=True)

    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_vec = vectorizer.fit_transform(X_train)

    params = {
        "n_estimators": n_estimators,
        "learning_rate": learning_rate
    }
    model = train_model(X_train_vec, y_train, params)
    # Evaluate
    y_pred = model.predict(X_train_vec)
    return f1_score(y_train, y_pred, average='macro')


def plot_and_save_confusion_matrix(y_true, y_pred, filename="cm.png"):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.tight_layout()
    sns.despine()
    plt.savefig(filename)
    plt.close()


def main():
    # ------------------------------------------------
    # 1) SET UP MLflow EXPERIMENT & RUN (via MlflowClient)
    # ------------------------------------------------
    client = MlflowClient()

    experiment_name = "my_manual_mlflow_experiment"
    run_name = "my_manual_optuna_run"
    artifact_subdir = "my_artifacts"

    # Create or fetch experiment ID
    try:
        experiment_id = client.create_experiment(experiment_name)
    except mlflow.exceptions.MlflowException:
        experiment = client.get_experiment_by_name(experiment_name)
        experiment_id = experiment.experiment_id

    # ------------------------------------------------
    # 2) LOAD & PREPARE DATA
    # ------------------------------------------------
    df = pd.read_csv("text.csv")  # must have 'text' and 'rating' columns
    df['text'] = df['text'].astype(str)
    df['rating'] = df['rating'].astype(int)

    X = df['text']
    y = df['rating']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # ------------------------------------------------
    # 3) OPTUNA STUDY
    # ------------------------------------------------
    study = optuna.create_study(direction="maximize")

    def obj_wrapper(trial):
        return objective(trial, X_train, y_train)

    study.optimize(obj_wrapper, n_trials=10)

    best_params = study.best_params
    logging.info(f"Best trial F1: {study.best_value}")
    logging.info(f"Best params: {best_params}")

    # ------------------------------------------------
    # 4) FINAL TRAINING AND LOGGING (MANUAL MLflow RUN)
    # ------------------------------------------------
    with mlflow.start_run(experiment_id=experiment_id, run_name=run_name) as run:

        mlflow.log_params(best_params)

        vectorizer = TfidfVectorizer(max_features=5000)
        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)

        best_model = train_model(X_train_vec, y_train, best_params)
        y_pred = best_model.predict(X_test_vec)

        acc = accuracy_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred, average='macro')
        f1v = f1_score(y_test, y_pred, average='macro')
        mlflow.log_metric("test_accuracy", acc)
        mlflow.log_metric("test_recall", rec)
        mlflow.log_metric("test_f1", f1v)

        # Save confusion matrix plot
        cm_name = "confusion_matrix.png"
        plot_and_save_confusion_matrix(y_test, y_pred, cm_name)
        mlflow.log_artifact(cm_name, artifact_path=artifact_subdir)

        # Save the model artifacts
        os.makedirs("models", exist_ok=True)
        with open("models/final_pipeline.pkl", "wb") as f:
            pickle.dump({"vectorizer": vectorizer, "model": best_model}, f)
        mlflow.log_artifact("models/final_pipeline.pkl", artifact_path=artifact_subdir)

        logging.info(f"Run {run.info.run_id} completed. "
                     f"Metrics logged: accuracy={acc}, recall={rec}, f1={f1v}")


if __name__ == "__main__":
    main()
