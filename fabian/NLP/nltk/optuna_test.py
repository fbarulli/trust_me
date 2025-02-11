import mlflow
import mlflow.sklearn
import mlflow.optuna

mlflow.set_tracking_uri("file:///your/local/mlruns")  # Set to your desired tracking URI
mlflow.autolog()  # Automatically logs parameters, metrics, and models

def main(df: pd.DataFrame) -> None:
    check_input_df(df)

    df['text'] = df['text'].astype(str)
    df['text'] = df['text'].apply(remove_emojis)

    X = df['text']
    y = df['rating']
    stop_words = set(stopwords.words('english'))

    study = optuna.create_study(direction='maximize')
    
    # Start an MLflow run
    with mlflow.start_run(run_name="Optuna Study"):
        study.optimize(lambda t: objective(t, X, y, stop_words), n_trials=20)

        best_params: Dict[str, Any] = study.best_trial.params
        mlflow.log_params(best_params)  # Log best parameters manually
        print("Best F1:", study.best_value)
        print("Best params:", best_params)

        best_preprocessor = TextPreprocessor(
            use_stem=best_params['use_stem'],
            use_lem=best_params['use_lem'],
            use_stop=best_params['use_stop'],
            use_regex=best_params['use_regex'],
            stop_words=stop_words
        )

        if best_params['vectorizer_type'] == 'tfidf':
            best_vectorizer = TfidfVectorizer(max_features=best_params['max_features'])
        else:
            best_vectorizer = CountVectorizer(max_features=best_params['max_features'])

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                            stratify=y, random_state=42)
        X_train_prep = best_preprocessor.fit_transform(X_train)
        X_test_prep = best_preprocessor.transform(X_test)
        X_train_vec = best_vectorizer.fit_transform(X_train_prep)
        X_test_vec = best_vectorizer.transform(X_test_prep)
        shape_test(X_train_vec, X_test_vec)

        best_model = GradientBoostingClassifier(
            n_estimators=best_params['n_estimators'],
            learning_rate=best_params['learning_rate'],
            max_depth=best_params['max_depth'],
            subsample=best_params['subsample'],
            random_state=42
        )
        best_model.fit(X_train_vec, y_train)
        y_pred = best_model.predict(X_test_vec)

        acc = accuracy_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred, average='macro')
        f1 = f1_score(y_test, y_pred, average='macro')

        logging.info(f"Accuracy: {acc}, Recall: {rec}, F1: {f1}")

        # Log final metrics
        mlflow.log_metrics({
            "accuracy": acc,
            "recall": rec,
            "f1_score": f1
        })

        # Save the model manually
        mlflow.sklearn.log_model(best_model, "best_model")

        plot_confusion_matrix_and_report(y_test, y_pred)
