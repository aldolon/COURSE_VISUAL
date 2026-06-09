import os
import json
import zipfile
import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

from sklearn.model_selection import train_test_split


# ==================================================
# LOAD DATASET 
# ==================================================
DATASET_DIR = "News_Datasets"

def load_dataset():

    news_data = []

    for file in os.listdir(DATASET_DIR):

        if not file.endswith(".zip"):
            continue

        zip_path = os.path.join(DATASET_DIR, file)

        print(f"Обрабатываю архив: {file}")

        with zipfile.ZipFile(zip_path, "r") as zip_ref:

            for json_file in zip_ref.namelist():

                if not json_file.endswith(".json"):
                    continue

                try:
                    with zip_ref.open(json_file) as f:
                        article = json.load(f)

                        news_data.append({
                            "title": article.get("title"),
                            "text": article.get("text"),
                            "categories": article.get("categories"),
                            "language": article.get("language"),
                            "published": article.get("published"),
                            "author": article.get("author")
                        })

                except Exception as e:
                    print(f"Ошибка в файле {json_file}: {e}")

    return pd.DataFrame(news_data)


# ==================================================
# PREPROCESSING
# ==================================================
def prepare_data(df):

    print("Preprocessing data...")

    df = df.dropna(subset=["text", "categories"])

    df["language"] = df["language"].astype(str).str.lower()
    df = df[df["language"].str.startswith("en", na=False)]

    df["category"] = df["categories"].apply(
        lambda x: x[0] if isinstance(x, list) and len(x) > 0 else None
    )

    df = df.dropna(subset=["category"])

    top_categories = df["category"].value_counts().nlargest(10).index
    df = df[df["category"].isin(top_categories)]

    df["text"] = df["text"].astype(str).str.lower()

    print("Final dataset size:", len(df))

    return df


# ==================================================
# TRAINING
# ==================================================
def train_model(X, y):

    print("\nVectorizing text...")

    vectorizer = TfidfVectorizer(
        max_features=50000,   
        stop_words="english",
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.9
    )

    X_vec = vectorizer.fit_transform(X)

    print("Train/test split...")

    X_train, X_test, y_train, y_test = train_test_split(
        X_vec,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    models = {
        "LinearSVC": LinearSVC(class_weight="balanced"),
        "LogisticRegression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced"
        ),
        "MultinomialNB": MultinomialNB()
    }

    results = []
    best_model = None
    best_name = None
    best_f1 = 0

    for name, model in models.items():

        print("\n==============================")
        print(f"Training {name}")

        model.fit(X_train, y_train)

        print(f"{name} finished")

        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        print(classification_report(y_test, y_pred))
        print(f"F1 = {f1:.4f}")

        results.append({
            "Model": name,
            "Accuracy": acc,
            "Precision": precision,
            "Recall": recall,
            "F1": f1
        })

        if f1 > best_f1:
            best_f1 = f1
            best_model = model
            best_name = name

    results_df = pd.DataFrame(results)

    results_df.to_csv("metrics.csv", index=False)

    joblib.dump(best_model, "model.pkl")
    joblib.dump(vectorizer, "vectorizer.pkl")

    print("\n==============================")
    print("BEST MODEL:", best_name)
    print("BEST F1:", best_f1)

    return best_model, vectorizer, results_df


# ==================================================
# MAIN
# ==================================================
if __name__ == "__main__":

    print("START PIPELINE")

    df = load_dataset()

    print("Raw size:", len(df))

    df = prepare_data(df)

    X = df["text"]
    y = df["category"]

    print("\nClass distribution:")
    print(y.value_counts())

    model, vectorizer, metrics = train_model(X, y)

    print("\nDONE")
