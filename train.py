import pandas as pd
from sklearn.model_selection import train_test_split
import os
import joblib

from dataLoader import load_dataset
from preprocess import prepare_data
from model import train_model


# загрузка
df = load_dataset()

# подготовка
df = prepare_data(df)

X = df["text"]
y = df["category"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model, vectorizer = train_model(X_train, y_train)

print("BEFORE SAVE")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

joblib.dump(model, os.path.join(BASE_DIR, "model.pkl"))
joblib.dump(vectorizer, os.path.join(BASE_DIR, "vectorizer.pkl"))

print("AFTER SAVE")