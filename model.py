from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

def train_model(X_train, y_train):

    vectorizer = TfidfVectorizer(
        max_features=100000,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.9
    )

    X_vec = vectorizer.fit_transform(X_train)

    model = LinearSVC(class_weight="balanced")
    model.fit(X_vec, y_train)

    return model, vectorizer