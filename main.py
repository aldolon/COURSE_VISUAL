import os
import json
import zipfile
import pandas as pd
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer
from transformers import pipeline
import spacy
DATASET_DIR = "News_Datasets"  # папка с архивами

news_data = []

# 1. идём по всем файлам в папке
for file in os.listdir(DATASET_DIR):

    if not file.endswith(".zip"):
        continue

    zip_path = os.path.join(DATASET_DIR, file)

    print(f"Обрабатываю архив: {file}")

    # 2. открываем zip
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:

        # 3. список файлов внутри архива
        for json_file in zip_ref.namelist():

            if not json_file.endswith(".json"):
                continue

            # 4. читаем JSON внутри архива
            with zip_ref.open(json_file) as f:

                try:
                    article = json.load(f)

                    # 5. извлекаем нужные поля
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

# 6. создаём таблицу
df = pd.DataFrame(news_data)

# 7. очистка
df = df.dropna(subset=["text", "title"])

# 8. статистика

print("\n==============================")
print("АНАЛИЗ ДАТАСЕТА")
print("==============================")

print(f"Количество новостей: {len(df)}")
print(f"Количество столбцов: {len(df.columns)}")

print("\nСтруктура данных:")
print(df.columns.tolist())

print("\nПропуски:")
print(df.isnull().sum())

print("\nРаспределение языков:")
print(df["language"].value_counts())

print("\nПример записи:")
print(df.iloc[0])

# =========================
# ЭТАП 3: ПОДГОТОВКА ДАННЫХ
# =========================

df = df.dropna(subset=["text", "categories"])
df = df[df["categories"].notnull()]
# =========================
# ПРОВЕРКА ЯЗЫКОВ
# =========================

print("\nУникальные значения language:")
print(df["language"].unique())

print("\nКоличество записей до фильтрации:")
print(len(df))

# Если есть английский язык в любом виде:
if df["language"].notna().any():

    # приводим к строке и нижнему регистру
    df["language"] = df["language"].astype(str).str.lower()

    # оставляем всё, что начинается на en
    df_en = df[df["language"].str.startswith("en", na=False)]

    print("\nКоличество после фильтрации EN:")
    print(len(df_en))

    # если английских записей нет, работаем со всем датасетом
    if len(df_en) > 0:
        df = df_en
        print("Используем только английские новости")
    else:
        print("Английские новости не найдены, используем весь датасет")
else:
    print("Столбец language пустой, используем весь датасет")

# берём первую категорию
df["category"] = df["categories"].apply(
    lambda x: x[0] if isinstance(x, list) and len(x) > 0 else None
)

df = df.dropna(subset=["category"])

# оставляем топ-10 классов
top_categories = df["category"].value_counts().nlargest(10).index
df = df[df["category"].isin(top_categories)]

X = df["text"]
y = df["category"]

print("X size:", len(X))
print("y size:", len(y))
print("\nРаспределение классов:\n", y.value_counts())


# =========================
# ОЧИСТКА ТЕКСТА
# =========================
def clean_text(text):
    if not isinstance(text, str):
        return ""
    return text.lower()

X = X.apply(clean_text)


# =========================
# TF-IDF (УЛУЧШЕННЫЙ)
# =========================
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(
    max_features=100000,
    stop_words="english",
    ngram_range=(1, 2),
    min_df=3,
    max_df=0.9
)

X_vec = vectorizer.fit_transform(X)


# =========================
# TRAIN / TEST SPLIT
# =========================
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X_vec,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# =========================
# МОДЕЛЬ 
# =========================
from sklearn.svm import LinearSVC

model = LinearSVC(class_weight="balanced")

print("START TRAINING")
model.fit(X_train, y_train)
print("END TRAINING")


# =========================
# ОЦЕНКА
# =========================
from sklearn.metrics import classification_report

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))


# =========================
# ПРИМЕР ПРЕДСКАЗАНИЯ
# =========================
sample = X.iloc[0]
sample_vec = vectorizer.transform([sample])
pred = model.predict(sample_vec)

print("\nТекст:", sample[:200])
print("Предсказанная категория:", pred[0])







import nltk
nltk.download('punkt')

print("\n==============================")
print("СУММАРИЗАЦИЯ (TextRank)")
print("==============================")

text = sample

# ограничим длину (иначе шум)
text = text[:3000]

parser = PlaintextParser.from_string(text, Tokenizer("english"))

summarizer = LexRankSummarizer()

summary_sentences = summarizer(parser.document, 3)  # 3 предложения

print("\nКРАТКОЕ СОДЕРЖАНИЕ:\n")

for sentence in summary_sentences:
    print(str(sentence))





# =========================
# ЭТАП 5: NER (СУЩНОСТИ)
# =========================

print("\n==============================")
print("ИЗВЛЕЧЕНИЕ СУЩНОСТЕЙ (NER)")
print("==============================")

nlp = spacy.load("en_core_web_sm")

doc = nlp(sample)

for ent in doc.ents:
    if ent.label_ in ["PERSON", "ORG", "GPE", "DATE"]:
        print(f"{ent.text} → {ent.label_}")
    


def analyze_news(text):
    # =========================
    # 1. ПРЕДОБРАБОТКА
    # =========================
    if not isinstance(text, str):
        return None

    text_clean = text.lower()

    # =========================
    # 2. КЛАССИФИКАЦИЯ
    # =========================
    text_vec = vectorizer.transform([text_clean])
    category = model.predict(text_vec)[0]

    # =========================
    # 3. СУММАРИЗАЦИЯ (TextRank)
    # =========================
    from sumy.parsers.plaintext import PlaintextParser
    from sumy.nlp.tokenizers import Tokenizer
    from sumy.summarizers.text_rank import TextRankSummarizer

    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = TextRankSummarizer()

    summary_sentences = summarizer(parser.document, 3)
    summary = " ".join([str(s) for s in summary_sentences])

    # =========================
    # 4. NER (spaCy)
    # =========================
    doc = nlp(text)

    entities = {
        "PERSON": [],
        "ORG": [],
        "GPE": [],
        "DATE": []
    }

    for ent in doc.ents:
        if ent.label_ in entities:
            entities[ent.label_].append(ent.text)

    # убираем дубликаты
    for k in entities:
        entities[k] = list(set(entities[k]))

    # =========================
    # 5. РЕЗУЛЬТАТ
    # =========================
    return {
        "category": category,
        "summary": summary,
        "entities": entities
    }