import os
import json
import zipfile
import pandas as pd

DATASET_DIR = "News_Datasets"

def load_dataset():
    news_data = []

    for file in os.listdir(DATASET_DIR):
        if not file.endswith(".zip"):
            continue

        zip_path = os.path.join(DATASET_DIR, file)

        print(f"Обрабатываю архив: {file}")

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            for json_file in zip_ref.namelist():

                if not json_file.endswith(".json"):
                    continue

                with zip_ref.open(json_file) as f:
                    try:
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

    df = pd.DataFrame(news_data)
    df = df.dropna(subset=["text", "title"])

    return df