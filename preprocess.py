def clean_text(text):
    if not isinstance(text, str):
        return ""
    return text.lower()


def prepare_data(df):
    df = df.dropna(subset=["text", "categories"])
    df = df[df["categories"].notnull()]

    df["language"] = df["language"].astype(str).str.lower()
    df = df[df["language"].str.startswith("en", na=False)]

    df["category"] = df["categories"].apply(
        lambda x: x[0] if isinstance(x, list) and len(x) > 0 else None
    )

    df = df.dropna(subset=["category"])

    top_categories = df["category"].value_counts().nlargest(10).index
    df = df[df["category"].isin(top_categories)]

    df["text"] = df["text"].apply(clean_text)

    return df