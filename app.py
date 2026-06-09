import streamlit as st
import pandas as pd
import os
import zipfile
import json
import joblib
import plotly.express as px
from collections import Counter

from nlp_pipeline import summarize, extract_entities
from preprocess import clean_text


# ==================================================
# CONFIG
# ==================================================

@st.cache_data
def get_sample(df, n=200, seed=42):

    if len(df) <= n:
        return df

    return df.sample(n, random_state=seed)

st.set_page_config(
    page_title="IAD News Dashboard",
    page_icon="🧠",
    layout="wide"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "News_Datasets")


# ==================================================
# MODELS
# ==================================================

model = joblib.load(os.path.join(BASE_DIR, "model.pkl"))
vectorizer = joblib.load(os.path.join(BASE_DIR, "vectorizer.pkl"))


# ==================================================
# DATA LOADER
# ==================================================

@st.cache_data
def load_data():

    data = []

    for file in os.listdir(DATASET_DIR):

        if not file.endswith(".zip"):
            continue

        zip_path = os.path.join(DATASET_DIR, file)

        with zipfile.ZipFile(zip_path, "r") as z:

            for json_file in z.namelist():

                if not json_file.endswith(".json"):
                    continue

                try:

                    with z.open(json_file) as f:

                        article = json.load(f)

                        lang = str(
                            article.get("language") or ""
                        ).lower()

                        if not lang.startswith("en"):
                            continue

                        text = article.get("text")
                        title = article.get("title")

                        if not text or not title:
                            continue

                        data.append({
                            "title": title,
                            "text": text,
                            "category":
                                (article.get("categories")
                                or ["unknown"])[0],
                            "language": lang,
                            "published":
                                article.get("published"),
                            "text_len": len(text)
                        })

                except:
                    continue

    df = pd.DataFrame(data)

    return df


df = load_data()

if df.empty:
    st.error("Dataset is empty")
    st.stop()


# ==================================================
# SIDEBAR
# ==================================================

st.sidebar.title("Filters")

categories = sorted(
    df["category"].dropna().unique().tolist()
)

selected_cat = st.sidebar.selectbox(
    "Category",
    ["All"] + categories
)

search = st.sidebar.text_input(
    "Search in article text"
)

df_filtered = df.copy()

if selected_cat != "All":
    df_filtered = df_filtered[
        df_filtered["category"] == selected_cat
    ]

if search:
    df_filtered = df_filtered[
        df_filtered["text"].str.contains(
            search,
            case=False,
            na=False
        )
    ]

st.sidebar.metric(
    "Articles",
    len(df_filtered)
)


# ==================================================
# TABS
# ==================================================

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Dashboard",
    "📰 News Feed",
    "🔍 Analysis",
    "📈 Model Evaluation"
])

# ==================================================
# DASHBOARD
# ==================================================

with tab1:

    st.title("📊 Analytics Dashboard")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "News Articles",
        len(df_filtered)
    )

    col2.metric(
        "Categories",
        df_filtered["category"].nunique()
    )

    col3.metric(
        "Avg Length",
        int(df_filtered["text_len"].mean())
    )

    st.markdown("---")

    st.subheader("Category Distribution")

    cat_counts = (
        df_filtered["category"]
        .value_counts()
        .reset_index()
    )

    cat_counts.columns = [
        "category",
        "count"
    ]

    fig1 = px.bar(
        cat_counts,
        x="category",
        y="count",
        title="Articles by Category"
    )

    st.plotly_chart(
        fig1,
        use_container_width=True
    )

    st.subheader(
        "Article Length Distribution"
    )

    fig2 = px.histogram(
        df_filtered,
        x="text_len",
        nbins=40,
        title="Text Length"
    )

    st.plotly_chart(
        fig2,
        use_container_width=True
    )

    st.subheader("Top Entities")

    entities_all = []
    
    sample_df = get_sample(df_filtered, n=200)

    entities_all = []

    for text in sample_df["text"].dropna():

        ents = extract_entities(text)

        for entity_type in ents:
            entities_all.extend(
                ents[entity_type]
            )

    if entities_all:

        top_entities = Counter(
            entities_all
        ).most_common(10)

        ent_df = pd.DataFrame(
            top_entities,
            columns=[
                "entity",
                "count"
            ]
        )

        fig3 = px.bar(
            ent_df,
            x="entity",
            y="count",
            title="Most Frequent Entities"
        )

        st.plotly_chart(
            fig3,
            use_container_width=True
        )

    st.markdown("---")

    st.subheader("System Insight")

    most_common_cat = (
        df_filtered["category"]
        .value_counts()
        .idxmax()
    )

    longest_news = df_filtered.loc[
        df_filtered["text_len"].idxmax()
    ]

    col1, col2 = st.columns(2)

    with col1:
        st.info(
            f"Most frequent category: "
            f"{most_common_cat}"
        )

    with col2:
        st.info(
            f"Longest article: "
            f"{longest_news['title']}"
        )


# ==================================================
# NEWS FEED
# ==================================================

with tab2:

    st.title("📰 News Feed")

    st.write(
        f"Showing {min(20, len(df_filtered))} "
        f"of {len(df_filtered)} articles"
    )

    for i, row in (
        df_filtered.head(20).iterrows()
    ):

        with st.container():

            st.markdown("---")

            st.markdown(
                f"### {row['title']}"
            )

            st.caption(
                f"Category: {row['category']} | "
                f"Language: {row['language']}"
            )

            st.write(
                row["text"][:500]
                + "..."
            )

            if st.button(
                "Analyze",
                key=f"analyze_{i}"
            ):

                st.session_state[
                    "selected_text"
                ] = row["text"]

                st.session_state[
                    "selected_title"
                ] = row["title"]


# ==================================================
# ANALYSIS
# ==================================================

with tab3:

    st.title("🔍 News Analysis")

    text = st.session_state.get(
        "selected_text"
    )

    if text:

        title = st.session_state.get(
            "selected_title",
            "Selected Article"
        )

        st.subheader(title)

        clean = clean_text(text)

        vec = vectorizer.transform(
            [clean]
        )

        category = model.predict(
            vec
        )[0]

        summary = summarize(text)

        entities = extract_entities(
            text
        )

        col1, col2 = st.columns(
            [1, 2]
        )

        with col1:

            st.metric(
                "Predicted Category",
                category
            )

            st.subheader(
                "Named Entities"
            )

            for k, v in entities.items():

                if v:

                    st.write(
                        f"**{k}**"
                    )

                    for item in v:
                        st.write(
                            f"• {item}"
                        )

                else:

                    st.write(
                        f"**{k}**: —"
                    )

        with col2:

            st.subheader(
                "Summary"
            )

            if summary:

                for i, s in enumerate(
                    summary,
                    start=1
                ):
                    st.info(
                        f"{i}. {s}"
                    )

            else:
                st.warning(
                    "Summary unavailable"
                )

    else:

        st.info(
            "Select an article in the News Feed tab."
        )


    
    from sklearn.metrics import classification_report, accuracy_score


with tab4:

    st.title("📈 Model Evaluation")

    metrics_path = os.path.join(BASE_DIR, "metrics.csv")

    if os.path.exists(metrics_path):

        metrics = pd.read_csv(metrics_path)

        st.subheader("Model Comparison Table")

        st.dataframe(metrics)

        st.subheader("F1-score Comparison")

        fig = px.bar(
            metrics,
            x="Model",
            y="F1",
            title="F1-score by Model",
            text="F1"
        )

        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Accuracy Comparison")

        fig2 = px.bar(
            metrics,
            x="Model",
            y="Accuracy",
            title="Accuracy by Model",
            text="Accuracy"
        )

        st.plotly_chart(fig2, use_container_width=True)

        best_model = metrics.loc[
            metrics["F1"].idxmax()
        ]

        st.success(
            f"Best Model: {best_model['Model']} "
            f"(F1 = {best_model['F1']:.4f})"
        )

    else:

        st.warning(
            "metrics.csv not found. Run train_model.py first."
        )

