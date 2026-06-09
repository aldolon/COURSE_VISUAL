import spacy
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer

# =========================
# LOAD MODELS ONCE
# =========================

nlp = spacy.load("en_core_web_sm")
summarizer = LexRankSummarizer()


# =========================
# SUMMARIZATION
# =========================

def summarize(text):

    if not isinstance(text, str):
        return []

    text = text[:3000]

    parser = PlaintextParser.from_string(
        text,
        Tokenizer("english")
    )

    return [
        str(s)
        for s in summarizer(parser.document, 3)
    ]


# =========================
# NER
# =========================

def extract_entities(text):

    if not isinstance(text, str):
        return {
            "PERSON": [],
            "ORG": [],
            "GPE": [],
            "DATE": []
        }

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

    for k in entities:
        entities[k] = list(set(entities[k]))

    return entities
