"""Score every Harbor House Café review with one pinned model load."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from transformers import pipeline

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "raw" / "reviews.csv"
PROCESSED_PATH = ROOT / "data" / "processed" / "reviews_with_sentiment.csv"

MODEL_ID = "nlptown/bert-base-multilingual-uncased-sentiment"
MODEL_REVISION = "8f6f4e3a8f70be4b65d3a4a8762b6d781cda240d"


def sentiment_band(stars: int) -> str:
    if stars <= 2:
        return "negative"
    if stars == 3:
        return "neutral"
    return "positive"


def stars_from_label(label: str) -> int:
    return int(str(label).split()[0])


def load_classifier():
    """Load the pinned checkpoint once. Weights stay in the local cache."""
    return pipeline(
        "sentiment-analysis",
        model=MODEL_ID,
        revision=MODEL_REVISION,
        truncation=True,
        device=-1,
    )


def score_frame(frame: pd.DataFrame, classifier) -> pd.DataFrame:
    predictions = classifier(frame["review_text"].astype(str).tolist(), batch_size=32)
    scored = frame.copy()
    scored["rating"] = scored["rating"].astype(int)
    scored["predicted_stars"] = [stars_from_label(item["label"]) for item in predictions]
    scored["sentiment_band"] = scored["predicted_stars"].map(sentiment_band)
    scored["model_score"] = [round(float(item["score"]), 4) for item in predictions]
    return scored


def run() -> pd.DataFrame:
    reviews = pd.read_csv(RAW_PATH)
    classifier = load_classifier()
    scored = score_frame(reviews, classifier)
    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(PROCESSED_PATH, index=False)
    return scored


if __name__ == "__main__":
    result = run()
    print(f"wrote {len(result)} rows to {PROCESSED_PATH}")
