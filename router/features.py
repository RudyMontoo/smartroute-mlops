"""Feature extraction for the query-difficulty router.

Single source of truth. Training (`train.py`, `sweep.py`), serving
(`gateway/router_service.py`) and drift detection (`drift.py`) all import from here,
so there is no way for the features at serve time to drift away from the features the
model was fitted on. Train/serve skew is the classic silent MLOps failure, and the
cheapest defence is to make it structurally impossible rather than rely on discipline.
"""

from __future__ import annotations

import re

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler

TEXT_COLUMN = "query"

#: Hand-engineered signals, in a fixed order. `drift.py` monitors exactly these.
NUMERIC_FEATURES = [
    "char_len",
    "word_count",
    "question_marks",
    "digit_ratio",
    "avg_word_len",
    "has_code_fence",
    "kw_reasoning",
    "kw_code",
    "kw_explain",
]

# Multi-step reasoning markers.
_RE_REASONING = re.compile(
    r"\b(step by step|prove|derive|calculate|compute|how many|how much|total|"
    r"therefore|if .* then|solve)\b",
    re.IGNORECASE,
)
# Code-generation markers.
_RE_CODE = re.compile(
    r"\b(write a (python |)function|implement|algorithm|code|program|script|"
    r"return a list|def )\b",
    re.IGNORECASE,
)
# Open-ended synthesis markers.
_RE_EXPLAIN = re.compile(
    r"\b(explain|describe|why|what are some|give me|suggest|list some|"
    r"tell me about|summarize)\b",
    re.IGNORECASE,
)
_RE_CODE_FENCE = re.compile(r"```|\bdef \w+\(|\bclass \w+:")


def numeric_features(query: str) -> dict[str, float]:
    """Compute the hand-engineered features for one query.

    Kept free of any fitted state so it can be called on a single live request just
    as cheaply as on the training set.
    """
    text = query or ""
    words = text.split()
    n_chars = len(text)
    n_words = len(words)

    return {
        "char_len": float(n_chars),
        "word_count": float(n_words),
        "question_marks": float(text.count("?")),
        "digit_ratio": (sum(c.isdigit() for c in text) / n_chars) if n_chars else 0.0,
        "avg_word_len": (sum(len(w) for w in words) / n_words) if n_words else 0.0,
        "has_code_fence": float(bool(_RE_CODE_FENCE.search(text))),
        "kw_reasoning": float(bool(_RE_REASONING.search(text))),
        "kw_code": float(bool(_RE_CODE.search(text))),
        "kw_explain": float(bool(_RE_EXPLAIN.search(text))),
    }


def build_frame(queries: list[str]) -> pd.DataFrame:
    """Turn raw query strings into the DataFrame the sklearn pipeline expects."""
    rows = [{TEXT_COLUMN: q, **numeric_features(q)} for q in queries]
    return pd.DataFrame(rows, columns=[TEXT_COLUMN, *NUMERIC_FEATURES])


def build_vectorizer(use_numeric: bool = True, max_features: int = 5000) -> ColumnTransformer:
    """Assemble the feature union.

    `use_numeric=False` gives the TF-IDF-only arm of the ablation in `sweep.py`, which
    is how we find out whether the hand-engineered features are earning their keep.

    MinMaxScaler rather than StandardScaler is deliberate: MultinomialNB is one of the
    candidates in the sweep and it rejects negative inputs.
    """
    text_step = (
        "tfidf",
        TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=max_features,
            min_df=2,
            sublinear_tf=True,
            strip_accents="unicode",
        ),
        TEXT_COLUMN,
    )

    transformers = [text_step]
    if use_numeric:
        transformers.append(("numeric", MinMaxScaler(), NUMERIC_FEATURES))

    return ColumnTransformer(transformers=transformers, sparse_threshold=0.3)
