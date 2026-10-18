"""

    scorers.py

    Scorers compared by run_eval.py. Every scorer maps one input string to a float
    where higher means "more like the target corpus", i.e. more likely to be an ending.

    SCORERS are built from the loaded agent (e.g. StopButton) and never see the
    evaluation data. TRAINED_SCORERS are built from the target and contrast texts of
    one cross-validation fold.
"""

import contextlib
import io
import sys
from pathlib import Path
from typing import Callable

import numpy as np
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model import Ember, LogisticHead

Score = Callable[[str], float]


def cosine(agent: Ember) -> Score:
    """Ember's own score: predict_proba()[0], the target share of the mean cosine similarities."""
    return lambda text: float(agent.predict_proba(text)[0])


def dot(agent: Ember) -> Score:
    """Same scheme as cosine with raw dot products. Identical to cosine when the encoder normalises its embeddings."""
    def score(text: str) -> float:
        x = agent.model.encode([text])
        pos, neg = (x @ agent.pos.x.T).mean(), (x @ agent.neg.x.T).mean()
        return float(pos / (pos + neg))
    return score


def _distance(agent: Ember, order: int) -> Score:
    """Contrast share of the mean distances, so that being closer to the target gives a higher score."""
    def score(text: str) -> float:
        x = agent.model.encode([text])
        pos = np.linalg.norm(agent.pos.x - x, ord=order, axis=1).mean()
        neg = np.linalg.norm(agent.neg.x - x, ord=order, axis=1).mean()
        return float(neg / (pos + neg))
    return score


def l1(agent: Ember) -> Score:
    """Mean Manhattan distance to each corpus."""
    return _distance(agent, 1)


def l2(agent: Ember) -> Score:
    """Mean Euclidean distance to each corpus."""
    return _distance(agent, 2)


def pca_reconstruction(agent: Ember, variance: float = 0.95) -> Score:
    """Negative reconstruction MSE under a PCA fitted on the target corpus only, as in the Stop Button post."""
    pca = PCA(n_components=variance).fit(agent.pos.x)

    def score(text: str) -> float:
        x = agent.model.encode([text])
        return float(-((pca.inverse_transform(pca.transform(x)) - x) ** 2).mean())
    return score


def _logistic(pos: list[str], neg: list[str]) -> Score:
    """Fit a LogisticHead on the given texts and score with its target probability."""
    with contextlib.redirect_stdout(io.StringIO()):
        head = LogisticHead("eval", pos, neg)
    return lambda text: float(head.predict_proba(text)[0])


def logistic_head_corpus(agent: Ember) -> Score:
    """LogisticHead fitted on the agent's own target and contrast phrases, the same data cosine uses."""
    return _logistic(list(agent.pos.text), list(agent.neg.text))


def logistic_head_cv(pos: list[str], neg: list[str]) -> Score:
    """LogisticHead fitted on the stop and continue rows of the training folds."""
    return _logistic(pos, neg)


SCORERS: dict[str, Callable[[Ember], Score]] = {
    "cosine": cosine,
    "dot": dot,
    "l1": l1,
    "l2": l2,
    "pca_reconstruction": pca_reconstruction,
    "logistic_head_corpus": logistic_head_corpus,
}

TRAINED_SCORERS: dict[str, Callable[[list[str], list[str]], Score]] = {
    "logistic_head_cv": logistic_head_cv,
}

# Scores at or above these count as "stop" in the same way Ember.predict() does:
# 0.5 means the input is at least as close to the target as to the contrast.
# PCA reconstruction error has no natural cut-off, so it has none.
DEFAULT_THRESHOLDS: dict[str, float] = {
    "cosine": 0.5,
    "dot": 0.5,
    "l1": 0.5,
    "l2": 0.5,
    "logistic_head_corpus": 0.5,
    "logistic_head_cv": 0.5,
}
