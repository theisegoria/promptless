"""

    model.py
"""

import os
import joblib
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from collections import namedtuple
from dataclasses import dataclass, field
from sentence_transformers import SentenceTransformer

DIR_PATH = Path(__file__).parent / "output"
if not DIR_PATH.exists():
    print(f'Output directory does not exist. Creating new:\n{DIR_PATH}')
    DIR_PATH.mkdir(parents=True, exist_ok=True)

MODEL_CARD = "sentence-transformers/all-MiniLM-L6-v2"
State = namedtuple("State", ["text", "x"])

@dataclass(slots=True)
class Observation:
    states: list = field(default_factory=list)

    def add(self, state, target_score, contrast_score, cat: Literal["pos", "neg"]):
        self.states.append(
            {
                "state": state,
                "target": target_score,
                "contrast": contrast_score,
                "cat": cat,
            }
        )

    def predict(self, model, emb):
        states = pd.DataFrame(self.states)
        pred = {}
        max_vote = None
        for cat, row in states.groupby("cat"):
            if not row.empty:
                cat_state = np.vstack([i.x for i in row["state"].values])
                assert len(cat_state.shape) == 2 and cat_state.shape[-1] == 384

                N = cat_state.shape[0]
                pred[cat] = model.similarity_pairwise(
                    np.repeat(emb, N, axis=0), cat_state
                ).cpu().numpy()
                assert pred[cat].shape[0] == N and len(pred[cat].shape) == 1

                if max_vote is None or pred[cat].mean().item() > max_vote[-1]:
                    max_vote = (cat, pred[cat].mean().item())
            else:
                pred[cat] = None

        pred['vote'] = max_vote
        return pred

class Ember:
    model = None

    def __new__(cls, *args, **kwargs):
        """Ensures only one sentence model is initialized in the workspace."""
        if cls.model is None:
            cls.model = SentenceTransformer(MODEL_CARD)
            print("Initializing new sentence encoder!")
        else:
            print("No new encoder initiated as one already exists.")
        return super().__new__(cls)

    def __init__(self, name: str, pos: list, neg: list):
        self.name = name
        self.pos = State(text=pos, x=self.model.encode(pos))
        self.neg = State(text=neg, x=self.model.encode(neg))
        self.observation = Observation()

    @property
    def filename(self):
        """Path filename where the instance will be saved."""
        return DIR_PATH / f"{self.name.lower()}.joblib"

    def _predict(self, inputs: str, add_observe=False):
        """
        Generate similarity scores for the given input against positive and negative states.

        Args:
            inputs (str): The input text to be analyzed.

        Yields:
            float: Mean similarity score with positive and negative inputs.
        """
        text_input = self.model.encode([inputs])
        pos_simi = self.model.similarity_pairwise(text_input, self.pos.x)
        neg_simi = self.model.similarity_pairwise(text_input, self.neg.x)

        yield pos_simi.mean()
        yield neg_simi.mean()

        if add_observe is True:
            state = State(text=inputs, x=text_input)
            voted = "pos" if pos_simi.mean() > neg_simi.mean() else "neg"
            self.observation.add(state, pos_simi.mean().item(), neg_simi.mean().item(), voted)

    def predict_proba(self, inputs: str, add_observe=False):
        """
        Predict the probabilities of the input being positive or negative.

        Args:
            inputs (str): The input text to be analyzed.

        Returns:
            np.ndarray: An array of probabilities for positive and negative classes.
        """
        mus = list(self._predict(inputs, add_observe))
        prob = np.array(mus)
        prob /= prob.sum()
        return prob

    def predict(self, inputs: str, add_observe=False):
        """
        Predict the sentiment of the input text.

        Args:
            inputs (str): The input text to be analyzed.

        Returns:
            bool: True if the input is predicted to belong to Target. False otherwise.
        """
        output = self.predict_proba(inputs, add_observe)
        return output.argmax() == 0

    def observe(self, inputs: str):
        return self.observation.predict(self.model, self.model.encode([inputs]))

    def save(self):
        # note that when you set this to None it will set the instance's saved model to none and not the class itself. Although this seems like a shady way of programming. Since joblib only serializes the attributes shown in __dict__, setting model to None is pointless.
        # self.model = None

        joblib.dump(self, self.filename)
        print(f"Saved {self.name} to {self.filename}!")

    @staticmethod
    def load(name):
        """Loads any instance."""
        file_name = DIR_PATH / f"{name.lower()}.joblib"
        if file_name.exists():
            return joblib.load(file_name)
        else:
            raise FileNotFoundError(f"No ember named {name}. File name: {file_name}")


def add_new_ember(name, pos, neg):
    """Adds new ember to collection. Returns None if already exists. """

    try:
        file_name = DIR_PATH / f"{name.lower()}.joblib"
        if file_name.exists():
            raise FileExistsError(
                f"Cannot setup {name} as this already exists. File name: {file_name}"
            )

        obj = Ember(name, pos, neg)
        obj.save()

    except Exception as e:
        return e

def remove_ember(name):
    """Removes ember stored in collection."""
    file_name = DIR_PATH / f"{name.lower()}.joblib"
    if not file_name.exists():
        raise FileNotFoundError(f"No ember named {name}. File name: {file_name}")
    else:
        os.remove(file_name)
        print(f"Deleted ember {name}")

if __name__ == "__main__":
    test = {
        'pos': ["[BYE]", "goodbye", "see you", "cya!"],
        'neg': ["good morning", "hello", "HEY", "how are you!"]
    }
    eep = Ember(name="StopButton", pos=test['pos'], neg=test['neg'])
    for i in ["bye lol", "goodbye", "heyyyyyyy", "whats up!", "do you mind?", "what about tomorrow?"]:
        eep.predict(i, add_observe=True)

    example = "i miss you"
    print(f'Predicting {example} in instance 1')
    init_pred = eep.predict_proba(example) # [pos, neg]
    history_pred = eep.observe(example)

    print(
    f"""
        Initial samples distance: {init_pred}
        Observed samples: {history_pred}
    """)