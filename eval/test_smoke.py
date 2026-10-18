"""

    test_smoke.py

    Quick end-to-end check of the evaluation harness on a slice of endings.jsonl.
    Does not need setup.sh and does not write to output/.
    Run with `python eval/test_smoke.py` (or pytest). Takes well under a minute.
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_eval
from model import Ember
from scorers import SCORERS, TRAINED_SCORERS


def test_harness_runs_end_to_end() -> None:
    """Every scorer produces valid metrics and every output file is written."""
    rows = run_eval.load_rows(run_eval.EVAL_DIR / "endings.jsonl")
    rows = rows[:12] + rows[-12:]  # 12 stop, 12 continue
    agent = Ember(
        "SmokeTest",
        pos=["Goodbye! Take care!", "See you soon!", "Bye for now!", "[BYE]"],
        neg=["Hello! Nice to see you!", "Welcome back!", "Hi! What's new?", "Sup"],
    )

    results = run_eval.evaluate(rows, agent, ks=[1, 3], folds=3, seed=0)
    assert len(results) == 2 * (len(SCORERS) + len(TRAINED_SCORERS))
    for r in results:
        assert 0 <= r["average_precision"] <= 1 and 0 <= r["best_f1"] <= 1
        assert r["latency_mean_ms"] > 0

    with tempfile.TemporaryDirectory() as out:
        run_eval.write_outputs(results, rows, agent, 3, Path(out))
        for name in ["results.md", "by_category.md", "results.json", "pr_k1.png", "pr_k3.png"]:
            assert (Path(out) / name).stat().st_size > 0, name


def test_cosine_is_embers_own_score() -> None:
    """The cosine scorer is exactly Ember.predict_proba()[0], so it measures StopButton as it is."""
    agent = Ember("SmokeTest", pos=["Goodbye!", "See you soon!"], neg=["Hello!", "Welcome back!"])
    text = "ok bye for now, actually one more thing"
    assert SCORERS["cosine"](agent)(text) == float(agent.predict_proba(text)[0])


def test_windows() -> None:
    """k = 1 is the raw last assistant reply; longer windows are role-prefixed lines."""
    row = run_eval.load_rows(run_eval.EVAL_DIR / "endings.jsonl")[0]
    assert run_eval.window(row, 1) == row["turns"][-1]["text"]
    assert run_eval.window(row, 2) == f"user: {row['turns'][1]['text']}\nassistant: {row['turns'][2]['text']}"


if __name__ == "__main__":
    for test in [test_windows, test_cosine_is_embers_own_score, test_harness_runs_end_to_end]:
        test()
        print(f"{test.__name__}: ok")
