"""

    run_eval.py

    Compares StopButton scorers on eval/endings.jsonl for windows of the last 1, 2 and 3
    turns, and writes the results table, accuracy by category, raw numbers and
    precision-recall curves to eval/results/.

    Run `bash setup.sh` first so that StopButton exists, then:
        python eval/run_eval.py
        python eval/run_eval.py --save-head StopButtonLogistic
"""

import argparse
import json
import platform
import sys
import time
from pathlib import Path
from typing import Callable

EVAL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EVAL_DIR.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sentence_transformers
import sklearn
import torch
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve
from sklearn.model_selection import StratifiedGroupKFold

from model import MODEL_CARD, Ember, LogisticHead
from scorers import DEFAULT_THRESHOLDS, SCORERS, TRAINED_SCORERS, Score

LLM_CHAIN_NOTE = (
    "llama3.1 prompt chain: not evaluated. It is not part of this repo, and the StopButton "
    "prompt from the prompt-chain post is not published, so there is nothing to run."
)

# Plot colours, in a fixed order checked for colour-blind separation.
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]
SURFACE, INK, INK_MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"


def load_rows(path: Path) -> list[dict]:
    """Read the labelled conversation tails from a JSON Lines file."""
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def window(row: dict, k: int) -> str:
    """
    Text that StopButton sees for the last k turns of a row.

    k = 1 is the last assistant reply on its own, which is how StopButton is called
    today. Longer windows put each turn on its own line, prefixed with its role.
    """
    turns = row["turns"][-k:]
    if k == 1:
        return turns[0]["text"]
    return "\n".join(f"{turn['role']}: {turn['text']}" for turn in turns)


def score_all(score: Score, texts: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Score each text with one call. Returns the scores and the latency of each call in milliseconds."""
    score(texts[0])  # warm up so the first timed call is not an outlier
    scores, latencies = [], []
    for text in texts:
        start = time.perf_counter()
        scores.append(score(text))
        latencies.append((time.perf_counter() - start) * 1000)
    return np.array(scores), np.array(latencies)


def cross_validate(build: Callable[[list[str], list[str]], Score], texts: list[str], y: np.ndarray, groups: list[str], folds: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Out-of-fold scores: each row is scored by a scorer built from the other folds only."""
    scores, latencies = np.zeros(len(texts)), np.zeros(len(texts))
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    for train, test in splitter.split(texts, y, groups):
        score = build(
            [texts[i] for i in train if y[i] == 1],
            [texts[i] for i in train if y[i] == 0],
        )
        scores[test], latencies[test] = score_all(score, [texts[i] for i in test])
    return scores, latencies


def summarise(rows: list[dict], y: np.ndarray, scores: np.ndarray, latencies: np.ndarray, default: float | None) -> dict:
    """Average precision, best F1 over all thresholds, F1 at the default threshold, latency and accuracy by category."""
    precision, recall, thresholds = precision_recall_curve(y, scores)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    best = int(np.argmax(f1[:-1]))  # the last curve point has no threshold
    predicted = scores >= thresholds[best]

    by_category = {}
    for category in dict.fromkeys(row["category"] for row in rows):
        idx = [i for i, row in enumerate(rows) if row["category"] == category]
        by_category[category] = float((predicted[idx] == y[idx]).mean())

    return {
        "average_precision": float(average_precision_score(y, scores)),
        "best_f1": float(f1[best]),
        "best_threshold": float(thresholds[best]),
        "precision_at_best": float(precision[best]),
        "recall_at_best": float(recall[best]),
        "f1_at_default": None if default is None else float(f1_score(y, scores >= default, zero_division=0)),
        "latency_mean_ms": float(latencies.mean()),
        "latency_p95_ms": float(np.percentile(latencies, 95)),
        "by_category": by_category,
        "curve": {"precision": precision.tolist(), "recall": recall.tolist()},
    }


def evaluate(rows: list[dict], agent: Ember, ks: list[int], folds: int, seed: int) -> list[dict]:
    """Run every scorer on every window size. Returns one result dict per (k, scorer)."""
    y = np.array([row["label"] == "stop" for row in rows], dtype=int)
    groups = [row["group"] for row in rows]
    results = []
    for k in ks:
        texts = [window(row, k) for row in rows]
        for text in texts:  # encode every input once untimed, so no scorer pays the encoder's warm-up
            agent.model.encode([text])
        for name, build in SCORERS.items():
            scores, latencies = score_all(build(agent), texts)
            results.append({"k": k, "scorer": name, **summarise(rows, y, scores, latencies, DEFAULT_THRESHOLDS.get(name))})
            print(f"k={k} {name}: done", flush=True)
        for name, build in TRAINED_SCORERS.items():
            scores, latencies = cross_validate(build, texts, y, groups, folds, seed)
            results.append({"k": k, "scorer": name, **summarise(rows, y, scores, latencies, DEFAULT_THRESHOLDS.get(name))})
            print(f"k={k} {name}: done", flush=True)
    return results


def results_table(results: list[dict]) -> str:
    """Markdown table of the main metrics, one row per (k, scorer)."""
    lines = [
        "| k | scorer | average precision | best F1 | at threshold | precision | recall | F1 at default threshold | mean ms | p95 ms |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        default = "-" if r["f1_at_default"] is None else f"{r['f1_at_default']:.3f}"
        lines.append(
            f"| {r['k']} | {r['scorer']} | {r['average_precision']:.3f} | {r['best_f1']:.3f} | {r['best_threshold']:.4g} "
            f"| {r['precision_at_best']:.3f} | {r['recall_at_best']:.3f} | {default} "
            f"| {r['latency_mean_ms']:.2f} | {r['latency_p95_ms']:.2f} |"
        )
    return "\n".join(lines)


def category_tables(results: list[dict], rows: list[dict]) -> str:
    """Markdown tables of accuracy by category at each scorer's best-F1 threshold, one table per k."""
    labels = {row["category"]: row["label"] for row in rows}
    counts = {category: sum(row["category"] == category for row in rows) for category in labels}
    sections = []
    for k in dict.fromkeys(r["k"] for r in results):
        at_k = [r for r in results if r["k"] == k]
        lines = [
            f"## Last {k} turn{'s' if k > 1 else ''}",
            "",
            "| category | label | rows | " + " | ".join(r["scorer"] for r in at_k) + " |",
            "|---|---|---|" + "---|" * len(at_k),
        ]
        for category, label in labels.items():
            cells = " | ".join(f"{r['by_category'][category]:.2f}" for r in at_k)
            lines.append(f"| {category} | {label} | {counts[category]} | {cells} |")
        sections.append("\n".join(lines))
    return "\n\n".join(sections)


def plot_curves(results: list[dict], k: int, path: Path) -> None:
    """Save the precision-recall curves of every scorer at one window size."""
    fig, ax = plt.subplots(figsize=(8.6, 4.8), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    ax.axhline(0.5, color=INK_MUTED, linewidth=1, label="random ranking (AP 0.500)")
    for color, r in zip(COLORS, results):
        ax.plot(
            r["curve"]["recall"], r["curve"]["precision"], drawstyle="steps-post", linewidth=1.6, color=color,
            linestyle="--" if r["scorer"] == "dot" else "-",
            label=f"{r['scorer']} (AP {r['average_precision']:.3f})",
        )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Recall", color=INK_MUTED)
    ax.set_ylabel("Precision", color=INK_MUTED)
    ax.set_title(f"Precision-recall for stop, last {k} turn{'s' if k > 1 else ''}", color=INK, loc="left", fontsize=11)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK_MUTED, labelsize=8)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=8, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def write_outputs(results: list[dict], rows: list[dict], agent: Ember, folds: int, out_dir: Path) -> str:
    """Write results.md, by_category.md, results.json and one PR curve PNG per k. Returns the results table."""
    out_dir.mkdir(parents=True, exist_ok=True)
    n_stop = sum(row["label"] == "stop" for row in rows)
    all_stop_f1 = 2 * n_stop / (len(rows) + n_stop)
    table = results_table(results)
    header = [
        "# StopButton evaluation results",
        "",
        "Written by `python eval/run_eval.py`. Do not edit by hand; rerun the script instead.",
        "",
        f"- Data: eval/endings.jsonl, {len(rows)} rows ({n_stop} stop, {len(rows) - n_stop} continue). Positive class: stop.",
        f"- Agent: {agent.name}, {len(agent.pos.text)} target and {len(agent.neg.text)} contrast phrases.",
        f"- Encoder: {MODEL_CARD} on {agent.model.device}.",
        f"- logistic_head_cv is scored out of fold with {folds}-fold StratifiedGroupKFold on this dataset. "
        "Every other scorer only sees the agent's phrases.",
        "- Best F1 and its threshold are picked on the same rows they are reported on, so read them as an upper bound. "
        "F1 at default threshold uses 0.5, the point where predict() switches, with no tuning.",
        f"- Calling every row stop gives F1 {all_stop_f1:.3f} at recall 1.000, and a random ranking gives average precision "
        f"{n_stop / len(rows):.3f}. A best F1 of {all_stop_f1:.3f} at recall 1.000 means no threshold did better than that.",
        "- Latency is one call per row, including encoding the input.",
        f"- {LLM_CHAIN_NOTE}",
        f"- Python {platform.python_version()}, torch {torch.__version__}, sentence-transformers {sentence_transformers.__version__}, "
        f"scikit-learn {sklearn.__version__}, {platform.platform()}.",
        "",
        table,
        "",
        "Precision-recall curves: " + ", ".join(f"[k={k}](pr_k{k}.png)" for k in dict.fromkeys(r["k"] for r in results)) + ".",
        "Accuracy by category: [by_category.md](by_category.md).",
        "",
    ]
    (out_dir / "results.md").write_text("\n".join(header), encoding="utf-8")
    (out_dir / "by_category.md").write_text(
        "# Accuracy by category\n\nShare of rows classified correctly at each scorer's best-F1 threshold. "
        "Written by `python eval/run_eval.py`.\n\n" + category_tables(results, rows) + "\n",
        encoding="utf-8",
    )
    raw = [{key: value for key, value in r.items() if key != "curve"} for r in results]
    (out_dir / "results.json").write_text(json.dumps(raw, indent=2) + "\n", encoding="utf-8")
    for k in dict.fromkeys(r["k"] for r in results):
        plot_curves([r for r in results if r["k"] == k], k, out_dir / f"pr_k{k}.png")
    return table


def save_head(rows: list[dict], name: str, k: int) -> None:
    """Fit a LogisticHead on every row at window size k and save it to output/ for the CLI."""
    pos = [window(row, k) for row in rows if row["label"] == "stop"]
    neg = [window(row, k) for row in rows if row["label"] == "continue"]
    head = LogisticHead(name, pos, neg)
    head.save()
    print(f'Try it with: python promptless load --name {name} --action predict_proba --inputs "Goodnight! Same time tomorrow?"')


def main() -> None:
    """Parse the command line, run the evaluation and write the results."""
    parser = argparse.ArgumentParser(description="Evaluate StopButton scorers on labelled conversation endings.")
    parser.add_argument("--data", type=Path, default=EVAL_DIR / "endings.jsonl")
    parser.add_argument("--agent", default="StopButton", help="Saved ember whose phrases the zero-shot scorers use")
    parser.add_argument("--k", type=int, nargs="+", default=[1, 2, 3], help="Window sizes, in turns")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=EVAL_DIR / "results")
    parser.add_argument("--save-head", metavar="NAME", help="Also fit a LogisticHead on all rows and save it to output/ under NAME")
    parser.add_argument("--save-k", type=int, default=1, help="Window size the saved head is trained on")
    args = parser.parse_args()

    rows = load_rows(args.data)
    try:
        agent = Ember.load(args.agent)
    except FileNotFoundError:
        raise SystemExit(f"No saved ember named {args.agent}. Run `bash setup.sh` first.")

    print(LLM_CHAIN_NOTE)
    results = evaluate(rows, agent, args.k, args.folds, args.seed)
    print(write_outputs(results, rows, agent, args.folds, args.out))
    print(f"Wrote {args.out}")

    if args.save_head:
        save_head(rows, args.save_head, args.save_k)


if __name__ == "__main__":
    main()
