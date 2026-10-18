# Evaluating StopButton

This folder measures how well StopButton separates conversations that are over from conversations that only look over.

### Dataset
`endings.jsonl` has 200 conversation tails. Each row is the last three turns (assistant, user, assistant) and a label:
- `stop`: the conversation is complete and could be saved as finished.
- `continue`: it is not, even if the last message sounds like it.

```json
{"id": "c008", "label": "continue", "category": "bye_then_more", "lang": "en", "group": "c008",
 "turns": [{"role": "assistant", "text": "Glad I could help. Bye!"},
           {"role": "user", "text": "sorry, actually can you also check the second paragraph"},
           {"role": "assistant", "text": "No problem. Paste the second paragraph and I'll take a look."}]}
```

The set is synthetic. The rows were written and labelled for this evaluation, not taken from real chats. Half are endings and half are deliberately hard non-endings: a goodbye followed by one more question, thanks in the middle of a task, goodnight in a thread that picks up the next day, farewells quoted inside an answer, jokes like "bye bye, bug!", brb, and `[END]` or `[BYE]` mentioned while talking about code. There are typos, slang and a few rows in other languages. Because the hard cases are over-represented, the numbers are a stress test, not an estimate of how often StopButton is wrong in normal use.

Real logs would make this much more useful, so replace or extend the file as you like. If several tails come from the same conversation, give them the same `group` so cross-validation keeps them in the same fold.

### Running it
```shell
pip install -r eval/requirements.txt
bash setup.sh                  # creates StopButton if it does not exist yet
python eval/run_eval.py
python eval/test_smoke.py      # about 20 seconds, needs no saved embers
```

Results are written to `eval/results/`:
- `results.md`: the main table
- `by_category.md`: accuracy for each category of row
- `results.json`: the same numbers, unrounded
- `pr_k1.png`, `pr_k2.png`, `pr_k3.png`: precision-recall curves

`--k 1 2 3` sets the window sizes. With `k = 1` the scorer sees only the last assistant reply, which is how StopButton is called now. With `k = 2` or `3` it sees the last turns joined into one string, one turn per line, each starting with `user:` or `assistant:`. `--agent` picks a different saved ember, and `--data`, `--folds` and `--out` do what they say.

### What is compared
- `cosine`: StopButton as it is, `Ember.predict_proba(text)[0]`.
- `dot`, `l1`, `l2`: the same target-versus-contrast comparison with the other measures listed in the Stop Button post. all-MiniLM-L6-v2 returns unit-length embeddings, so `dot` gives exactly the same numbers as `cosine`.
- `pca_reconstruction`: PCA fitted on the target phrases (95% of the variance), scored by minus the reconstruction MSE.
- `logistic_head_corpus`: `LogisticHead` fitted on StopButton's own target and contrast phrases, the same data `cosine` uses.
- `logistic_head_cv`: `LogisticHead` fitted on this dataset and scored out of fold with 5-fold `StratifiedGroupKFold`.

The llama3.1 prompt chain is not included. It is not part of this repo and its StopButton prompt is not published, so the script prints a note and skips it.

### Reading the numbers
- Half the rows are `stop`, so a random ranking has average precision 0.5, and calling everything `stop` gives F1 0.667 at recall 1.0. A best F1 of 0.667 at recall 1.0 means the scorer found no threshold better than that.
- Best F1 and its threshold are chosen on the same rows they are reported on, so treat them as an upper bound. The "F1 at default threshold" column uses 0.5, where `predict()` switches, with no tuning.
- `logistic_head_cv` is trained on rows from the same synthetic set it is tested on. It can learn how this set was written rather than what an ending is, so expect it to do worse on real logs.
- The cosine threshold of -0.0513 at `k = 3` is not a typo. `predict_proba` divides by the sum of the two mean similarities, and for inputs that are unlike both corpora that sum can be close to zero or negative.
- Latency is measured on one machine, one call at a time, and moves by about a millisecond between runs.

### Using LogisticHead from the CLI
`LogisticHead` is an `Ember`, so it is saved to `output/` and loaded the same way:

```shell
python eval/run_eval.py --save-head StopButtonLogistic
python promptless load --name StopButtonLogistic --action predict_proba --inputs "Goodnight! Same time tomorrow?"
```

or from Python, with your own phrases:

```python
from promptless import LogisticHead
LogisticHead("MyHead", pos=["ok bye!", "thanks, that's all"], neg=["thanks! what's next?", "brb"]).save()
```

### Adding a scorer
In `scorers.py`, write a function that takes the loaded agent and returns a function from text to a float, higher meaning more likely an ending, then add it to `SCORERS`:

```python
def max_cosine(agent: Ember) -> Score:
    """Highest cosine similarity to any single target phrase."""
    def score(text: str) -> float:
        x = agent.model.encode([text])
        return float((x @ agent.pos.x.T).max())
    return score

SCORERS["max_cosine"] = max_cosine
```

If the scorer has a fixed cut-off like `predict()` does, add it to `DEFAULT_THRESHOLDS`. A scorer that needs training data goes in `TRAINED_SCORERS` instead: it receives the `stop` and `continue` texts of the training folds and is scored on the held-out fold.
