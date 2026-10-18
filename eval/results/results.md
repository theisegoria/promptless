# StopButton evaluation results

Written by `python eval/run_eval.py`. Do not edit by hand; rerun the script instead.

- Data: eval/endings.jsonl, 200 rows (100 stop, 100 continue). Positive class: stop.
- Agent: StopButton, 21 target and 21 contrast phrases.
- Encoder: sentence-transformers/all-MiniLM-L6-v2 on mps:0.
- logistic_head_cv is scored out of fold with 5-fold StratifiedGroupKFold on this dataset. Every other scorer only sees the agent's phrases.
- Best F1 and its threshold are picked on the same rows they are reported on, so read them as an upper bound. F1 at default threshold uses 0.5, the point where predict() switches, with no tuning.
- Calling every row stop gives F1 0.667 at recall 1.000, and a random ranking gives average precision 0.500. A best F1 of 0.667 at recall 1.000 means no threshold did better than that.
- Latency is one call per row, including encoding the input.
- llama3.1 prompt chain: not evaluated. It is not part of this repo, and the StopButton prompt from the prompt-chain post is not published, so there is nothing to run.
- Python 3.12.14, torch 2.14.1, sentence-transformers 6.1.0, scikit-learn 1.9.1, macOS-27.0.1-arm64-arm-64bit.

| k | scorer | average precision | best F1 | at threshold | precision | recall | F1 at default threshold | mean ms | p95 ms |
|---|---|---|---|---|---|---|---|---|---|
| 1 | cosine | 0.452 | 0.690 | 0.4011 | 0.536 | 0.970 | 0.485 | 4.01 | 5.88 |
| 1 | dot | 0.452 | 0.690 | 0.4011 | 0.536 | 0.970 | 0.485 | 3.99 | 5.38 |
| 1 | l1 | 0.528 | 0.667 | 0.4793 | 0.500 | 1.000 | 0.481 | 3.77 | 5.09 |
| 1 | l2 | 0.522 | 0.667 | 0.4788 | 0.500 | 1.000 | 0.495 | 3.66 | 5.05 |
| 1 | pca_reconstruction | 0.749 | 0.774 | -0.002196 | 0.718 | 0.840 | - | 3.80 | 5.11 |
| 1 | logistic_head_corpus | 0.495 | 0.667 | 0.3539 | 0.500 | 1.000 | 0.549 | 3.95 | 5.49 |
| 1 | logistic_head_cv | 0.949 | 0.870 | 0.5224 | 0.903 | 0.840 | 0.864 | 4.33 | 6.23 |
| 2 | cosine | 0.435 | 0.676 | 0.4039 | 0.525 | 0.950 | 0.296 | 3.89 | 5.20 |
| 2 | dot | 0.435 | 0.676 | 0.4039 | 0.525 | 0.950 | 0.296 | 4.15 | 6.06 |
| 2 | l1 | 0.409 | 0.667 | 0.4824 | 0.500 | 1.000 | 0.304 | 4.02 | 5.52 |
| 2 | l2 | 0.409 | 0.667 | 0.482 | 0.500 | 1.000 | 0.296 | 4.05 | 6.01 |
| 2 | pca_reconstruction | 0.699 | 0.762 | -0.002549 | 0.646 | 0.930 | - | 4.06 | 5.44 |
| 2 | logistic_head_corpus | 0.407 | 0.667 | 0.3628 | 0.500 | 1.000 | 0.420 | 3.98 | 5.69 |
| 2 | logistic_head_cv | 0.958 | 0.897 | 0.4675 | 0.883 | 0.910 | 0.876 | 4.07 | 5.35 |
| 3 | cosine | 0.420 | 0.667 | -0.0513 | 0.500 | 1.000 | 0.273 | 4.32 | 6.25 |
| 3 | dot | 0.420 | 0.667 | -0.0513 | 0.500 | 1.000 | 0.273 | 4.17 | 5.95 |
| 3 | l1 | 0.410 | 0.667 | 0.4865 | 0.500 | 1.000 | 0.253 | 4.30 | 6.45 |
| 3 | l2 | 0.408 | 0.667 | 0.4873 | 0.500 | 1.000 | 0.282 | 3.93 | 5.82 |
| 3 | pca_reconstruction | 0.492 | 0.667 | -0.003313 | 0.500 | 1.000 | - | 3.62 | 4.83 |
| 3 | logistic_head_corpus | 0.411 | 0.667 | 0.405 | 0.500 | 1.000 | 0.519 | 3.75 | 5.40 |
| 3 | logistic_head_cv | 0.710 | 0.725 | 0.4116 | 0.582 | 0.960 | 0.682 | 3.58 | 4.87 |

Precision-recall curves: [k=1](pr_k1.png), [k=2](pr_k2.png), [k=3](pr_k3.png).
Accuracy by category: [by_category.md](by_category.md).
