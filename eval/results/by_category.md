# Accuracy by category

Share of rows classified correctly at each scorer's best-F1 threshold. Written by `python eval/run_eval.py`.

## Last 1 turn

| category | label | rows | cosine | dot | l1 | l2 | pca_reconstruction | logistic_head_corpus | logistic_head_cv |
|---|---|---|---|---|---|---|---|---|---|
| farewell_after_task | stop | 20 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.95 |
| formal_close | stop | 12 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| done_no_farewell | stop | 15 | 0.93 | 0.93 | 1.00 | 1.00 | 0.80 | 1.00 | 0.80 |
| slang_typos | stop | 12 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.92 |
| non_english | stop | 12 | 0.92 | 0.92 | 1.00 | 1.00 | 0.08 | 1.00 | 0.42 |
| goodnight_single_session | stop | 8 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.88 |
| user_ends_abruptly | stop | 8 | 1.00 | 1.00 | 1.00 | 1.00 | 0.88 | 1.00 | 0.62 |
| thanks_done | stop | 8 | 0.88 | 0.88 | 1.00 | 1.00 | 0.88 | 1.00 | 1.00 |
| long_turns | stop | 5 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| bye_then_more | continue | 15 | 0.47 | 0.47 | 0.00 | 0.00 | 1.00 | 0.00 | 1.00 |
| thanks_not_ending | continue | 15 | 0.33 | 0.33 | 0.00 | 0.00 | 0.93 | 0.00 | 0.93 |
| goodnight_multi_day | continue | 12 | 0.00 | 0.00 | 0.00 | 0.00 | 0.08 | 0.00 | 0.75 |
| farewell_in_content | continue | 14 | 0.00 | 0.00 | 0.00 | 0.00 | 0.43 | 0.00 | 0.86 |
| farewell_joke | continue | 8 | 0.00 | 0.00 | 0.00 | 0.00 | 0.88 | 0.00 | 1.00 |
| brb_pause | continue | 10 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.80 |
| quoted_end_token | continue | 6 | 0.00 | 0.00 | 0.00 | 0.00 | 0.67 | 0.00 | 1.00 |
| non_english_bye_then_more | continue | 10 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 0.00 | 0.90 |
| part_done | continue | 10 | 0.40 | 0.40 | 0.00 | 0.00 | 1.00 | 0.00 | 1.00 |

## Last 2 turns

| category | label | rows | cosine | dot | l1 | l2 | pca_reconstruction | logistic_head_corpus | logistic_head_cv |
|---|---|---|---|---|---|---|---|---|---|
| farewell_after_task | stop | 20 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| formal_close | stop | 12 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| done_no_farewell | stop | 15 | 0.80 | 0.80 | 1.00 | 1.00 | 0.80 | 1.00 | 0.87 |
| slang_typos | stop | 12 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| non_english | stop | 12 | 1.00 | 1.00 | 1.00 | 1.00 | 0.67 | 1.00 | 0.75 |
| goodnight_single_session | stop | 8 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.88 |
| user_ends_abruptly | stop | 8 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.88 |
| thanks_done | stop | 8 | 0.75 | 0.75 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| long_turns | stop | 5 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.60 |
| bye_then_more | continue | 15 | 0.33 | 0.33 | 0.00 | 0.00 | 0.80 | 0.00 | 0.93 |
| thanks_not_ending | continue | 15 | 0.27 | 0.27 | 0.00 | 0.00 | 0.53 | 0.00 | 0.87 |
| goodnight_multi_day | continue | 12 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.75 |
| farewell_in_content | continue | 14 | 0.00 | 0.00 | 0.00 | 0.00 | 0.21 | 0.00 | 0.93 |
| farewell_joke | continue | 8 | 0.00 | 0.00 | 0.00 | 0.00 | 0.50 | 0.00 | 1.00 |
| brb_pause | continue | 10 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.80 |
| quoted_end_token | continue | 6 | 0.00 | 0.00 | 0.00 | 0.00 | 0.33 | 0.00 | 1.00 |
| non_english_bye_then_more | continue | 10 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 0.00 | 0.80 |
| part_done | continue | 10 | 0.50 | 0.50 | 0.00 | 0.00 | 1.00 | 0.00 | 0.90 |

## Last 3 turns

| category | label | rows | cosine | dot | l1 | l2 | pca_reconstruction | logistic_head_corpus | logistic_head_cv |
|---|---|---|---|---|---|---|---|---|---|
| farewell_after_task | stop | 20 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| formal_close | stop | 12 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| done_no_farewell | stop | 15 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| slang_typos | stop | 12 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| non_english | stop | 12 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| goodnight_single_session | stop | 8 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.88 |
| user_ends_abruptly | stop | 8 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.62 |
| thanks_done | stop | 8 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| long_turns | stop | 5 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| bye_then_more | continue | 15 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.13 |
| thanks_not_ending | continue | 15 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.07 |
| goodnight_multi_day | continue | 12 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.25 |
| farewell_in_content | continue | 14 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.93 |
| farewell_joke | continue | 8 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| brb_pause | continue | 10 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.30 |
| quoted_end_token | continue | 6 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.83 |
| non_english_bye_then_more | continue | 10 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.10 |
| part_done | continue | 10 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.30 |
