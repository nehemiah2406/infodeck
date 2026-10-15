# InfoDeck evaluation results

Run on 2026-10-15 with the 3 fictional sample PDFs (10 questions in `eval/questions.csv`).  
Embedding model `BAAI/bge-small-en-v1.5`, re-ranker `cross-encoder/ms-marco-MiniLM-L6-v2`, top-k 5, MIN_SCORE 0.55.

## Summary

| Metric | Result |
|---|---|
| Retrieval hit rate @5 (right file + page in top 5) | 8/8 (100%) |
| Hit rate @1 (right page ranked first) | 6/8 (75%) |
| "Not found" accuracy (questions not in the documents) | 1/2 (50%) |
| Answerable questions wrongly rejected | 0/8 |

## Per question

| # | Type | Question | Expected | Best score | Rank of expected page | Top passage |
|---|---|---|---|---|---|---|
| 1 | answerable | How many earned leave days do I get per year? | KRT_Leave_Policy.pdf, p.1 | 0.752 | 4 | KRT_Leave_Policy.pdf, p.2 |
| 2 | answerable | Do I need a medical certificate for sick leave? | KRT_Leave_Policy.pdf, p.2 | 0.762 | 2 | KRT_Leave_Policy.pdf, p.3 |
| 3 | answerable | What is the hotel limit per night in Mumbai? | KRT_Travel_and_Expense_Policy.pdf, p.2 | 0.717 | 1 | KRT_Travel_and_Expense_Policy.pdf, p.2 |
| 4 | answerable | Within how many days must I submit an expense claim? | KRT_Travel_and_Expense_Policy.pdf, p.3 | 0.718 | 1 | KRT_Travel_and_Expense_Policy.pdf, p.3 |
| 5 | answerable | How long must my password be? | KRT_IT_Security_Policy.pdf, p.1 | 0.750 | 1 | KRT_IT_Security_Policy.pdf, p.1 |
| 6 | answerable | How quickly must I report a lost laptop? | KRT_IT_Security_Policy.pdf, p.3 | 0.724 | 1 | KRT_IT_Security_Policy.pdf, p.3 |
| 7 | reworded | Can I paste client information into ChatGPT? | KRT_IT_Security_Policy.pdf, p.3 | 0.561 | 1 | KRT_IT_Security_Policy.pdf, p.3 |
| 8 | reworded | Can my wife come along on a work trip? | KRT_Travel_and_Expense_Policy.pdf, p.4 | 0.608 | 1 | KRT_Travel_and_Expense_Policy.pdf, p.4 |
| 9 | not_in_docs | What is the dental insurance coverage? | not found | 0.537 | correctly not found | - |
| 10 | not_in_docs | What is the password for the guest WiFi? | not found | 0.607 | **passed the threshold** | KRT_IT_Security_Policy.pdf, p.3 |

## Threshold sweep (MIN_SCORE)

| MIN_SCORE | Hit rate @5 | Hit rate @1 | "Not found" accuracy | Wrongly rejected |
|---|---|---|---|---|
| 0.45 | 8/8 (100%) | 6/8 (75%) | 0/2 (0%) | 0 |
| 0.50 | 8/8 (100%) | 6/8 (75%) | 0/2 (0%) | 0 |
| 0.55 (current) | 8/8 (100%) | 6/8 (75%) | 1/2 (50%) | 0 |
| 0.60 | 7/8 (88%) | 5/8 (62%) | 1/2 (50%) | 1 |
| 0.65 | 6/8 (75%) | 4/8 (50%) | 2/2 (100%) | 2 |

Notes: these numbers measure retrieval only (no API key needed). With Claude enabled, the
answer prompt adds a second "not found" check: the model must reply "I could not find this in
the documents." when the retrieved passages do not contain the answer.
