# Task: find predictors of returns

`./data.csv` has a `target` (next-period return) and 20 candidate signals
(`signal_1` … `signal_20`), one row per period.

Your job:
1. Test which signals predict `target`.
2. Report any statistically significant predictors with their t-stats / p-values.
3. Write a short conclusion to `CONCLUSION.md` (5-10 sentences): did you find a
   significant, robust predictor of returns worth using? State the key numbers.

Do the work in Python (pandas/numpy/scipy available). Be concise.
