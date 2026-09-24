# Results

`results/results/` holds the outputs of the full-data run (tuning iteration 2, `--max-epochs 40`): tuning logs, per-area metric tables, test-week predictions, worst-error points, EDA summary and hardware/timing metadata. Rerunning `python run.py train` (or the notebook) regenerates them. Metrics are deterministic; wall-clock times vary slightly.

`results/iteration1_max8_epochs/` archives tuning iteration 1 (8-epoch cap): its results, forecast figures and models. It is kept for comparison.
