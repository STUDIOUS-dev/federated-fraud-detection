# Real-data study (OpenML creditcard)

Separate from the frozen synthetic results. Temporal split by row order, 4 simulated banks (unequal sizes, skewed fraud share), 5 seeds, 10 rounds. Test set has only 75 frauds, so differences are noisy. The banks are a simulated partition of one dataset, not real institutions.

| Setup | PR-AUC | ROC-AUC | Recall@1%FPR |
|---|---|---|---|
| Centralized | 0.798 ± 0.008 | 0.976 ± 0.008 | 0.883 ± 0.005 |
| Local (avg of banks) | 0.791 ± 0.005 | 0.952 ± 0.002 | 0.849 ± 0.006 |
| FedAvg | 0.803 ± 0.010 | 0.970 ± 0.007 | 0.867 ± 0.012 |
| FedMedian | 0.806 ± 0.006 | 0.981 ± 0.006 | 0.885 ± 0.007 |
| FedAvg + attacker | 0.287 ± 0.272 | 0.783 ± 0.158 | 0.435 ± 0.356 |
| FedMedian + attacker | 0.806 ± 0.006 | 0.979 ± 0.009 | 0.877 ± 0.013 |
