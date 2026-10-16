# E8B / E12 / E11B — six fixed ablations

Percent values. Each scene is equally weighted. Partial scene sets are explicitly marked.
Primary baselines were fixed before this run: E8B M3 conditional_only; E12 S1 100% head_only; E11B CP.
All original controls/readouts remain in all_metrics.csv. Historical test: fold 0 only.
Ranking metrics use continuous scores; FPR/Recall/F1 use the actual six alarm policies.

## E8B — M3__conditional_only

| Configuration | AUROC | AP | FPR | Recall | F1 | Coverage |
|---|---:|---:|---:|---:|---:|---|
| Baseline | 79.40 | 72.46 | 16.76 | 44.82 | 44.66 | 4/4 scenes |
| + Adaptive threshold only | 79.40 | 72.46 | 15.66 | 44.07 | 44.21 | 4/4 scenes |
| + Stall alarm only | 79.40 | 72.46 | 17.01 | 49.66 | 49.22 | 4/4 scenes |
| + Stall score fusion only | 79.62 | 74.66 | 16.85 | 48.54 | 47.64 | 4/4 scenes |
| + Adaptive threshold + Fusion | 79.62 | 74.66 | 11.39 | 41.04 | 44.58 | 4/4 scenes |
| + Adaptive threshold + Fusion + Stall alarm | 79.62 | 74.66 | 11.49 | 41.27 | 44.70 | 4/4 scenes |

## E12 — S1_f100__head_only

| Configuration | AUROC | AP | FPR | Recall | F1 | Coverage |
|---|---:|---:|---:|---:|---:|---|
| Baseline | 79.39 | 72.50 | 16.79 | 44.97 | 44.78 | 4/4 scenes |
| + Adaptive threshold only | 79.39 | 72.50 | 15.66 | 44.16 | 44.29 | 4/4 scenes |
| + Stall alarm only | 79.39 | 72.50 | 17.04 | 49.81 | 49.34 | 4/4 scenes |
| + Stall score fusion only | 79.66 | 74.75 | 16.77 | 48.69 | 47.84 | 4/4 scenes |
| + Adaptive threshold + Fusion | 79.66 | 74.75 | 10.77 | 40.43 | 44.69 | 4/4 scenes |
| + Adaptive threshold + Fusion + Stall alarm | 79.66 | 74.75 | 10.90 | 40.72 | 44.85 | 4/4 scenes |

## E11B — CP

| Configuration | AUROC | AP | FPR | Recall | F1 | Coverage |
|---|---:|---:|---:|---:|---:|---|
| Baseline | 79.79 | 73.26 | 12.34 | 36.57 | 36.67 | 4/4 scenes |
| + Adaptive threshold only | 79.79 | 73.26 | 4.54 | 17.10 | 26.59 | 4/4 scenes |
| + Stall alarm only | 79.79 | 73.26 | 12.59 | 41.41 | 42.63 | 4/4 scenes |
| + Stall score fusion only | 79.05 | 74.06 | 0.75 | 11.53 | 19.41 | 4/4 scenes |
| + Adaptive threshold + Fusion | 79.05 | 74.06 | 0.73 | 11.04 | 18.56 | 4/4 scenes |
| + Adaptive threshold + Fusion + Stall alarm | 79.05 | 74.06 | 0.99 | 12.90 | 21.51 | 4/4 scenes |
