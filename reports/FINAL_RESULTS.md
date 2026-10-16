# CycleVAD completed experiment results — 2026-10-16

All requested seeds (42/43/44), folds (0–4), scenes (R01–R04) completed. 180 experiment/seed/scene/fold units passed; 15,808 baseline metric comparisons passed. Source/configuration/runner hashes remain unchanged. Runtime: 79.26 minutes.

Historical test uses fold0 only. Normal evaluation uses all five out-of-fold sets. These tables are the equal-scene macro historical metrics (%). F1 is computed per scene from actual alarms and then averaged. No test-based retuning occurred in this run. Prior exploratory data informed the fixed extension, so these are not independent confirmation results.

## Seed 42 — primary

### E8B: M3__conditional_only

| Configuration | AUROC | AP | FPR | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| Baseline | 79.40 | 72.46 | 16.76 | 44.82 | 44.66 |
| Adaptive threshold only | 79.40 | 72.46 | 15.66 | 44.07 | 44.21 |
| Stall alarm only | 79.40 | 72.46 | 17.01 | 49.66 | 49.22 |
| Stall score fusion only | 79.62 | 74.66 | 16.85 | 48.54 | 47.64 |
| Adaptive threshold + Fusion | 79.62 | 74.66 | 11.39 | 41.04 | 44.58 |
| Adaptive threshold + Fusion + Stall alarm | 79.62 | 74.66 | 11.49 | 41.27 | 44.70 |

### E12: S1_f100__head_only

| Configuration | AUROC | AP | FPR | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| Baseline | 79.39 | 72.50 | 16.79 | 44.97 | 44.78 |
| Adaptive threshold only | 79.39 | 72.50 | 15.66 | 44.16 | 44.29 |
| Stall alarm only | 79.39 | 72.50 | 17.04 | 49.81 | 49.34 |
| Stall score fusion only | 79.66 | 74.75 | 16.77 | 48.69 | 47.84 |
| Adaptive threshold + Fusion | 79.66 | 74.75 | 10.77 | 40.43 | 44.69 |
| Adaptive threshold + Fusion + Stall alarm | 79.66 | 74.75 | 10.90 | 40.72 | 44.85 |

### E11B: CP

| Configuration | AUROC | AP | FPR | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| Baseline | 79.79 | 73.26 | 12.34 | 36.57 | 36.67 |
| Adaptive threshold only | 79.79 | 73.26 | 4.54 | 17.10 | 26.59 |
| Stall alarm only | 79.79 | 73.26 | 12.59 | 41.41 | 42.63 |
| Stall score fusion only | 79.05 | 74.06 | 0.75 | 11.53 | 19.41 |
| Adaptive threshold + Fusion | 79.05 | 74.06 | 0.73 | 11.04 | 18.56 |
| Adaptive threshold + Fusion + Stall alarm | 79.05 | 74.06 | 0.99 | 12.90 | 21.51 |

## Seed 43 — robustness

### E8B: M3__conditional_only

| Configuration | AUROC | AP | FPR | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| Baseline | 79.35 | 72.41 | 16.93 | 45.55 | 45.45 |
| Adaptive threshold only | 79.35 | 72.41 | 15.90 | 44.66 | 44.86 |
| Stall alarm only | 79.35 | 72.41 | 17.18 | 50.39 | 50.00 |
| Stall score fusion only | 79.65 | 74.68 | 16.84 | 48.97 | 48.21 |
| Adaptive threshold + Fusion | 79.65 | 74.68 | 12.04 | 42.09 | 45.13 |
| Adaptive threshold + Fusion + Stall alarm | 79.65 | 74.68 | 12.12 | 42.30 | 45.24 |

### E12: S1_f100__head_only

| Configuration | AUROC | AP | FPR | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| Baseline | 79.42 | 72.50 | 16.92 | 45.61 | 45.53 |
| Adaptive threshold only | 79.42 | 72.50 | 15.77 | 44.82 | 45.10 |
| Stall alarm only | 79.42 | 72.50 | 17.17 | 50.46 | 50.08 |
| Stall score fusion only | 79.71 | 74.77 | 16.96 | 49.17 | 48.42 |
| Adaptive threshold + Fusion | 79.71 | 74.77 | 11.18 | 41.14 | 45.23 |
| Adaptive threshold + Fusion + Stall alarm | 79.71 | 74.77 | 11.28 | 41.39 | 45.35 |

### E11B: CP

| Configuration | AUROC | AP | FPR | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| Baseline | 79.76 | 73.29 | 12.58 | 36.46 | 36.40 |
| Adaptive threshold only | 79.76 | 73.29 | 4.60 | 17.52 | 26.98 |
| Stall alarm only | 79.76 | 73.29 | 12.83 | 41.31 | 42.37 |
| Stall score fusion only | 79.06 | 74.12 | 0.77 | 11.47 | 19.31 |
| Adaptive threshold + Fusion | 79.06 | 74.12 | 0.74 | 10.93 | 18.38 |
| Adaptive threshold + Fusion + Stall alarm | 79.06 | 74.12 | 1.01 | 12.79 | 21.33 |

## Seed 44 — robustness

### E8B: M3__conditional_only

| Configuration | AUROC | AP | FPR | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| Baseline | 79.43 | 72.57 | 16.92 | 45.32 | 45.23 |
| Adaptive threshold only | 79.43 | 72.57 | 15.92 | 44.52 | 44.70 |
| Stall alarm only | 79.43 | 72.57 | 17.15 | 49.75 | 49.13 |
| Stall score fusion only | 79.62 | 74.59 | 16.79 | 49.53 | 48.18 |
| Adaptive threshold + Fusion | 79.62 | 74.59 | 10.56 | 41.90 | 45.83 |
| Adaptive threshold + Fusion + Stall alarm | 79.62 | 74.59 | 10.65 | 42.14 | 45.96 |

### E12: S1_f100__head_only

| Configuration | AUROC | AP | FPR | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| Baseline | 79.35 | 72.43 | 16.89 | 45.39 | 45.29 |
| Adaptive threshold only | 79.35 | 72.43 | 15.92 | 44.59 | 44.77 |
| Stall alarm only | 79.35 | 72.43 | 17.12 | 49.82 | 49.19 |
| Stall score fusion only | 79.57 | 74.49 | 16.81 | 49.61 | 48.21 |
| Adaptive threshold + Fusion | 79.57 | 74.49 | 11.80 | 43.01 | 45.70 |
| Adaptive threshold + Fusion + Stall alarm | 79.57 | 74.49 | 11.88 | 43.21 | 45.83 |

### E11B: CP

| Configuration | AUROC | AP | FPR | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| Baseline | 79.86 | 73.63 | 12.42 | 36.44 | 36.44 |
| Adaptive threshold only | 79.86 | 73.63 | 4.52 | 17.13 | 26.59 |
| Stall alarm only | 79.86 | 73.63 | 12.65 | 40.87 | 41.73 |
| Stall score fusion only | 79.01 | 74.01 | 0.75 | 11.20 | 18.79 |
| Adaptive threshold + Fusion | 79.01 | 74.01 | 0.72 | 10.67 | 17.88 |
| Adaptive threshold + Fusion + Stall alarm | 79.01 | 74.01 | 0.99 | 12.47 | 20.72 |

## Interpretation

- Seed42 stall-alarm-only F1: E8B 44.66 to 49.22; E12 44.78 to 49.34; E11B 36.67 to 42.63. FPR increases in each comparison.
- Full combination improves AP in all three experiments across all three seeds. E8B/E12 full-combination AUROC improves, while E11B AUROC decreases.
- Full-combination F1 is not uniformly improved: it decreases for E8B/E12 at seed43, and falls substantially for E11B at all three seeds.
- In E11B the original window-maximum calibration makes fusion conservative: seed42 full FPR falls from 12.34 to 0.99, while Recall falls from 36.57 to 12.90. Low FPR alone is not a general improvement.
- Threshold/alarm-only changes leave continuous-score AUROC/AP unchanged by design. No best setting is selected or retuned from these results.

## Files

- TABLES.md: primary seed42 tables
- all_metrics.csv: all original controls/readouts, seeds, folds and scene metrics
- macro.json: all complete historical macros
- ../provenance/final_validation.json: final completeness/equivalence checks
- ../results/ablation/: per-video numerical results, baseline checks and upstream baseline deltas; per-frame NPZ predictions and fitted bundles stay local