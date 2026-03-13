# Baseline Model Results — v1
**Date:** 2026-03-09
**Data:** 6,607 completed games, 5 seasons (2021-22 through 2025-26 in-progress)
**Validation:** Walk-forward expanding window, 4 folds
**Features:** 136 (rolling xG/Corsi/HD × 5/10/20 windows, home/away/diff, context)

---

## Walk-Forward Folds

| Fold | Train | Test | Games (train/test) |
|------|-------|------|--------------------|
| 1 | 2021-22 | 2022-23 | 1,400 / 1,400 |
| 2 | 2021-22 → 2022-23 | 2023-24 | 2,800 / 1,400 |
| 3 | 2021-22 → 2023-24 | 2024-25 | 4,200 / 1,398 |
| 4 | 2021-22 → 2024-25 | 2025-26 | 5,598 / 1,009 |

---

## Logistic Regression (L2, C=1.0)

| Fold | Test Season | Accuracy | Brier | Log Loss | AUC | Beat Naive |
|------|------------|----------|-------|----------|-----|-----------|
| 1 | 2022-23 | 0.594 | 0.2491 | 0.7714 | 0.621 | 0.591 |
| 2 | 2023-24 | 0.572 | 0.2463 | 0.6906 | 0.604 | 0.574 |
| 3 | 2024-25 | 0.594 | 0.2450 | 0.6855 | 0.608 | 0.592 |
| 4 | 2025-26 | 0.543 | 0.2471 | 0.6871 | 0.567 | 0.555 |
| **WAVG** | | **0.578** | **0.2468** | **0.7103** | **0.603** | **0.579** |

---

## Random Forest (200 trees, max_depth=6, min_samples_leaf=20)

| Fold | Test Season | Accuracy | Brier | Log Loss | AUC | Beat Naive |
|------|------------|----------|-------|----------|-----|-----------|
| 1 | 2022-23 | 0.589 | 0.2403 | 0.6737 | 0.625 | 0.590 |
| 2 | 2023-24 | 0.578 | 0.2392 | 0.6710 | 0.619 | 0.577 |
| 3 | 2024-25 | 0.572 | 0.2439 | 0.6810 | 0.595 | 0.570 |
| 4 | 2025-26 | 0.544 | 0.2449 | 0.6829 | 0.582 | 0.546 |
| **WAVG** | | **0.573** | **0.2419** | **0.6767** | **0.607** | **0.573** |

---

## Naive Baseline (predict training-set home win rate each fold)

| Metric | Value |
|--------|-------|
| Home win rate (avg across folds) | ~50.2% |
| Brier score (WAVG) | **0.2501** |
| Accuracy | ~50.2% |
| AUC | 0.500 |

---

## Summary vs Naive Baseline

| Model | Brier ↓ | Δ Brier vs Naive | AUC ↑ | Accuracy ↑ |
|-------|---------|-----------------|-------|-----------|
| Logistic Regression | 0.2468 | **−0.0033** | 0.603 | 57.8% |
| Random Forest | **0.2419** | **−0.0082** | **0.607** | 57.3% |
| Naive | 0.2501 | — | 0.500 | ~50% |

RF beats naive Brier by **0.0082** — both models are meaningfully better than random.

---

## Top Features

### Logistic Regression — Top 15 by |coefficient|
| Feature | |Coef| |
|---------|---------|
| away_xgf_pct_5v5_l20 | 0.7665 |
| away_xg_against_5v5_l20 | 0.5603 |
| away_xg_for_5v5_l20 | 0.4348 |
| home_xg_for_l10 | 0.4062 |
| diff_xg_against_5v5_l10 | 0.3811 |
| diff_xg_against_5v5_l5 | 0.3584 |
| diff_xgf_pct_5v5_l10 | 0.3502 |
| home_xgf_pct_l20 | 0.3417 |
| diff_xgf_pct_5v5_l5 | 0.3263 |
| away_sf_pct_l10 | 0.3155 |

### Random Forest — Top 15 by Importance
| Feature | Importance |
|---------|-----------|
| diff_games_played ⚠️ | 0.0442 |
| diff_xg_for_l20 | 0.0304 |
| diff_cf_pct_l20 | 0.0300 |
| diff_sf_pct_l10 | 0.0268 |
| diff_xgf_pct_5v5_l20 | 0.0251 |
| diff_xgf_pct_l20 | 0.0249 |
| diff_sf_pct_l20 | 0.0225 |
| diff_sf_pct_l5 | 0.0212 |
| diff_xgf_pct_5v5_l10 | 0.0198 |
| diff_cf_pct_l10 | 0.0190 |

> ⚠️ `diff_games_played` ranked #1 in RF despite being positional noise (season game count, not team quality). Drop before Phase 4.

---

## Observations & Notes

- **xGF% 5v5 is the dominant signal** across both models, consistent with hockey analytics consensus
- **Differential features outperform raw team values** — the gap between teams matters more than absolute level
- **Longer windows (l20) tend to dominate** over l5, suggesting stable team quality signals over ~3-4 weeks
- **Fold 4 (2025-26) shows the lowest AUC** (0.567/0.582) — likely because 2025-26 data in MoneyPuck is mid-season and any teams with recent roster changes are poorly represented in rolling features
- **Log loss Fold 1 is elevated** (0.7714 LR) — only 1 season of training data produces a poorly calibrated model; stabilizes by Fold 2
- **Context features (rest, back-to-back)** have ~5% NaN rate (season openers); imputed with mean — minimal impact expected

---

## Target Benchmarks for Phase 4 (XGBoost)

| Metric | Baseline (RF) | Phase 4 Target |
|--------|--------------|----------------|
| Brier score (WAVG) | 0.2419 | < 0.240 |
| AUC (WAVG) | 0.607 | > 0.620 |
| Accuracy (WAVG) | 57.3% | > 58% |
| Beat naive rate | 57.3% | > 59% |

---

## Reproducibility

```bash
# Regenerate feature matrix
python -m pipeline.backfill

# Re-run baselines
python -m models.baseline

# View MLflow runs
mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db
```

**MLflow experiment:** `nhl_baseline_models`
**Stored at:** `mlruns/mlflow.db`
