# Model Boost v2 — Results
**Date:** 2026-03-12
**Features:** 183 (expanded from 133 with goalie, PP/PK, ELO)
**Optuna trials:** 150 per model
**Calibration:** none (uncalibrated — confirmed better than isotonic/Platt)

---

## New Features Added

### 1. Goalie Features (from MoneyPuck shot CSVs)
- `home_g_save_pct_l5/l10`, `away_g_save_pct_l5/l10` — starter save% rolling
- `home_g_gsax_l5/l10`, `away_g_gsax_l5/l10` — goals saved above expected
- Source: `features/goalie_mp.py` — bypasses Postgres dependency

### 2. Special Teams (PP/PK from MoneyPuck)
- `pp_goals_l10/l20`, `pp_xg_l10/l20`, `pp_shots_l10/l20` — power play offense
- `pk_goals_against_l10/l20`, `pk_xg_against_l10/l20`, `pk_shots_against_l10/l20` — penalty kill defense
- home/away/diff prefixes (36 columns total)
- Source: `features/special_teams.py`

### 3. ELO Ratings
- `home_elo`, `away_elo`, `diff_elo` — recursive team strength (K=20, HA=50, regress=0.33)
- Source: `features/elo.py`

---

## Full Results Comparison

### Before (Phase 3/4 v1 — 133 features)

| Model | Brier | AUC | Accuracy |
|-------|-------|-----|----------|
| **RF (Phase 3)** | **0.2419** | **0.607** | 57.3% |
| LR (Phase 3) | 0.2468 | 0.603 | 57.8% |
| XGB calibrated (Phase 4) | 0.2468 | 0.597 | 56.1% |
| LightGBM calibrated (Phase 4) | 0.2482 | 0.592 | 55.6% |

### After (Boost v2 — 183 features, uncalibrated)

| Model | Brier | AUC | Accuracy | vs old |
|-------|-------|-----|----------|--------|
| **RF v2** | **0.2417** | **0.608** | 57.2% | -0.0002 |
| LR v2 | 0.2501 | 0.600 | 56.6% | +0.0033 |
| **XGB v2 uncal** | **0.2420** | **0.605** | 57.3% | **-0.0048** |
| **LightGBM v2 uncal** | **0.2424** | **0.602** | 57.5% | **-0.0058** |
| **Ensemble RF+XGB** | **0.2419** | **0.606** | 57.2% | new |
| XGB top-20 features | 0.2418 | 0.606 | 57.4% | new |

---

## Per-Fold Breakdown (RF v2)

| Fold | Test Season | Brier | AUC | Accuracy |
|------|------------|-------|-----|----------|
| 1 | 2022-23 | 0.2409 | 0.621 | 58.5% |
| 2 | 2023-24 | 0.2387 | 0.620 | 57.3% |
| 3 | 2024-25 | 0.2433 | 0.598 | 57.4% |
| 4 | 2025-26 | 0.2449 | 0.583 | 54.9% |
| **WAVG** | | **0.2417** | **0.608** | **57.2%** |

## Per-Fold Breakdown (XGB v2 uncalibrated)

| Fold | Test Season | Brier | AUC | Accuracy |
|------|------------|-------|-----|----------|
| 1 | 2022-23 | 0.2400 | 0.623 | 59.1% |
| 2 | 2023-24 | 0.2413 | 0.608 | 57.5% |
| 3 | 2024-25 | 0.2429 | 0.600 | 55.9% |
| 4 | 2025-26 | 0.2460 | 0.579 | 55.8% |
| **WAVG** | | **0.2420** | **0.605** | **57.3%** |

## Per-Fold Breakdown (Ensemble RF+XGB)

| Fold | Test Season | Brier | AUC | Accuracy |
|------|------------|-------|-----|----------|
| 1 | 2022-23 | 0.2398 | 0.624 | 59.4% |
| 2 | 2023-24 | 0.2402 | 0.613 | 56.8% |
| 3 | 2024-25 | 0.2429 | 0.600 | 56.7% |
| 4 | 2025-26 | 0.2455 | 0.580 | 55.5% |
| **WAVG** | | **0.2419** | **0.606** | **57.2%** |

---

## SHAP Feature Importance (XGBoost v2)

| Rank | Feature | Mean |SHAP| | Category |
|------|---------|-------------|----------|
| 1 | **diff_elo** | **0.1308** | ELO (NEW) |
| 2 | diff_sf_pct_l20 | 0.0439 | Team rolling |
| 3 | diff_cf_pct_l10 | 0.0350 | Team rolling |
| 4 | away_sf_pct_l20 | 0.0289 | Team rolling |
| 5 | diff_xgf_pct_5v5_l20 | 0.0269 | Team rolling |
| 6 | away_back_to_back | 0.0221 | Context |
| 7 | away_elo | 0.0199 | ELO (NEW) |
| 8 | diff_xgf_pct_l20 | 0.0189 | Team rolling |
| 9 | **diff_pp_xg_l20** | **0.0177** | **PP/PK (NEW)** |
| 10 | **away_pp_goals_l10** | **0.0165** | **PP/PK (NEW)** |
| 11 | diff_cf_pct_l20 | 0.0165 | Team rolling |
| 12 | rest_advantage | 0.0163 | Context |
| 13 | home_elo | 0.0149 | ELO (NEW) |

**ELO is the #1 feature by a factor of 3x.** PP/PK features break into the top 10.

---

## Feature Selection Sweep (XGB uncalibrated)

| N features | WAVG Brier | WAVG AUC |
|-----------|-----------|---------|
| 20 | **0.2418** | 0.606 |
| 30 | 0.2423 | 0.605 |
| 50 | 0.2426 | 0.603 |
| 80 | 0.2427 | 0.602 |
| 183 (all) | 0.2420 | 0.606 |

Top-20 features XGB performs best. With noisy data, fewer features reduces overfitting for gradient boosting.

---

## Key Takeaways

1. **ELO is the single most impactful new feature** — dominates SHAP at 0.1308, 3x the next best. Simple recursive team strength tracking adds more signal than any single rolling stat.

2. **XGBoost closed the gap with RF** — from 0.0049 behind (0.2468 vs 0.2419) to only 0.0003 behind (0.2420 vs 0.2417). The combination of new features + uncalibrated probabilities fixed XGB's Phase 4 underperformance.

3. **Uncalibrated > calibrated** — removing calibration improved XGB Brier by ~0.005, confirming the Phase 4 analysis that isotonic calibration was hurting on small NHL samples.

4. **PP/PK features add signal** — `diff_pp_xg_l20` and `away_pp_goals_l10` appear in SHAP top 10, demonstrating that special teams efficiency is a real predictor not captured by 5v5 stats.

5. **Goalie features had limited impact** — goalie save% rolling averages did not rank in SHAP top 20. This may be because: (a) 18% NaN rate reduces effective signal, (b) goalie quality correlates with team quality (already captured by ELO), or (c) goalie variance in save% is more noise than signal at the rolling-5/10 level.

6. **RF remains best model** at Brier=0.2417, but the margin over XGB/ensemble is now razor-thin.

---

## Production Artifacts

- **Model:** `models/saved/random_forest.pkl` (180 features, trained on all 5 seasons)
- **Features:** `models/saved/random_forest_feature_cols.json`
- **Feature matrix:** `data/parquet/feature_matrix.parquet` (6607 games x 188 cols)
- **ELO state:** `data/parquet/elo_ratings.parquet`
- **Goalie stats:** `data/parquet/goalie_game_stats.parquet`
- **SHAP plot:** `results/shap_importance_v2.png`

## Reproducibility

```bash
# Rebuild feature matrix
python -m pipeline.backfill

# Run baselines
python -m models.baseline

# Run XGB + ensemble + feature selection
python -m models.xgboost_model --trials 150 --ensemble --calibration none --feature-selection

# Save best model
python -m pipeline.train --model random_forest

# Live predictions
python -m pipeline.live --dry-run
```
