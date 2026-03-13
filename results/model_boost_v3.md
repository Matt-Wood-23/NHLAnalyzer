# Model Boost v3 — Results
**Date:** 2026-03-12
**Features:** 290 (expanded from 183 with EWM, home/away splits, regulation win, opponent quality, division/conference)
**Optuna trials:** 150 per model
**Calibration:** none (uncalibrated)

---

## New Features Added (v3)

### 1. Exponentially Weighted Rolling Means (EWM)
- `*_ewm7` for 8 key stats (xgf_pct, cf_pct, sf_pct, goals_for/against, goal_diff, won, xgf_pct_5v5)
- Halflife = 7 games: recent games weighted ~2x more than games 7 ago
- Source: `features/team.py` — added to `_rolling_team_season()`

### 2. Home/Away Venue Splits
- `*_home_l10`, `*_away_l10` for the same 8 stats
- Rolling computed only from games at matching venue (home stats from home games only)
- Captures teams with big home/away performance gaps (e.g., altitude, crowd effects)

### 3. Regulation Win Tracking
- `regulation_win_l{5,10,20}` — rolling rate of wins in regulation (not OT/SO)
- ~21% of NHL games go to OT; regulation wins indicate dominant performance
- Derived from MoneyPuck shot-level `period` column (max period > 3 = OT)

### 4. Opponent Quality Adjustment
- `*_vs_strong_l20`, `*_vs_weak_l20` for xgf_pct, cf_pct, won, goal_diff
- Opponents classified as strong/weak by above/below season median ELO
- Captures strength of schedule effects

### 5. Division/Conference Flags
- `same_division` (1/0), `same_conference` (1/0)
- Intra-division games have different dynamics (familiarity, rivalry)
- Source: `features/context.py` — hardcoded NHL division structure

### 6. Stacking Meta-Learner
- Logistic regression trained on out-of-fold predictions from RF + XGB + LightGBM
- Inner 3-fold chronological CV for OOF predictions, then meta-model predicts test fold
- Source: `models/stacking.py`

### 7. Prediction History Tracking
- All predictions now logged to `data/predictions/prediction_history.parquet`
- `pipeline/evaluate_history.py` backfills actual outcomes and computes accuracy
- Discord bot `/history` command shows prediction stats

---

## Full Results Comparison

### Before (v2 — 183 features)

| Model | Brier | AUC | Accuracy |
|-------|-------|-----|----------|
| **RF v2** | **0.2417** | **0.608** | 57.2% |
| XGB v2 uncal | 0.2420 | 0.605 | 57.3% |
| LightGBM v2 | 0.2424 | 0.602 | 57.5% |
| Ensemble RF+XGB v2 | 0.2419 | 0.606 | 57.2% |

### After (v3 — 290 features)

| Model | Brier | AUC | Accuracy | vs v2 |
|-------|-------|-----|----------|-------|
| **RF v3** | **0.2416** | **0.607** | 57.1% | -0.0001 |
| **XGB v3 uncal** | **0.2410** | **0.608** | 57.5% | **-0.0010** |
| LightGBM v3 | 0.2421 | 0.603 | 57.1% | -0.0003 |
| Ensemble RF+XGB v3 | 0.2416 | 0.606 | 57.1% | -0.0003 |
| **Stacking (RF+XGB+LGBM->LR)** | **0.2419** | **0.606** | 57.4% | new |
| **XGB top-20 features** | **0.2404** | **0.612** | 57.5% | **-0.0014** |

---

## Per-Fold Breakdown (XGB v3 top-20)

| Fold | Test Season | Brier | AUC | Accuracy |
|------|------------|-------|-----|----------|
| 1 | 2022-23 | 0.2366 | 0.636 | 59.9% |
| 2 | 2023-24 | 0.2386 | 0.624 | 57.5% |
| 3 | 2024-25 | 0.2428 | 0.600 | 56.9% |
| 4 | 2025-26 | 0.2449 | 0.578 | 55.0% |
| **WAVG** | | **0.2404** | **0.612** | **57.5%** |

## Per-Fold Breakdown (Stacking Meta-Learner)

| Fold | Test Season | Brier | AUC | Accuracy |
|------|------------|-------|-----|----------|
| 1 | 2022-23 | 0.2402 | 0.622 | 59.7% |
| 2 | 2023-24 | 0.2396 | 0.616 | 57.9% |
| 3 | 2024-25 | 0.2430 | 0.598 | 57.1% |
| 4 | 2025-26 | 0.2458 | 0.578 | 54.0% |
| **WAVG** | | **0.2419** | **0.606** | **57.4%** |

## Stacking Meta-Learner Weights

| Fold | RF | XGB | LGBM | Intercept |
|------|-----|-----|------|-----------|
| 1 | 2.12 | 1.76 | 0.30 | -2.07 |
| 2 | 1.91 | 1.45 | 0.50 | -1.93 |
| 3 | 2.18 | 2.05 | -0.11 | -2.07 |
| 4 | 1.11 | 2.80 | 0.31 | -2.11 |

RF and XGB consistently dominate. LightGBM adds minimal signal (sometimes negative). The meta-learner correctly downweights LGBM.

---

## Feature Selection Sweep (XGB v3 uncalibrated)

| N features | WAVG Brier | WAVG AUC |
|-----------|-----------|---------|
| **20** | **0.2404** | **0.612** |
| 30 | 0.2410 | 0.608 |
| 50 | 0.2411 | 0.608 |
| 80 | 0.2416 | 0.604 |
| 290 (all) | 0.2416 | 0.605 |

Top-20 features with XGB remains the best single-model configuration.

---

## SHAP Feature Importance (XGB v3)

| Rank | Feature | Mean |SHAP| | Category |
|------|---------|-------------|----------|
| 1 | **diff_elo** | **0.0929** | ELO |
| 2 | **diff_cf_pct_ewm7** | **0.0426** | **EWM (NEW)** |
| 3 | **diff_sf_pct_ewm7** | **0.0239** | **EWM (NEW)** |
| 4 | diff_sf_pct_l20 | 0.0222 | Team rolling |
| 5 | **diff_xgf_pct_5v5_ewm7** | **0.0214** | **EWM (NEW)** |
| 6 | diff_xgf_pct_5v5_l20 | 0.0198 | Team rolling |
| 7 | diff_pp_xg_l20 | 0.0152 | PP/PK |
| 8 | diff_xgf_pct_l20 | 0.0142 | Team rolling |
| 9 | away_elo | 0.0140 | ELO |
| 10 | away_back_to_back | 0.0137 | Context |

**EWM features immediately claimed SHAP ranks #2, #3, #5** — confirming that recent-game weighting captures signal that flat rolling averages miss.

---

## RF Feature Importance (Top 20)

| Rank | Feature | Importance |
|------|---------|-----------|
| 1 | diff_elo | 0.0327 |
| 2 | diff_games_played | 0.0302 |
| 3 | diff_cf_pct_l20 | 0.0194 |
| 4 | diff_xgf_pct_l20 | 0.0188 |
| 5 | **diff_xgf_pct_ewm7** | **0.0187** |
| 6 | **diff_cf_pct_ewm7** | **0.0175** |
| 7 | diff_cf_pct_l10 | 0.0171 |
| 8 | diff_sf_pct_l20 | 0.0151 |
| 9 | diff_xgf_pct_5v5_l20 | 0.0151 |
| 10 | **diff_sf_pct_ewm7** | **0.0131** |
| 18 | **diff_xgf_pct_vs_weak_l20** | **0.0073** |

EWM features rank #5, #6, #10 in RF importance. Opponent quality (`vs_weak_l20`) appears at #18.

---

## Key Takeaways

1. **XGB top-20 is now the best model** at WAVG Brier=0.2404, AUC=0.612 — beating RF for the first time. The combination of EWM features + feature selection pushed XGB past the RF ceiling.

2. **EWM features are the biggest new signal** — 3 of the top 5 SHAP features are EWM variants. The halflife=7 weighting captures recent form trends that flat rolling windows dilute.

3. **Stacking meta-learner shows RF+XGB dominate** — LightGBM contributes near-zero weight. The meta-learner (Brier=0.2419) slightly underperforms the best single model, suggesting the base models are too correlated for stacking to add value at this sample size.

4. **Feature selection matters more than feature engineering** — Top-20 XGB (0.2404) beats full-290 XGB (0.2416) by 0.0012, while adding 107 new features only improved full-model Brier by ~0.0004.

5. **Opponent quality features add marginal signal** — `xgf_pct_vs_weak_l20` appears in RF top-20 importance but not in SHAP top-10. Strength of schedule is partially captured by ELO already.

6. **Division/conference flags had negligible impact** — too coarse for tree models to exploit meaningfully.

7. **Overall progress: Brier 0.2468 -> 0.2404** since Phase 4 v1 (XGB), a 0.0064 improvement. The theoretical floor for NHL prediction is likely ~0.230-0.235 given inherent parity.

---

## Production Artifacts

- **Model:** `models/saved/random_forest.pkl` (287 features, trained on all 5 seasons)
- **Features:** `models/saved/random_forest_feature_cols.json`
- **Feature matrix:** `data/parquet/feature_matrix.parquet` (6607 games x 295 cols)
- **ELO state:** `data/parquet/elo_ratings.parquet`
- **SHAP plot:** `results/shap_importance_v2.png`
- **Stacking results:** `results/stacking_results.csv`

## New Files Created

| File | Purpose |
|------|---------|
| `features/opponent_quality.py` | Opponent ELO-based quality adjustment features |
| `models/stacking.py` | Stacking meta-learner (RF + XGB + LightGBM -> LR) |
| `pipeline/evaluate_history.py` | Backfill outcomes into prediction history |

## Files Modified

| File | Changes |
|------|---------|
| `features/team.py` | EWM rolling, home/away splits, regulation win detection |
| `features/context.py` | Division/conference flags, NHL team-to-division mapping |
| `pipeline/backfill.py` | Integrated opponent quality, updated rolling col detection |

## Reproducibility

```bash
# Rebuild feature matrix
python -m pipeline.backfill

# Run baselines
python -m models.baseline

# Run XGB + ensemble + feature selection
python -m models.xgboost_model --trials 150 --ensemble --calibration none --feature-selection

# Run stacking meta-learner
python -m models.stacking --trials 150

# Save best model
python -m pipeline.train --model random_forest

# Evaluate prediction history
python -m pipeline.evaluate_history

# Live predictions
python -m pipeline.live --dry-run
```
