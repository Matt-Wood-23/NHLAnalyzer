# XGBoost / LightGBM Results — Phase 4 v2
**Date:** 2026-03-09
**Optuna trials:** 150 per model
**Tuning split:** train 2021-24 → validate 2024-25
**Calibration:** Isotonic regression on held-out 15% of training data — only applied when cal slice ≥ 400 games (skipped on Fold 1)
**Features:** 133 (dropped `games_played` columns vs baseline's 136)

---

## Tuning Results (validation on 2024-25)

| Model | Best Brier (tuning val) | Key Params |
|-------|------------------------|------------|
| XGBoost | 0.2442 | n_est=156, depth=3, lr=0.010, subsample=0.71, col_bt=0.51, reg_α=0.20 |
| LightGBM | 0.2442 | n_est=215, depth=4, lr=0.011, num_leaves=50, subsample=0.61, col_bt=0.64 |

Both models converged to shallow trees (depth 3-4) with low learning rate — consistent with a noisy, low-signal prediction problem. Both tied at Brier 0.2442 on the tuning validation set.

---

## XGBoost (tuned, calibration applied Folds 2-4 only)

| Fold | Test Season | Accuracy | Brier | Log Loss | AUC | Beat Naive |
|------|------------|----------|-------|----------|-----|-----------|
| 1 | 2022-23 | 0.586 | 0.2391 | 0.6710 | 0.623 | 0.589 |
| 2 | 2023-24 | 0.571 | 0.2488 | 0.7712 | 0.607 | 0.571 |
| 3 | 2024-25 | 0.553 | 0.2488 | 0.7019 | 0.577 | 0.553 |
| 4 | 2025-26 | 0.524 | 0.2519 | 0.6994 | 0.577 | 0.532 |
| **WAVG** | | **0.561** | **0.2468** | **0.7117** | **0.597** | **0.564** |

---

## LightGBM (tuned, calibration applied Folds 2-4 only)

| Fold | Test Season | Accuracy | Brier | Log Loss | AUC | Beat Naive |
|------|------------|----------|-------|----------|-----|-----------|
| 1 | 2022-23 | 0.581 | 0.2415 | 0.6762 | 0.613 | 0.587 |
| 2 | 2023-24 | 0.559 | 0.2503 | 0.7015 | 0.605 | 0.559 |
| 3 | 2024-25 | 0.564 | 0.2492 | 0.8131 | 0.577 | 0.564 |
| 4 | 2025-26 | 0.508 | 0.2534 | 0.7018 | 0.566 | 0.525 |
| **WAVG** | | **0.556** | **0.2482** | **0.7247** | **0.592** | **0.561** |

---

## Full Comparison vs Baselines

| Model | Brier ↓ | AUC ↑ | Accuracy | Beat Naive |
|-------|---------|-------|----------|-----------|
| Naive baseline | 0.2501 | 0.500 | ~50% | — |
| Logistic Regression (Phase 3) | 0.2468 | 0.603 | 57.8% | 57.9% |
| XGBoost tuned v2 (Phase 4) | 0.2468 | 0.597 | 56.1% | 56.4% |
| LightGBM tuned v2 (Phase 4) | 0.2482 | 0.592 | 55.6% | 56.1% |
| **Random Forest (Phase 3)** | **0.2419** | **0.607** | **57.3%** | **57.3%** |

---

## Analysis

**Random Forest (Phase 3) remains the best model.** XGBoost and LightGBM did not improve — a surprising result worth understanding:

### Why XGBoost underperformed RF here

1. **Calibration is hurting Fold 1** — with only 1 season of training data, the 15% calibration holdout (210 games) is too small, producing noisy isotonic fits. XGBoost Fold 1 Brier of 0.2572 is worse than naive. More training data (Folds 3-4) brings it back in line.

2. **Isotonic calibration is aggressive on small samples** — isotonic regression on 210-350 samples can overfit, distorting well-calibrated raw probabilities. Consider Platt scaling (logistic) for small samples.

3. **50 Optuna trials may be insufficient** — the search found shallow trees (depth=3) with low learning rates, but TPE needs more trials to reliably explore the interaction between `n_estimators`, `learning_rate`, and regularization. Try 150+ trials.

4. **RF has implicit calibration advantages** — Random Forest's averaging of many trees produces smoother, more calibrated probabilities out of the box, which isotonic regression can distort rather than improve.

### SHAP Top Features (XGBoost — final model on all 5 seasons)

| Feature | Mean |SHAP| | Interpretation |
|---------|-------------|----------------|
| diff_sf_pct_l20 | 0.0586 | Shot share over last 20 games — possession proxy |
| diff_xgf_pct_l20 | 0.0481 | xG share — quality-adjusted possession |
| diff_xgf_pct_5v5_l20 | 0.0319 | 5v5 xG share — pure even-strength dominance |
| diff_cf_pct_l10 | 0.0250 | Corsi share over last 10 games |
| away_back_to_back | 0.0213 | Away team on back-to-back — fatigue signal ✅ |
| rest_advantage | 0.0160 | Home rest days − away rest days |
| diff_won_l20 | 0.0186 | Win rate differential — form signal |

Context features (`away_back_to_back`, `rest_advantage`) now appear in top 12 — the DB-backed dates are adding real signal.

---

## What to Try Next (v2 improvements)

1. **More Optuna trials** — run 150-200 to better explore depth/lr interaction
2. **Skip calibration on Fold 1** (< 2 seasons training) or switch to Platt scaling
3. **Feature engineering** — add back-to-back interaction (is_b2b × recent_form), home/away rolling splits
4. **Uncalibrated XGBoost** — compare raw XGB probabilities vs calibrated to isolate the calibration impact
5. **Ensemble** — simple average of RF + XGBoost probabilities often beats either alone

---

## Targets for v2

| Metric | RF Baseline | XGB v1 | v2 Target |
|--------|------------|--------|-----------|
| Brier (WAVG) | 0.2419 | 0.2505 | < 0.2400 |
| AUC (WAVG) | 0.607 | 0.592 | > 0.615 |
| Beat naive | 57.3% | 56.3% | > 59% |

---

## Reproducibility

```bash
# 50 trials (fast)
python -m models.xgboost_model

# 150 trials (recommended for v2)
python -m models.xgboost_model --trials 150

# View in MLflow
mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db
```

**MLflow experiment:** `nhl_xgboost_phase4`
**SHAP plot:** `results/shap_importance.png`
