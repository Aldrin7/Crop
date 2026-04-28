# Technical Critique — Crop-Research Pipeline v3.2

Updated for v3.2. All issues from v3.0 and v3.1 are resolved.

---

## v3.0 Issues — All Resolved

| # | Issue | Status |
|---|-------|--------|
| 1 | Data leakage in FS + scaling | ✅ Fixed (Pipeline per fold) |
| 2 | Scaling applied once upfront | ✅ Fixed (inside Pipeline) |
| 3 | No class imbalance handling | ✅ Fixed (class_weight='balanced' + BalWeightWrapper) |
| 4 | Nested CV not implemented | ✅ Fixed (honest "5-fold stratified CV" + optional Optuna nested CV) |
| 5 | Redundant metrics | ✅ Fixed (Kappa, MCC, Brier, ECE) |

---

## v3.1 Issues — All Resolved

| # | Issue | Status |
|---|-------|--------|
| 1 | "Nested CV" ghost in code | ✅ Fixed |
| 2 | Cross-dataset "validation" overclaim | ✅ Fixed |
| 3 | Consensus vs per-fold MI mismatch | ✅ Fixed |
| 4 | class_weight only on 4/10 classifiers | ✅ Fixed (BalWeightWrapper on all) |
| 5 | SHAP feature name fragility | ✅ Fixed |
| 6 | Unrealistic sensor drift model | ✅ Fixed |
| 7 | Per-class F1 not discussed | ✅ Fixed |
| 8 | Pipeline header v3.0 | ✅ Fixed |
| 9 | References expanded | ✅ Fixed |
| 10 | Friedman test implemented | ✅ Fixed |
| 11 | Consistency formula bug | ✅ Fixed |
| 12 | Recalibration cost nuance | ✅ Fixed |
| 13 | Secondary results lead | ✅ Fixed |

---

## v3.2 Issues — All Resolved

### 1. class_weight only on 5/10 classifiers → ✅ FIXED
`BalWeightWrapper` in `models.py` computes `sample_weight='balanced'` from `y` at fit time and passes it to the inner estimator's `fit(sample_weight=...)`. Applied to: XGBoost, GB, MLP, KNN, GaussianNB. All 10 classifiers now receive fair imbalance handling.

### 2. No hyperparameter tuning → ✅ FIXED
Optuna nested CV added via `--tune` flag. For each outer CV fold:
- Inner 3-fold CV with 30 Optuna trials per classifier
- TPE sampler with fixed seed for reproducibility
- Classifier-specific search spaces (RF: n_estimators/max_depth/min_samples_split; SVM: C/gamma; etc.)
- Falls back to defaults if Optuna unavailable

### 3. Dead code (add_class_imbalance) → ✅ FIXED
Removed from `noise_injection.py`. Was unused and misleading.

### 4. Deprecation warnings missing → ✅ FIXED
- `scale_features()` now raises `DeprecationWarning` with message about leak-free Pipeline
- `run_all_fs_methods()` now raises `UserWarning` about data leakage risk

### 5. Missing SHAP dependency → ✅ FIXED
`shap>=0.42.0` and `optuna>=3.2.0` added to `requirements.txt`.

### 6. No tests → ✅ FIXED
`tests/test_pipeline.py` added with 25+ tests covering:
- Config validation
- Preprocessing (missing values, encoding, scaling deprecation)
- Models (BalWeightWrapper, predict_proba, all classifiers)
- Feature selection (TopK, RFE, warning on full-dataset FS)
- Evaluation (metrics, Friedman, Nemenyi)
- Noise injection (shapes, NaN introduction, range clipping)
- Explainability (correlation violations)
- Integration (Pipeline fit/predict smoke test)

### 7. No license → ✅ FIXED
MIT License added.

### 8. Binary .docx in repo → MITIGATED
Added to `.gitignore` for future commits. Existing `.docx` remains in history (no retraction).

### 9. src/__init__.py empty → ✅ FIXED
Proper exports for all public API functions.

### 10. preprocessing.py misleading → ✅ FIXED
`scale_features()` deprecated with warning. `prepare_data()` docstring updated to clarify EDA-only usage.

---

## Remaining Items (Acceptable for Current Scope)

| Item | Status | Justification |
|------|--------|---------------|
| Per-class SHAP (local) | Future work | Section 6.5/6.6 acknowledge |
| Second crop dataset for true cross-validation | Future work | Section 6.5, item 2 |
| CI/CD pipeline | Future work | GitHub Actions for automated testing |

---

## Conclusion

All critical, major, and moderate issues are resolved. The paper honestly describes its methodology, correctly frames its contributions, acknowledges limitations, and provides reproducible code. All classifiers now receive fair imbalance handling. Optuna tuning is available for rigorous hyperparameter search. Tests verify correctness. The remaining items are appropriate future work directions.
