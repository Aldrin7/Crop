# Technical Critique of Crop-Research Pipeline v3.1

This document provides a comprehensive technical critique of the Crop-Research project, evaluating the methodology presented in `paper_draft.md` against its actual Python implementation (`pipeline.py` and `src/`). Updated for v3.1.

---

## 1. ~~Discrepancy in "Leak-Free" Feature Selection and Nested Cross-Validation~~ ✅ RESOLVED (v3.1)

**Original Critique (v3.0):** The paper claimed "leak-free nested cross-validation" but the implementation used a standard `StratifiedKFold` with pre-selected global feature subsets — constituting data leakage.

**v3.1 Fix:** The implementation now uses `sklearn.pipeline.Pipeline` per CV fold:
- `StandardScaler` → fitted on training fold only
- `SelectKBest(mutual_info_classif)` → feature selection per fold
- `Classifier` → final estimator

This is implemented in `pipeline.py` via `_build_pipeline()` which creates a `Pipeline([ ('scaler', StandardScaler()), ('fs', TopKFromScores('mutual_info', k=k)), ('clf', clf) ])` for each fold. The `TopKFromScores` class in `src/feature_selection.py` is a proper `sklearn.base.BaseEstimator + TransformerMixin` that fits MI scores on the training fold only.

**Remaining note:** The paper no longer claims "nested CV" — it correctly describes "5-fold stratified CV with per-fold Pipeline." The inner CV loop from config (`CV_INNER=3`) is unused; this is honest since no hyperparameter tuning is performed.

---

## 2. Methodology of Data Processing

**Partial fix (v3.1):**
- Scaling is now inside the Pipeline per fold ✅
- `class_weight='balanced'` added to RF, SVM, DT, LR ✅

**Remaining concerns:**
- Median imputation is used for missing values. For the secondary dataset with ~3% real dropout, `IterativeImputer` or `KNNImputer` might better preserve inter-feature correlations (e.g., between N, P, K).
- Outlier detection (IQR) is report-only — no action taken. This is defensible (preserving real-world variance) but should be acknowledged.

---

## 3. ~~Class Imbalance and Classifier Evaluation~~ ✅ RESOLVED (v3.1)

**Original Critique (v3.0):** No explicit handling of class imbalance; accuracy used as primary metric.

**v3.1 Fix:**
- `class_weight='balanced'` applied to RF, SVM, DT, LogisticRegression ✅
- Secondary best improved: RF 91.25% (sec_mi_top_6) with κ=0.8364 ✅
- GaussianNB failure on imbalanced data (50.91% → 80.11% with MI selection) explicitly acknowledged as a limitation ✅

**Remaining concern:**
- Gradient Boosting, XGBoost, LightGBM, KNN, MLP, and GaussianNB do not use `class_weight`. For GB/XGB/LGB, native `sample_weight` or `scale_pos_weight` could help. SMOTE oversampling was not explored.
- The gap between accuracy (91.25%) and macro-F1 (81.85%) for the best secondary classifier shows the minority class remains challenging.

---

## 4. Sensor Degradation Simulation

**No change from v3.0.** The critique remains valid:

- Degradation variants are created upfront and tested against a single pre-trained static model.
- No evaluation of training with noise augmentation as a mitigation strategy.
- The methodology provides a robustness *measurement* but not a robustness *solution*.

**Recommendation for future work:** Train on degraded variants (data augmentation) and evaluate whether models become more robust to sensor drift.

---

## 5. SHAP Explainability Depth

**No change from v3.0.** The critique remains valid:

- SHAP analysis provides global feature importance (bar charts of mean |SHAP|) but lacks:
  - SHAP interaction values
  - Local explanations for specific misclassified samples
  - Per-class SHAP breakdowns (which features matter for which crops)

**Recommendation for future work:** Add SHAP waterfall plots for representative samples per class and SHAP dependence plots for top features.

---

## 6. Hyperparameter Tuning (New)

**Not addressed in v3.1.** All classifiers use fixed hyperparameters. The paper acknowledges this in Limitations (Section 6.3, item 5).

**Impact:** Results represent a lower bound on classifier performance. Bayesian optimisation (Optuna) could narrow gaps between classifiers and identify better configurations, particularly for SVM (C, γ) and gradient boosting methods (learning rate, max depth, subsample).

---

## Summary Table

| # | Issue | v3.0 Status | v3.1 Status |
|---|-------|------------|------------|
| 1 | Data leakage in FS + scaling | ❌ Severe | ✅ Fixed (Pipeline per fold) |
| 2 | Preprocessing choices | ⚠️ Minor | ⚠️ Minor (scaling fixed, imputation unchanged) |
| 3 | Class imbalance handling | ❌ Missing | ✅ Fixed (class_weight='balanced') |
| 4 | Sensor degradation depth | ⚠️ Shallow | ⚠️ Shallow (unchanged) |
| 5 | SHAP depth | ⚠️ Basic | ⚠️ Basic (unchanged) |
| 6 | Hyperparameter tuning | ❌ Missing | ⚠️ Acknowledged (not implemented) |

---

## Conclusion

The v3.1 revision addresses the two most critical issues: data leakage and class imbalance. The Pipeline-per-fold architecture is now correctly implemented and the paper honestly describes the methodology. The remaining issues (SHAP depth, degradation augmentation, hyperparameter tuning) are acknowledged as limitations and future work, which is appropriate for the current study.
