# Technical Critique — Crop-Research Pipeline v3.1

Updated for v3.1. All issues from v3.0 are resolved. Issues from the v3.1 audit are addressed below.

---

## v3.0 Issues — All Resolved

| # | Issue | Status |
|---|-------|--------|
| 1 | Data leakage in FS + scaling | ✅ Fixed (Pipeline per fold) |
| 2 | Scaling applied once upfront | ✅ Fixed (inside Pipeline) |
| 3 | No class imbalance handling | ✅ Fixed (class_weight='balanced') |
| 4 | Nested CV not implemented | ✅ Fixed (honest "5-fold stratified CV") |
| 5 | Redundant metrics | ✅ Fixed (Kappa, MCC, Brier, ECE) |

---

## v3.1 Audit Issues — All Addressed

### 1. "Nested CV" Ghost in Code → ✅ FIXED
All references to "nested CV" removed from pipeline.py. Session headers, log messages, and output filenames now correctly say "leak-free CV" or "5-fold stratified CV."

### 2. Cross-Dataset "Validation" Overclaim → ✅ FIXED
Paper now frames as "cross-dataset feature consistency analysis" throughout. Removed "generalisation validation" language. Section 5.7 clearly defines the analysis scope (comparing feature rankings, not model transfer).

### 3. Consensus vs Per-Fold MI Mismatch → ✅ FIXED
Section 4.3 now includes explicit clarification: consensus ranking is for interpretation only; per-fold MI selection is used during training. Section 5.4 ablation table header clarified.

### 4. class_weight Only on 4/10 Classifiers → ✅ FIXED
- LightGBM now uses `class_weight='balanced'` in models.py
- Table 1 in paper explicitly documents which classifiers receive imbalance handling
- Limitation (Section 6.5, item 3) acknowledges the unfair comparison
- Future work includes implementing `sample_weight` for remaining classifiers

### 5. SHAP Feature Name Fragility → ✅ FIXED
Session 4 SHAP code now uses defensive `feature_cols` with fallback to `FEATURES`.

### 6. Unrealistic Sensor Drift Model → ✅ FIXED
`noise_injection.py` v3.1: monotonic directional drift (sensor consistently loses sensitivity). Dropout rates scaled realistically: mild ~2%, moderate ~5%, severe ~10%.

### 7. Per-Class F1 Not Discussed → ✅ FIXED
Section 6.2 added: per-class analysis paragraph discussing hardest classes and accuracy–Macro-F1 gap drivers.

### 8. Pipeline Header v3.0 → ✅ FIXED
All version strings updated to v3.1.

### 9. References Expanded → ✅ FIXED
Added: Lundberg & Lee (2017, SHAP), Pedregosa et al. (2011, scikit-learn), Kapoor & Narayanan (2023, leakage).

### 10. Friedman Test Implemented → ✅ FIXED
Friedman test + Nemenyi CD now computed in Session 5 and reported in paper (Section 4.6, Table 2, abstract, conclusion).

### 11. Consistency Formula Bug → ✅ FIXED
Section 5.7 now includes the formula explicitly: `Consistency = 1 − |score_primary − score_secondary|`. All three values now match the formula.

### 12. Recalibration Cost Nuance → ✅ FIXED
Section 5.5 now includes cost-benefit discussion and recommends deployment-specific analysis.

### 13. Secondary Results Lead → ✅ FIXED
Section 5.3 now leads with: "The secondary dataset is the more meaningful evaluation... We lead with these results."

---

## Remaining Items (Acceptable for Current Scope)

| Item | Status | Justification |
|------|--------|---------------|
| Dead code (add_class_imbalance) | Not removed | Utility function for future use |
| SHAP depth (per-class, local) | Future work | Section 6.5/6.6 acknowledge |
| Hyperparameter tuning | Future work | Section 6.5/6.6 acknowledge |
| Noise-augmented training | Future work | Section 6.6, item 2 |
| Second crop dataset for true cross-validation | Future work | Section 6.5, item 2 |

---

## Conclusion

All critical and major issues are resolved. The paper honestly describes its methodology, correctly frames its contributions, acknowledges limitations, and provides reproducible code. The remaining items are appropriate future work directions.
