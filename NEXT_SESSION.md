# NEXT SESSION TASKLIST

## Completed (Sessions 1-2) ✅
- [x] Session 1: Data acquisition, EDA, 6 figures, 3 degradation variants (mild/moderate/severe)
- [x] Session 2: Preprocessing, outlier detection, descriptive FS (6 methods + consensus)
- [x] Partial Session 3: Nested CV started (all_7 subset mostly done, top_5 in progress)

## Pending — Run in Order

### Step 1: Finish Session 3 (Nested CV Training) — ~20 min
```bash
cd Crop-Research-v2
python3 pipeline.py --session 3
```
- Runs leak-free nested CV (5-fold outer × 8-10 classifiers × 4 feature subsets)
- Checkpoint saved as `data/checkpoints/s3.pkl`
- If it hangs: delete `data/checkpoints/s3.pkl` and re-run
- **Verify:** check `results/tables/nested_cv_results.csv` exists with ~32-40 rows

### Step 2: Run Session 4 (SHAP + Calibration + Robustness) — ~15 min
```bash
python3 pipeline.py --session 4
```
- SHAP TreeExplainer for top 3 classifiers
- GaussianNB calibration analysis (Brier score, independence violations)
- Robustness under sensor degradation (fresh/mild/moderate/severe)
- Calibration curves for top 3
- Per-class F1 heatmap across all classifiers
- **Outputs:** `08_shap_*.png`, `09_robustness.png`, `10_calibration.png`, `11_per_class_heatmap.png`

### Step 3: Run Session 5 (Paper Artifacts) — ~5 min
```bash
python3 pipeline.py --session 5
```
- Master results table (LaTeX-ready)
- Ablation study figure
- Consensus feature ranking table
- Final summary JSON
- **Verify:** check `results/tables/master_results.csv` and `results/metrics/final_summary.json`

### Step 4: Push Results to GitHub
```bash
cd Crop-Research-v2
git add -A && git commit -m "Sessions 3-5 complete — nested CV, SHAP, calibration, paper artifacts"
git push origin main
```

### Step 5 (Optional): Fix Multi-Class SHAP Plot
- Session 4 SHAP may fail on multi-class models (list of arrays vs single array)
- If so, fix in `src/explainability.py` line ~40 — handle `shap_values` as list properly
- Quick fix: `mean_shap = np.mean([np.abs(sv).mean(axis=0) for sv in shap_vals], axis=0)`

### Step 6 (Optional): Write Paper Draft
- Use `results/metrics/final_summary.json` for abstract/methods
- Use `results/tables/master_results.csv` for results section
- Use figures 01-11 for all illustrations
- Target: Heliyon (free APC, SCI indexed)

## Key Peer Review Fixes Already Implemented
| Critique | Fix | Status |
|----------|-----|--------|
| 2.1 Semi-synthetic dataset | Acknowledged + degradation variants | ✅ |
| 2.2 FS data leakage | FS inside CV loop | ✅ |
| 2.3 Sensor degradation | Literature-grounded (Rana 2019, Lobnik 2011) | ✅ |
| 3.1 Monolithic code | Modular src/*.py | ✅ |
| 3.2 Redundant metrics | Kappa, MCC, Brier, ECE | ✅ |
| 3.3 Interpretability | SHAP + NB calibration analysis | ⏳ (Session 4) |
