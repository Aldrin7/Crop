# NEXT SESSION TASKLIST — v3.0 (Dual Dataset)

## What Changed (v2 → v3)
- ✅ **Real secondary dataset integrated** (Soil Fertility, 880 samples, real lab measurements)
- ✅ **Cross-dataset validation** on shared features (N, P, K, pH)
- ✅ **README restructured** — SCI/Scopus journals only
- ✅ **Pipeline restructured** — both datasets processed in parallel
- ✅ **Target journals** — no predatory, only SCI/Scopus indexed

## Completed (Sessions 1-2 from v2) ✅
- [x] Session 1: Data acquisition, EDA (primary dataset)
- [x] Session 2: Preprocessing, descriptive FS (primary dataset)

## Pending — Run in Order

### Step 1: Clear old checkpoints (fresh start for v3)
```bash
cd Crop-Research
rm -f data/checkpoints/*.pkl data/checkpoints/*.done
rm -rf results/figures/* results/tables/* results/metrics/*
```

### Step 2: Run Session 1 (Data & EDA — BOTH datasets) — ~15 min
```bash
python3 pipeline.py --session 1
```
- Downloads/loads both primary + secondary datasets
- EDA for both: distributions, correlations, class balance
- Shared feature space analysis
- Degradation variants
- **Outputs:** 7 figures, descriptive stats tables
- **Verify:** `results/figures/07_shared_features_comparison.png` exists

### Step 3: Run Session 2 (Preprocessing + FS) — ~15 min
```bash
python3 pipeline.py --session 2
```
- Preprocesses both datasets (missing handling for secondary)
- 6 FS methods on each dataset + consensus ranking
- Cross-dataset FS consistency analysis
- **Outputs:** `08_feature_selection.png`, `09_feature_selection_secondary.png`, FS tables
- **Verify:** `results/tables/cross_dataset_fs_consistency.csv` exists

### Step 4: Run Session 3 (Nested CV Training) — ~25 min
```bash
python3 pipeline.py --session 3
```
- Leak-free nested CV on primary (all_7, top_5, top_4, top_3)
- Leak-free nested CV on secondary (sec_all_12, sec_top_6, sec_top_4)
- 10 classifiers × 7 feature subsets = ~70 training runs
- **Checkpoint:** `data/checkpoints/s3.pkl`
- **Verify:** `results/tables/nested_cv_results.csv` with ~60+ rows

### Step 5: Run Session 4 (SHAP + Calibration + Cross-Dataset) — ~15 min
```bash
python3 pipeline.py --session 4
```
- SHAP TreeExplainer for top 3 classifiers
- GaussianNB calibration analysis
- Robustness under sensor degradation
- Calibration curves
- Per-class F1 heatmap
- Cross-dataset feature consistency validation
- **Outputs:** `11_shap_*.png`, `12_robustness.png`, `13_calibration.png`, `14_per_class_heatmap.png`

### Step 6: Run Session 5 (Paper Artifacts) — ~5 min
```bash
python3 pipeline.py --session 5
```
- Master results table (LaTeX-ready)
- Final summary JSON
- **Verify:** `results/tables/master_results.csv` and `results/metrics/final_summary.json`

### Step 7: Push to GitHub
```bash
cd Crop-Research
git add -A
git commit -m "v3.0: Dual-dataset pipeline with real secondary + SCI/Scopus journals"
git push origin main
```

### Step 8 (Optional): Write Paper Draft
- Use `results/metrics/final_summary.json` for abstract/methods
- Use `results/tables/master_results.csv` for results
- Figures 01-14 for illustrations
- Cross-dataset validation strengthens generalisation claims
- **Target:** Computers and Electronics in Agriculture (SCI, IF 8.3)

---

## Key Advantages of v3.0

| Aspect | v2.0 | v3.0 |
|--------|------|------|
| Datasets | 1 (semi-synthetic) | 2 (semi-synthetic + **real**) |
| Validation | Single-dataset CV | **Cross-dataset** generalisation |
| Real-world | Sensor noise only | Real lab data + noise |
| Class balance | Perfect (synthetic) | **Natural imbalance** (real) |
| Missing values | Injected | **Actual** lab dropout |
| Journal targets | Mixed | **SCI/Scopus only** |

## If Session 3 Hangs
- Delete `data/checkpoints/s3.pkl` and re-run
- Secondary dataset training is faster (fewer classes)
- Primary ~15 min, Secondary ~10 min

## If SHAP Fails (Multi-class)
Fix in `src/explainability.py` — handle `shap_values` as list:
```python
mean_shap = np.mean([np.abs(sv).mean(axis=0) for sv in shap_vals], axis=0)
```
