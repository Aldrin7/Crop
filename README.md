# RobustCrop: A Leak-Free Machine Learning Pipeline for Crop Recommendation

**Dual-Dataset Evaluation with Sensor Degradation Analysis and Cross-Dataset Feature Consistency**

> **v3.2** — Leak-free `Pipeline(StandardScaler → SelectKBest(mutual_info) → Classifier)` per CV fold. `class_weight='balanced'` or `BalWeightWrapper` (sample_weight) on ALL classifiers. Optuna nested CV tuning available via `--tune`. All results computed with corrected methodology and verified via Friedman test.

**Authors:** Anuradha Brijwal¹, Praveena Chaturvedi²  
¹ Research Scholar, Department of Computer Science, Kanya Gurukul Campus Dehradun  
² Professor, Department of Computer Science, Kanya Gurukul Campus Dehradun  
Gurukul Kangri (Deemed to be University), Haridwar, Uttarakhand, India

---

## What's New in v3.2

| Change | Details |
|--------|---------|
| **BalWeightWrapper** | All classifiers now receive fair imbalance handling via `class_weight='balanced'` or `sample_weight='balanced'` wrapper |
| **Optuna nested CV** | `--tune` flag enables hyperparameter tuning via Optuna (30 trials, 3-fold inner CV per outer fold) |
| **Dead code removed** | `add_class_imbalance()` removed from `noise_injection.py` |
| **Deprecation warnings** | `scale_features()` and `run_all_fs_methods()` now warn about proper usage |
| **Missing deps fixed** | `shap` and `optuna` added to `requirements.txt` |
| **Tests added** | `tests/test_pipeline.py` — preprocessing, models, FS, evaluation, noise, integration |
| **MIT License** | Added explicit MIT license |
| **`.gitignore` updated** | `.docx` excluded, Optuna DB patterns added |

---

## Key Results

| Metric | Primary (22 crops) | Secondary (real soil, 3 classes) |
|--------|-------------------|--------------------------------|
| Best Classifier | Random Forest | Random Forest |
| Accuracy | 99.50% ± 0.09% | 91.25% ± 0.77% |
| Cohen's κ | 0.9948 | 0.8364 |
| Macro-F1 | 0.9950 | 0.8185 |
| Friedman Test | χ² = 32.32, p < 0.001 | — |

### Sensor Degradation (Monotonic Drift)

| Deployment | Accuracy | Δ vs Fresh |
|-----------|----------|-----------|
| Fresh | 99.50% | — |
| 7 days | 94.05% | −5.45% |
| 30 days | 70.41% | −29.09% |
| 90 days | 16.09% | −83.41% |

### Cross-Dataset Feature Consistency

| Feature | Consistency | Interpretation |
|---------|------------|----------------|
| P | 0.804 | Most transferable across domains |
| N | 0.367 | Task-dependent importance |
| K | 0.293 | Least transferable |

---

## Datasets

### Primary — Crop Recommendation (Semi-Synthetic)
- **Source:** [Kaggle — Atharva Ingle](https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset)
- **Samples:** 2,200 | **Features:** 7 (N, P, K, Temperature, Humidity, pH, Rainfall)
- **Classes:** 22 crop types (100 samples each — perfectly balanced)
- **Nature:** Semi-synthetic (augmented from Indian agricultural statistics)

### Secondary — Soil Fertility (Real Lab Measurements)
- **Source:** [Kaggle — Rahul Jaiswal](https://www.kaggle.com/datasets/rahuljaiswalonkaggle/soil-fertility-dataset)
- **Samples:** 880 | **Features:** 12 (N, P, K, pH, EC, OC, S, Zn, Fe, Cu, Mn, B)
- **Classes:** 3 fertility levels (401/440/39 — natural imbalance 11.28:1)
- **Nature:** **Real soil lab test results** from Indian agricultural testing centres

---

## Pipeline Architecture

```
pipeline.py — Orchestrator (5 sessions)
├── Session 1: Data acquisition + EDA (both datasets)
├── Session 2: Preprocessing + descriptive feature selection
├── Session 3: Leak-free CV training (Pipeline per fold, optional --tune)
├── Session 4: Evaluation + SHAP + calibration + robustness
└── Session 5: Paper artifacts + Friedman test

src/
├── config.py              # Paths, hyperparams, sensor specs
├── data_loader.py         # Dataset loading
├── preprocessing.py       # Scaling, imputation, outlier detection
├── feature_selection.py   # 6 FS methods + Pipeline-compatible selectors
├── models.py              # 10 classifiers + BalWeightWrapper for all
├── evaluation.py          # Kappa, MCC, Brier, ECE, Friedman test
├── explainability.py      # SHAP + GaussianNB calibration
├── noise_injection.py     # Monotonic sensor degradation simulation
└── utils.py               # Checkpointing, logging, figures

tests/
└── test_pipeline.py       # Unit + integration tests
```

## Usage

```bash
git clone https://github.com/Aldrin7/Crop.git
cd Crop
pip install -r requirements.txt

# Run everything (default hyperparameters)
python3 pipeline.py --all

# Run with Optuna hyperparameter tuning (nested CV)
python3 pipeline.py --all --tune

# Run individual sessions
python3 pipeline.py --session 3
python3 pipeline.py --session 3 --tune

# Run tests
python -m pytest tests/ -v
```

---

## Methodology

### Leak-Free Pipeline
Each classifier is wrapped in a scikit-learn Pipeline per CV fold:
```
Pipeline(StandardScaler → SelectKBest(mutual_info, k=N) → Classifier)
```
- Scaler fitted on training fold only (no leakage)
- Feature selection per fold via mutual information (no global pre-selection)
- 5-fold stratified CV for unbiased evaluation
- **NEW:** Optional Optuna inner CV (3-fold, 30 trials) for hyperparameter tuning

### Class Imbalance Handling (ALL Classifiers)
- `class_weight='balanced'` natively supported: RF, SVM, DT, LR, LightGBM
- `BalWeightWrapper` (sample_weight) for: XGBoost, GB, MLP, KNN, GaussianNB
- **Every classifier receives balanced class weights** — no unfair advantage

### Feature Selection (Descriptive — 6 Methods + Consensus)
Mutual Information · Chi-Square · RFE · LASSO · Extra Trees · Random Forest Importance  
Consensus ranking = normalized mean across all methods. Used for interpretation only; training uses per-fold MI.

### Classifiers
Random Forest (proposed) · SVM-RBF · KNN · Decision Tree · Gradient Boosting · XGBoost · LightGBM · Logistic Regression · MLP · GaussianNB

---

## Output

| Type | Count | Location |
|------|-------|----------|
| Figures | 32 | `results/figures/` (PNG + PDF) |
| Tables | 23 | `results/tables/` (CSV + LaTeX) |
| Metrics | 5 | `results/metrics/` (JSON) |
| Models | — | `models/` (Pipeline objects) |

---

## Requirements

- Python 3.10+
- RAM: 3GB minimum
- `pip install -r requirements.txt`

## License

MIT License — see [LICENSE](LICENSE).

## Cite

If you use this code, cite the original dataset sources:
- Primary: Atharva Ingle, Crop Recommendation Dataset (Kaggle)
- Secondary: Rahul Jaiswal, Soil Fertility Dataset (Kaggle)
