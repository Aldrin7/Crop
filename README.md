# RobustCrop: A Leak-Free Machine Learning Pipeline for Crop Recommendation

**Dual-Dataset Evaluation with Sensor Degradation Analysis and Cross-Dataset Feature Consistency**

> **v3.1** — Leak-free `Pipeline(StandardScaler → SelectKBest(mutual_info) → Classifier)` per CV fold. `class_weight='balanced'` on RF/SVM/DT/LR/LightGBM. All results computed with corrected methodology and verified via Friedman test.

**Authors:** Anuradha Brijwal¹, Praveena Chaturvedi²  
¹ Research Scholar, Department of Computer Science, Kanya Gurukul Campus Dehradun  
² Professor, Department of Computer Science, Kanya Gurukul Campus Dehradun  
Gurukul Kangri (Deemed to be University), Haridwar, Uttarakhand, India

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
├── Session 3: Leak-free CV training (Pipeline per fold)
├── Session 4: Evaluation + SHAP + calibration + robustness
└── Session 5: Paper artifacts + Friedman test

src/
├── config.py              # Paths, hyperparams, sensor specs
├── data_loader.py         # Dataset loading
├── preprocessing.py       # Scaling, imputation, outlier detection
├── feature_selection.py   # 6 FS methods + Pipeline-compatible selectors
├── models.py              # 10 classifiers (class_weight='balanced')
├── evaluation.py          # Kappa, MCC, Brier, ECE, Friedman test
├── explainability.py      # SHAP + GaussianNB calibration
├── noise_injection.py     # Monotonic sensor degradation simulation
└── utils.py               # Checkpointing, logging, figures
```

## Usage

```bash
git clone https://github.com/Aldrin7/Crop.git
cd Crop
pip install -r requirements.txt
python3 pipeline.py --all         # Run everything
python3 pipeline.py --session 3   # Or run individual sessions
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

### Class Imbalance Handling
`class_weight='balanced'` applied to RF, SVM, DT, LR, LightGBM.

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

Research use — cite the original dataset sources.
