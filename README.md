# RobustCrop — Leak-Free Pipeline v3.1

**Dual-Dataset Evaluation with Cross-Dataset Feature Consistency Analysis**

> **v3.1 (2026-04-27):** Leak-free `Pipeline(StandardScaler → SelectKBest(mutual_info) → Classifier)` per CV fold. `class_weight=.balanced.` on RF/SVM/DT/LR/LightGBM. All results re-run with corrected methodology.

## Research Paper

**Title:** *RobustCrop: A Leak-Free Machine Learning Pipeline for Crop Recommendation with Sensor Degradation Analysis and Cross-Dataset Feature Consistency*

### Objectives
1. Comparative study of crop classification by integrating soil nutrients and climate conditions
2. Analysis of effective data handling methods for the integrated dataset
3. Identification of relevant feature selection algorithms for classification

---

## Datasets

### Primary — Crop Recommendation (Semi-Synthetic)
| Property | Value |
|----------|-------|
| Source | [Kaggle — Atharva Ingle](https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset) |
| Samples | 2,200 |
| Features | N, P, K, Temperature, Humidity, pH, Rainfall (7) |
| Classes | 22 crop types (100 samples each — perfectly balanced) |
| Nature | Semi-synthetic (augmented from Indian agricultural statistics) |

### Secondary — Soil Fertility (REAL Lab Measurements)
| Property | Value |
|----------|-------|
| Source | [Kaggle — Rahul Jaiswal](https://www.kaggle.com/datasets/rahuljaiswalonkaggle/soil-fertility-dataset) |
| Samples | 880 |
| Features | N, P, K, pH, EC, OC, S, Zn, Fe, Cu, Mn, B (12) |
| Classes | 3 fertility levels (0=High, 1=Medium, 2=Low) |
| Nature | **Real soil lab test results** from Indian agricultural testing centres |
| Imbalance | 401 / 440 / 39 (natural real-world distribution) |
| Missing | ~3% (real sensor/lab dropout) |

### Shared Feature Space
Both datasets share **N, P, K, pH** — enabling cross-dataset validation of feature selection consistency.

### Degradation Variants (Primary)
Literature-grounded sensor drift simulation (Rana et al. 2019, Lobnik et al. 2011):
- **Mild** (7-day), **Moderate** (30-day), **Severe** (90-day) deployments

---

## Pipeline Architecture

```
Crop-Research/
├── pipeline.py                  # Orchestrator (5 sessions)
├── src/
│   ├── config.py                # Paths, hyperparams, dual-dataset config
│   ├── data_loader.py           # Primary + Secondary dataset loading
│   ├── preprocessing.py         # Scaling, encoding, outlier detection, missing handling
│   ├── feature_selection.py     # 6 FS methods + Pipeline-compatible selectors
│   ├── models.py                # 10 classifiers (with class_weight='balanced')
│   ├── evaluation.py            # Kappa, MCC, Brier, ECE, Friedman test
│   ├── explainability.py        # SHAP + GaussianNB calibration
│   ├── noise_injection.py       # Sensor degradation simulation
│   └── utils.py                 # Checkpointing, logging, figures
├── data/
│   ├── raw/
│   │   ├── Crop_recommendation.csv      # Primary
│   │   └── soil_fertility_secondary.csv  # Secondary (REAL)
│   ├── processed/
│   └── checkpoints/
├── results/
│   ├── figures/                 # Publication-quality PNG + PDF (30 files)
│   ├── tables/                  # CSV + LaTeX tables (21 files)
│   └── metrics/                 # JSON summaries (4 files)
├── models/                      # Trained model artifacts (Pipeline objects)
├── logs/
├── requirements.txt
└── README.md
```

## Usage

```bash
cd Crop-Research
python3 pipeline.py --session 1   # Data acquisition + EDA (both datasets)
python3 pipeline.py --session 2   # Preprocessing + descriptive FS
python3 pipeline.py --session 3   # Leak-free CV training (Pipeline per fold)
python3 pipeline.py --session 4   # Evaluation + SHAP + calibration + cross-dataset
python3 pipeline.py --session 5   # Paper artifacts
python3 pipeline.py --all         # Run everything
```

## Key Results (v3.1 — Leak-Free)

### Primary Dataset (22 crop classes)

| Classifier | Accuracy (all_7) | κ | MCC | Macro-F1 | ECE |
|-----------|------------------|------|------|----------|------|
| **Random Forest** | **0.9950 ± 0.0009** | **0.9948** | **0.9948** | **0.9950** | 0.0430 |
| GaussianNB | 0.9945 ± 0.0023 | 0.9943 | 0.9943 | 0.9945 | 0.0069 |
| LightGBM | 0.9918 ± 0.0053 | 0.9914 | 0.9914 | 0.9918 | 0.0068 |
| XGBoost | 0.9909 ± 0.0032 | 0.9905 | 0.9905 | 0.9909 | 0.0136 |

### Secondary Dataset (Real Soil Fertility — 3 classes, 11.28× imbalance)

| Classifier | Accuracy (sec_mi_top_6) | κ | MCC | Macro-F1 | ECE |
|-----------|------------------------|------|------|----------|------|
| **Random Forest** | **0.9125 ± 0.0077** | **0.8364** | **0.8371** | **0.8185** | 0.0562 |
| XGBoost | 0.9034 ± 0.0200 | 0.8200 | 0.8205 | 0.7932 | 0.0755 |
| LightGBM | 0.9034 ± 0.0183 | 0.8188 | 0.8197 | 0.7727 | 0.0849 |
| GradientBoosting | 0.8932 ± 0.0174 | 0.8002 | 0.8007 | 0.7659 | 0.0954 |

### Feature Ablation (Primary — MI per fold)

| Subset | Features | Best Classifier | Accuracy | Drop vs all_7 |
|--------|----------|----------------|----------|--------------|
| all_7 | All 7 | RF | 0.9950 | — |
| mi_top_5 | MI top-5 per fold | RF | 0.9905 | -0.45% |
| mi_top_4 | MI top-4 per fold | RF | 0.9782 | -1.68% |
| mi_top_3 | MI top-3 per fold | RF | 0.9645 | -3.05% |

### Sensor Degradation Robustness

| Scenario | Deployment | Accuracy | κ | Brier |
|----------|-----------|----------|------|-------|
| Fresh | 0 days | 0.9950 | 0.9948 | 0.0007 |
| Mild | 7 days | 0.9664 | 0.9648 | 0.0039 |
| Moderate | 30 days | 0.8177 | 0.8090 | 0.0138 |
| Severe | 90 days | 0.4382 | 0.4114 | 0.0331 |

---

## Methodology (v3.1)

### Leak-Free Cross-Validation
Each classifier is wrapped in an sklearn `Pipeline`:
```
Pipeline(StandardScaler → SelectKBest(mutual_info, k=N) → Classifier)
```
- **Scaler** is fit on each training fold only (no global scaling leakage)
- **Feature selection** is performed per fold using Mutual Information (no global pre-selection leakage)
- **Ablation** tests different k values (7, 5, 4, 3) with MI re-ranking each fold
- Outer 5-fold StratifiedKFold for unbiased evaluation

### Class Imbalance Handling
- `class_weight='balanced'` applied to: Random Forest, SVM-RBF, Decision Tree, Logistic Regression
- Automatically adjusts class weights inversely proportional to class frequency

### Feature Selection Methods (6 + Consensus — Descriptive Only)
1. Mutual Information
2. Chi-Square Test
3. Recursive Feature Elimination (RFE)
4. LASSO Regularization
5. Extra Trees Importance
6. Random Forest Importance
7. **Consensus Ranking** (normalized mean across all methods)

> **Note:** Consensus rankings are used for descriptive analysis only. The actual CV training uses per-fold MI selection to prevent data leakage.

### Classifiers (10)
Random Forest · SVM-RBF · KNN · Decision Tree · Gradient Boosting · XGBoost · LightGBM · Logistic Regression · MLP · GaussianNB

---

## Peer Review Fixes (v3.0 → v3.1)

| Critique | Fix | Status |
|----------|-----|--------|
| 2.1 Semi-synthetic dataset | Acknowledged + real secondary + degradation variants | ✅ |
| 2.2 FS data leakage | **v3.1:** Pipeline with SelectKBest per CV fold | ✅ |
| 2.3 Sensor degradation | Literature-grounded (Rana 2019, Lobnik 2011) | ✅ |
| 3.1 Monolithic code | Modular `src/*.py` | ✅ |
| 3.2 Redundant metrics | Kappa, MCC, Brier, ECE | ✅ |
| 3.3 Interpretability | SHAP + NB calibration analysis | ✅ |
| 3.4 Class imbalance | **v3.1:** `class_weight=.balanced.` on RF/SVM/DT/LR/LightGBM | ✅ |
| 4.1 Generalisation | Cross-dataset validation on real secondary | ✅ |
| **NEW** Scaling leakage | **v3.1:** Scaler fit per fold via Pipeline | ✅ |

---

## Target Journals (Q2 — SCI / Scopus Indexed)

| Journal | Publisher | IF | APC | Review | Notes |
|---------|-----------|-----|-----|--------|-------|
| **Heliyon** | Elsevier/Cell Press | ~4.0 | Free | 1-2 months | SCI indexed, fast, free |
| **PeerJ Computer Science** | PeerJ | ~3.5 | ~₹14,000 | 1-2 months | Open access, fast |
| **IEEE Access** | IEEE | ~3.9 | ~₹15,000 | 4-6 weeks | Gold OA, very fast |
| **PLOS ONE** | PLOS | ~3.7 | ~₹20,000 | 2-3 months | Multidisciplinary |
| **Array** | Elsevier | — | Free | 4-6 weeks | Scopus, free |

### 🏆 Recommended Submission Strategy
1. **Primary target:** Heliyon (free APC, SCI indexed, fast review)
2. **Fast backup:** IEEE Access (4-6 weeks, gold OA)
3. **Budget option:** PeerJ Computer Science (open access)

---

## System Requirements
- Python 3.10+
- RAM: 3GB minimum
- Storage: ~500MB
- Dependencies: `pip install -r requirements.txt`

## License
Research use — cite the original dataset sources.
