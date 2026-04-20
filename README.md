# Crop-Research — Q1 Journal Pipeline v3.0

**Dual-Dataset Design with Cross-Dataset Validation for Crop Recommendation**

## Research Paper

**Title:** *A Comparative Study of Feature Selection Algorithms and Classification Methods for Crop Recommendation Using Integrated Soil Nutrient and Climate Data*

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

### Secondary — Soil Fertility (REAL Lab Measurements) ✅ NEW
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
│   ├── feature_selection.py     # 6 FS methods (leak-free, inside CV)
│   ├── models.py                # 10 classifiers
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
│   ├── figures/                 # Publication-quality PNG + PDF
│   ├── tables/                  # CSV + LaTeX tables
│   └── metrics/                 # JSON summaries
├── models/                      # Trained model artifacts
├── logs/
├── requirements.txt
└── README.md
```

## Usage

```bash
cd Crop-Research
python3 pipeline.py --session 1   # Data acquisition + EDA (both datasets)
python3 pipeline.py --session 2   # Preprocessing + descriptive FS
python3 pipeline.py --session 3   # Nested CV training (leak-free)
python3 pipeline.py --session 4   # Evaluation + SHAP + calibration + cross-dataset
python3 pipeline.py --session 5   # Paper artifacts
python3 pipeline.py --all         # Run everything
```

## Peer Review Fixes (v3.0)

| Critique | Fix | Status |
|----------|-----|--------|
| 2.1 Semi-synthetic dataset | Acknowledged + **real secondary** + degradation variants | ✅ |
| 2.2 FS data leakage | FS inside CV loop via Pipeline | ✅ |
| 2.3 Sensor degradation | Literature-grounded (Rana 2019, Lobnik 2011) | ✅ |
| 3.1 Monolithic code | Modular `src/*.py` | ✅ |
| 3.2 Redundant metrics | Kappa, MCC, Brier, ECE | ✅ |
| 3.3 Interpretability | SHAP + NB calibration analysis | ✅ |
| 4.1 **NEW** Generalisation | Cross-dataset validation on real secondary | ✅ |

## Feature Selection Methods (6 + Consensus)
1. Mutual Information
2. Chi-Square Test
3. Recursive Feature Elimination (RFE)
4. LASSO Regularization
5. Extra Trees Importance
6. Random Forest Importance
7. **Consensus Ranking** (normalized mean across all methods)

## Classifiers (10)
Random Forest · SVM-RBF · KNN · Decision Tree · Gradient Boosting · XGBoost · LightGBM · Logistic Regression · MLP · GaussianNB

---

## Target Journals (SCI / Scopus Indexed Only)

### Tier 1 — SCI Indexed (High Impact)
| Journal | Publisher | IF | APC | Review | Scope |
|---------|-----------|-----|-----|--------|-------|
| **Computers and Electronics in Agriculture** | Elsevier | 8.3 | ~₹30,000 | 2-4 months | Agricultural ML, precision farming |
| **Computers in Biology and Medicine** | Elsevier | 7.7 | ~₹25,000 | 2-3 months | Applied ML in bio/agri |
| **Expert Systems with Applications** | Elsevier | 8.5 | ~₹28,000 | 2-3 months | ML applications |
| **Information Processing & Management** | Elsevier | 8.6 | ~₹26,000 | 2-4 months | Data processing, ML |
| **Knowledge-Based Systems** | Elsevier | 8.8 | ~₹28,000 | 2-4 months | AI/ML methods |

### Tier 2 — SCI Indexed (Moderate Impact, Faster)
| Journal | Publisher | APC | Review | Notes |
|---------|-----------|-----|--------|-------|
| **IEEE Access** | IEEE | ~₹15,000 (Gold OA) | 4-6 weeks | Very fast, SCI indexed |
| **PLOS ONE** | PLOS | ~₹20,000 | 2-3 months | Multidisciplinary, high acceptance |
| **Heliyon** | Elsevier/Cell Press | Free APC | 1-2 months | SCI indexed, fast review |
| **PeerJ Computer Science** | PeerJ | ~₹14,000 | 1-2 months | Open access, fast |
| **Results in Engineering** | Elsevier | Free APC | 4-8 weeks | Engineering focus |

### Tier 3 — Scopus Indexed (Fast, Low Cost)
| Journal | Publisher | APC | Review |
|---------|-----------|-----|--------|
| **Array** | Elsevier | Free | 4-6 weeks |
| **Engineering Applications of AI** | Elsevier | ~₹22,000 | 2-3 months |
| **Applied Soft Computing** | Elsevier | ~₹25,000 | 2-3 months |
| **Neural Computing and Applications** | Springer | ~₹20,000 | 2-4 months |
| **Intelligent Systems with Applications** | Elsevier | Free | 4-8 weeks |
| **Decision Analytics** | Springer | ~₹12,000 | 1-2 months |

### 🏆 Recommended Submission Strategy
1. **Primary target:** Computers and Electronics in Agriculture (best fit for agricultural ML)
2. **Fast backup:** IEEE Access or Heliyon (if speed is critical)
3. **Budget option:** Heliyon (free APC, SCI indexed, 1-2 month review)

---

## System Requirements
- Python 3.10+
- RAM: 3GB minimum
- Storage: ~500MB
- Dependencies: `pip install -r requirements.txt`

## License
Research use — cite the original dataset sources.
