# Robustness-Aware Crop Recommendation Using Soil–Climate Data
## A Comparative Study of Feature Selection and Classification Methods

---

## Research Paper Details

### Title
**"Robustness-Aware Crop Recommendation Using Soil-Climate Data: A Comparative Study of Feature Selection and Classification Methods"**

### Central Claim
> **Model robustness, not accuracy, is the limiting factor in real-world crop recommendation systems.**

### Abstract
Machine learning for crop recommendation must contend with real-world uncertainty — sensor noise, missing data, and environmental variability — yet the vast majority of studies evaluate models only under ideal laboratory conditions. This gap between evaluation and deployment limits the practical utility of reported results. We present a robustness-aware evaluation framework for crop recommendation that integrates feature selection comparison, model interpretability, and systematic stress testing on soil nutrient (N, P, K) and climate (temperature, humidity, rainfall, pH) data comprising 22 crop classes. We evaluate five feature selection algorithms and ten classifiers across four feature subsets. Under ideal conditions, all classifiers achieve >99% accuracy, reflecting the dataset's high separability. However, robustness testing reveals a non-linear performance degradation: accuracy drops to 56.8% under Gaussian noise (σ=0.5) and to 47.5% under 50% missing data, with performance remaining stable only under low perturbations (σ ≤ 0.1). SHAP analysis shows moderate agreement with statistical feature selection (Spearman ρ=0.679), suggesting the two paradigms capture partially distinct aspects of feature relevance. Error analysis identifies crop pairs with shared agro-climatic profiles (e.g., rice and jute) as primary confusion sources, pointing to the need for additional discriminative features. Despite statistically significant differences across classifiers (Friedman p<0.01), practical performance differences remain below 3%, and no classifier family demonstrates a decisive advantage — reinforcing that dataset simplicity, not model complexity, drives performance. These findings reframe the research contribution from accuracy benchmarking to robustness-aware evaluation, supporting the use of computationally efficient models in resource-constrained agricultural settings.

### Keywords
Crop Recommendation, Feature Selection, Soil Nutrients, Climate Data, Precision Agriculture, Machine Learning, Classification, Robustness, SHAP, Interpretability

---

## Key Results

### Classification Performance (Clean Data)
| Metric | Best Classifier | Score |
|--------|----------------|-------|
| Test Accuracy | RandomForest | **99.55%** |
| F1-Score | RandomForest | **0.9955** |
| 5-Fold CV | RandomForest | 99.43% ± 0.48% |
| 10-Fold CV | GaussianNB | 99.49% ± 0.59% |

### Feature Selection Consensus Ranking
1. **Rainfall** (0.875) — primary limiting factor for crop growth
2. **K / Potassium** (0.853) — critical for fruit/grain crops (5-10× variation)
3. **Humidity** (0.838) — separates tropical vs arid-adapted crops
4. **P / Phosphorus** (0.646) — root development and energy transfer
5. **N / Nitrogen** (0.527) — vegetative growth driver
6. **Temperature** (0.308) — moderate discrimination (narrow range)
7. **pH** (0.190) — weakest predictor (Indian soils uniformly neutral)

### Robustness Analysis (Where the Model Breaks)
| Test | Condition | Accuracy | Drop |
|------|-----------|----------|------|
| Baseline | Clean data | 99.55% | — |
| Noise σ=0.1 | Light perturbation | 97.50% | -2.0% |
| Noise σ=0.5 | Moderate noise | 56.82% | -42.7% |
| Noise σ=1.0 | Heavy noise | 28.64% | -70.9% |
| 10% missing | Median impute | 89.32% | -10.2% |
| 30% missing | Median impute | 69.32% | -30.2% |
| 50% missing | Median impute | 47.50% | -52.1% |
| Drop humidity | Zero-fill | 79.32% | -20.2% |
| Drop rainfall | Zero-fill | 82.73% | -16.8% |

**Pattern**: Performance remains relatively stable under low perturbations (σ ≤ 0.1), but degrades sharply beyond moderate noise levels (σ ≥ 0.5). The transition from acceptable to unacceptable performance is rapid, not gradual.

### SHAP vs Feature Selection Agreement
| Metric | Value | p-value |
|--------|-------|---------|
| Spearman ρ | 0.679 | 0.094 |
| Kendall τ | 0.524 | 0.136 |

- **Moderate agreement** — statistical and model-driven importance capture partially distinct aspects of feature relevance; neither alone is sufficient
- K, temperature, and pH: perfect rank alignment (diff = 0)
- Rainfall: largest disagreement (SHAP rank 4 vs FS rank 1)

### Statistical Significance
- **Friedman test**: χ² = 59.74, p < 0.001 (significant)
- **Max practical difference**: 2.67% (GaussianNB vs MLP)
- **Nemenyi CD** (α=0.05): 4.284
- **Conclusion**: Despite statistical significance, practical differences are negligible — supporting the use of computationally efficient models in resource-constrained agricultural settings. The narrow performance band across diverse classifier families further reinforces that dataset simplicity, not model complexity, is the primary driver of high accuracy.

### Classifier Family Analysis
| Family | Classifiers | Mean Accuracy | Takeaway |
|--------|-------------|---------------|----------|
| Neural | MLP | 0.9909 | Best single model |
| Tree-Based | RF, XGB, LGBM, DT, GB | 0.9882 ± 0.004 | Most consistent |
| Probabilistic | GNB, LR | 0.9841 ± 0.011 | **Best efficiency** |
| Distance-Based | KNN | 0.9818 | Simplest |

**Takeaway**: No classifier family demonstrates a decisive advantage — Neural (0.9909), Tree-Based (0.9882), and Probabilistic (0.9841) are all within 0.7%. This reinforces that dataset simplicity, not model complexity, drives performance. For resource-constrained deployment (edge devices, IoT), GaussianNB offers near-optimal accuracy at minimal computational cost.

### Per-Crop Error Analysis
- **Worst crops**: blackgram (F1=0.974), rice (F1=0.974)
- **Key confusion**: Rice ↔ jute (shared high rainfall + humidity + medium K)
- **Best crops**: All others at F1 ≥ 0.976

---

## Datasets (All Real-World, No Synthetic)

### Primary Dataset
**Crop Recommendation Dataset (Kaggle)**
- Source: https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset
- Rows: 2,200 samples (100 per class)
- Features: N, P, K, Temperature, Humidity, pH, Rainfall (7 features)
- Target: 22 crop types
- Balance: Perfectly balanced (100 samples each)
- License: Open for research use

---

## Complete ML Pipeline (7 Sessions)

### Session 1: Data Acquisition & EDA
- Download real dataset from GitHub mirror
- Full EDA: distributions, correlations, violin plots, pair plots, radar chart
- Output: 7 figures, descriptive statistics table

### Session 2: Preprocessing & Feature Selection
- StandardScaler (selected over MinMax/Robust)
- 5 feature selection methods: Chi-Square, Mutual Information, RFE, LASSO, Boruta
- Consensus ranking via normalized aggregation
- Output: 2 figures, 8 tables

### Session 3: Model Training (10 classifiers × 4 feature subsets)
- RandomForest, SVM-RBF, KNN, DecisionTree, GradientBoosting, LogisticRegression, MLP, GaussianNB, XGBoost, LightGBM
- Feature subsets: all (7), top5, top4, top3
- 5-fold CV per model
- Output: 1 figure, training results table

### Session 4: Evaluation & Figures
- Confusion matrices (all 10 classifiers)
- ROC curves (micro/macro/per-class)
- Per-class F1 analysis
- 5-fold and 10-fold cross-validation
- Friedman statistical test
- Radar chart and comprehensive performance bar charts
- Output: 6 figures, 4 tables

### Session 5: Final Compilation
- Master results table (all classifiers × all feature subsets)
- Ablation study (accuracy vs feature count)
- Sensitivity analysis (RF n_estimators: 10→300)
- Literature comparison
- Paper-ready summary with central claim
- Output: 2 figures, 3 tables

### Session 6: Interpretability & Robustness
- SHAP TreeExplainer on RandomForest (200 test samples)
- Beeswarm plots for apple and coffee
- Noise injection (σ: 0.1→3.0)
- Feature dropout (all 7 features individually)
- Missing value imputation (10%→50%)
- Nemenyi post-hoc test with critical difference diagram
- Agricultural domain interpretation
- Publication overview (1-page visual abstract)
- Output: 6 figures, 3 tables, domain interpretation JSON

### Session 7: Strategic Research Enhancements *(NEW)*
- **SHAP vs FS correlation** (Spearman/Kendall Tau)
- **Per-crop error analysis** with agronomic reasoning
- **Robustness threshold** identification (σ=0.5 critical boundary)
- **Statistical conclusion** (practical vs statistical significance)
- **Classifier family grouping** with deployment recommendations
- **Updated paper summary** with 6 contributions
- Output: 4 figures, 1 table, 6 metric JSONs

### Running the Pipeline
```bash
cd Crop-Research

# Run all sessions sequentially
python3 pipeline.py --all

# Run a specific session
python3 pipeline.py --session 1   # Data & EDA
python3 pipeline.py --session 2   # Preprocessing & FS
python3 pipeline.py --session 3   # Model Training
python3 pipeline.py --session 4   # Evaluation
python3 pipeline.py --session 5   # Compilation
python3 pipeline.py --session 6   # Interpretability
python3 pipeline.py --session 7   # Research Enhancements
python3 pipeline.py --session 8   # Technical Depth (CV audit, effect sizes, calibration, limitations)

# Skip completed sessions
python3 pipeline.py --all --skip 5
```

---

## Project Structure
```
Crop-Research/
├── pipeline.py              # Complete pipeline (7 sessions, checkpoint-based)
├── requirements.txt         # Python dependencies
├── download_data.py         # Data download utility (fallback)
├── README.md                # This file
├── data/
│   ├── raw/                 # Downloaded dataset (CSV)
│   └── checkpoints/         # Session flags + pickled models
├── results/
│   ├── figures/             # 78 publication-ready figures (PNG + PDF)
│   ├── tables/              # 19 CSV/LaTeX tables
│   └── metrics/             # 15 JSON metric/summary files
└── logs/                    # Pipeline execution logs
```

---

## Generated Artifacts Summary

### Figures (78 files: 39 PNG + 39 PDF)
| # | File | Description |
|---|------|-------------|
| 1 | `01_feature_distributions` | Feature value distributions (histograms) |
| 2 | `02_boxplots` | Feature boxplots |
| 3 | `03_correlation_heatmap` | Pearson correlation matrix |
| 4 | `04_class_distribution` | Class balance (22 crops × 100) |
| 5 | `05_violin_per_class` | Feature distributions per crop |
| 6 | `06_pairplot` | Feature pair scatter plots |
| 7 | `07_radar_chart` | Feature radar chart per crop |
| 8 | `08_scaling_comparison` | Standard vs MinMax vs Robust scaler |
| 9 | `09_feature_selection_comparison` | 6 FS methods compared |
| 10 | `10_feature_selection_heatmap` | FS scores heatmap |
| 11 | `11_model_comparison` | Classifier accuracy across subsets |
| 12 | `12_confusion_matrices` | Normalized confusion matrices (all 10) |
| 13 | `13_roc_*` | ROC curves per classifier (×10) |
| 14 | `14_per_class_f1` | Per-crop F1 scores |
| 15 | `15_cross_validation` | 5-fold vs 10-fold CV comparison |
| 16 | `16_top_classifiers_radar` | Top-5 classifier radar chart |
| 17 | `17_comprehensive_performance` | Accuracy/Precision/Recall/F1 bars |
| 18 | `18_ablation_study` | Accuracy vs feature count |
| 19 | `19_sensitivity_rf` | RF sensitivity to n_estimators |
| 20 | `20_shap_importance` | SHAP global feature importance |
| 21 | `21_shap_beeswarm_*` | SHAP beeswarm (apple, coffee) |
| 22 | `22_robustness_analysis` | Noise injection + feature dropout |
| 23 | `23_missing_value_robustness` | Missing data tolerance |
| 24 | `24_critical_difference_diagram` | Nemenyi CD diagram |
| 25 | `25_publication_overview` | 1-page visual abstract |
| **26** | `26_shap_vs_fs_correlation` | **SHAP vs FS scatter + bar** |
| **27** | `27_error_analysis` | **Per-crop error + misclassification** |
| **28** | `28_robustness_threshold` | **Threshold with reliability zones** |
| **29** | `29_classifier_families` | **Family comparison** |

### Metric JSON Files (15 files)
| File | Contents |
|------|----------|
| `final_summary.json` | Overall results + central claim |
| `paper_summary.json` | Title, abstract, 6 contributions |
| `feature_selection_summary.json` | FS consensus ranking |
| `friedman_test.json` | Friedman χ² test result |
| `nemenyi_test.json` | Nemenyi CD + mean ranks |
| `shap_importance.json` | SHAP feature importance values |
| `robustness_results.json` | Noise/dropout/missing raw data |
| `domain_interpretation.json` | Agronomic reasoning per feature |
| `figure_guide.json` | Main (10) vs supplementary (14+) |
| `session1_summary.json` | EDA statistics |
| `shap_fs_correlation.json` | Spearman/Kendall agreement |
| `error_analysis.json` | Per-crop metrics + confusion pairs |
| `robustness_interpretation.json` | Threshold + deployment advice |
| `statistical_conclusion.json` | Practical vs statistical significance |
| `classifier_families.json` | Family grouping + takeaway |

---

## Research Contributions

1. **Unified evaluation framework** integrating feature selection comparison, model robustness analysis, and SHAP-based interpretability for crop recommendation.
2. **Quantitative SHAP vs feature selection agreement analysis** (Spearman ρ=0.679) revealing that statistical and model-driven importance capture partially distinct aspects of feature relevance — neither alone is sufficient.
3. **Non-linear robustness characterization** under Gaussian noise and missing data: performance remains stable under low perturbations (σ ≤ 0.1) but degrades sharply beyond moderate noise levels (σ ≥ 0.5), exposing a critical reliability gap for field deployment.
4. **Per-crop error analysis with agronomic reasoning**, linking model misclassifications to shared soil-climate profiles (e.g., rice↔jute), suggesting that incorporating additional discriminative features (e.g., soil texture or seasonal patterns) may reduce ambiguity.
5. **Statistical vs practical significance analysis** demonstrating that despite Friedman test significance (p<0.01), maximum classifier difference is only 2.67%, supporting the use of computationally efficient models in resource-constrained agricultural settings.
6. **Classifier family comparison** showing no family demonstrates a decisive advantage — reinforcing that dataset simplicity, not model complexity, drives performance. Probabilistic models offer the best efficiency tradeoff for edge deployment.

---

## Target Journals

### 🏆 Top Recommendation
**Heliyon (Cell Press / Elsevier)**
- Fee: Free submission (APC ~₹15,000)
- Review: ~1-2 months
- SCI indexed, moderate acceptance rate
- Covers agricultural + computing research

### Strong Alternatives
| Journal | Fee (INR) | Review | Notes |
|---------|-----------|--------|-------|
| **IEEE Access** | ₹15,000 OA | 4-6 weeks | Gold standard, fast |
| **Results in Engineering** | Free | 4-8 weeks | Engineering focus |
| **PeerJ CS** | ₹12,000-15,000 | 1-2 months | Open access, fast |
| **PLOS ONE** | ₹18,000-22,000 | 2-3 months | High acceptance |

---

## System Requirements
- Python 3.8+
- RAM: 3GB minimum (pipeline optimized for this)
- Storage: ~500MB for all artifacts
- Dependencies: `pip install -r requirements.txt`

---

## Quick Start
```bash
# Clone
git clone https://github.com/Aldrin7/Crop.git
cd Crop

# Install dependencies
pip install -r requirements.txt

# Run full pipeline
python3 pipeline.py --all
```
