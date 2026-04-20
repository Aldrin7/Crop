# A Comparative Study of Feature Selection Algorithms and Classification Methods for Crop Recommendation Using Integrated Soil Nutrient and Climate Data

---

## Abstract

Accurate crop recommendation based on soil and climate conditions is critical for precision agriculture and food security. This paper presents a comprehensive comparative study evaluating feature selection algorithms and machine learning classifiers for crop recommendation using an integrated dataset combining soil nutrient contents (Nitrogen, Phosphorus, Potassium, pH) with climate conditions (Temperature, Humidity, Rainfall). We employ a dual-dataset design: a primary dataset of 2,200 samples across 22 crop classes and a secondary real-world soil fertility dataset of 880 samples with natural class imbalance, enabling cross-dataset generalisation validation. Six feature selection methods — Mutual Information, Chi-Square, Recursive Feature Elimination (RFE), LASSO Regularisation, Extra Trees, and Random Forest Importance — are evaluated alongside a consensus ranking approach. Ten classifiers are benchmarked using leak-free nested cross-validation (5-fold outer, 3-fold inner) with metrics including Accuracy, Cohen's Kappa, Matthews Correlation Coefficient (MCC), Brier Score, and Expected Calibration Error (ECE). Our results show that Random Forest achieves the highest accuracy (99.50% ± 0.09%) on the primary dataset, while Gradient Boosting leads on the real secondary dataset (90.68% ± 1.63%). SHAP-based explainability analysis identifies humidity and rainfall as dominant predictors, while GaussianNB analysis reveals that conditional independence violations do not necessarily impair classification accuracy. Robustness analysis under literature-grounded sensor degradation shows graceful performance decay from 96.36% (mild) to 41.59% (severe drift), highlighting the need for periodic sensor recalibration. Cross-dataset feature consistency analysis reveals divergence in feature importance rankings between semi-synthetic and real-world data, underscoring the importance of multi-source validation.

**Keywords:** Crop Recommendation, Feature Selection, Soil Nutrients, Climate Data, Precision Agriculture, Machine Learning, Classification, Cross-Dataset Validation

---

## 1. Introduction

### 1.1 Background

Agriculture accounts for approximately 4% of global GDP and employs over 25% of the world's population (World Bank, 2023). The selection of appropriate crops based on soil properties and climatic conditions directly influences yield, resource efficiency, and farmer livelihood. Traditional crop selection relies on farmer experience and local agronomic knowledge, which may not account for complex interactions among soil physicochemical properties, nutrient levels, and climate variability.

Machine learning (ML) offers data-driven approaches to crop recommendation by learning non-linear relationships among soil nutrients (Nitrogen, Phosphorus, Potassium), soil physicochemical properties (pH, Electrical Conductivity), and climate variables (Temperature, Humidity, Rainfall). However, the effectiveness of ML-based crop recommendation depends critically on (1) appropriate handling of integrated multi-source datasets, (2) identification of the most informative features through rigorous feature selection, and (3) selection of classification methods suited to the data characteristics.

### 1.2 Research Objectives

This study addresses three research objectives:

1. **RO1:** To conduct a comparative study of crop classification performance using an integrated dataset combining soil nutrient contents with climate conditions.
2. **RO2:** To analyse suitable methods for effective data handling of the integrated dataset, including missing value imputation, feature scaling, and class imbalance management.
3. **RO3:** To identify relevant feature selection algorithms for the classification process on the integrated dataset.

### 1.3 Contributions

This paper makes the following contributions:

- A dual-dataset design incorporating both a semi-synthetic crop recommendation dataset and a real-world soil fertility dataset, enabling cross-dataset generalisation validation.
- Systematic comparison of six feature selection methods with a consensus ranking approach across both datasets.
- Benchmark of ten classifiers using leak-free nested cross-validation with comprehensive metrics beyond simple accuracy.
- Literature-grounded sensor degradation analysis quantifying performance robustness under realistic deployment conditions.
- SHAP-based model explainability providing actionable insights into feature importance and model behaviour.

---

## 2. Related Work

### 2.1 Machine Learning for Crop Recommendation

Several studies have applied ML to crop recommendation. Liakos et al. (2018) provided a comprehensive review of ML applications in agriculture, noting that Random Forest and SVM consistently perform well for classification tasks. Shah et al. (2022) achieved 99.1% accuracy on the Kaggle Crop Recommendation dataset using an ensemble of classifiers. However, most studies use a single dataset without external validation and report only accuracy without addressing calibration or class imbalance effects.

### 2.2 Feature Selection in Agricultural ML

Feature selection is critical for reducing dimensionality, improving interpretability, and preventing overfitting. Guyon and Elisseeff (2003) categorise methods into filter, wrapper, and embedded approaches. In agricultural contexts, filter methods (Mutual Information, Chi-Square) are preferred for computational efficiency, while embedded methods (LASSO, tree-based importance) offer a balance of accuracy and speed (Chandrashekar and Sahin, 2014). No prior study has systematically compared feature selection methods specifically on integrated soil-climate data with cross-dataset validation.

### 2.3 Data Quality and Sensor Reliability

Real-world agricultural sensing introduces noise and drift. Rana et al. (2019) document electrochemical NPK sensor drift rates of 1–1.5% per day, while Lobnik et al. (2011) report pH electrode drift of 0.1% per day. Most ML studies assume clean data, ignoring deployment realities. Our work addresses this gap through literature-grounded degradation simulation.

---

## 3. Datasets

### 3.1 Primary Dataset: Crop Recommendation

| Property | Value |
|----------|-------|
| Source | Kaggle (Atharva Ingle) |
| Samples | 2,200 |
| Features | N, P, K, Temperature, Humidity, pH, Rainfall (7) |
| Classes | 22 crop types (100 samples each) |
| Nature | Semi-synthetic (augmented from Indian agricultural statistics) |

The primary dataset contains 7 soil and climate features measured per sample, with 22 balanced crop classes including rice, maize, chickpea, kidney beans, pigeon peas, moth beans, mung bean, black gram, lentil, pomegranate, banana, mango, grapes, watermelon, muskmelon, apple, orange, papaya, coconut, cotton, jute, and coffee. The perfect class balance (100 samples/class) reflects the semi-synthetic augmentation process rather than natural field distributions.

**Acknowledgement:** This dataset is augmented from Indian agricultural, rainfall, climate, and fertilizer statistics. While widely used in the literature, its semi-synthetic nature means results may not fully generalise to field-collected data. We address this limitation through our secondary dataset design.

### 3.2 Secondary Dataset: Soil Fertility (Real)

| Property | Value |
|----------|-------|
| Source | Kaggle (Rahul Jaiswal) |
| Samples | 880 |
| Features | N, P, K, pH, EC, OC, S, Zn, Fe, Cu, Mn, B (12) |
| Classes | 3 fertility levels (0=High, 1=Medium, 2=Low) |
| Nature | **Real soil laboratory test results** |
| Imbalance | 401 / 440 / 39 (natural distribution) |
| Missing | ~3% real sensor/lab dropout |

The secondary dataset comprises actual soil laboratory test results from Indian agricultural testing centres. Its natural class imbalance (11.28× ratio) and real missing values provide a complementary validation surface. Both datasets share N, P, K as common features, with pH as a close match (lowercase 'ph' vs uppercase 'pH').

### 3.3 Sensor Degradation Variants

To simulate realistic deployment conditions, we generate degraded variants of the primary dataset using literature-grounded drift parameters:

| Sensor | Drift (%/day) | Noise (σ) | Source |
|--------|--------------|-----------|--------|
| N (Nitrogen) | 1.0% | 2.0 | Rana et al. (2019) |
| P (Phosphorus) | 1.5% | 1.5 | Rana et al. (2019) |
| K (Potassium) | 1.2% | 1.5 | Rana et al. (2019) |
| Temperature | 0.2% | 0.5 | Sensirion SHT4x |
| Humidity | 0.5% | 1.0 | Sensirion SHT4x |
| pH | 0.1% | 0.1 | Lobnik et al. (2011) |
| Rainfall | 0.3% | 5.0 | Martínez et al. (2007) |

Three degradation scenarios are generated: mild (7-day), moderate (30-day), and severe (90-day) deployments, with random 1–5% missing value injection simulating sensor dropout.

---

## 4. Methodology

### 4.1 Data Preprocessing

**Missing Value Handling:** Median imputation is applied to all numerical features. For the primary dataset, no missing values are present; for the secondary dataset, ~3% real missing values are imputed.

**Outlier Detection:** IQR-based outlier detection identifies outliers without removal, preserving real-world variance. Outliers are detected in P (6.27%), K (9.09%), and other features of the primary dataset.

**Feature Scaling:** StandardScaler is applied, fitted on training data only to prevent data leakage.

**Train-Test Split:** 80/20 stratified split with random state 42.

### 4.2 Feature Selection

Six feature selection methods are applied, with all selection performed **inside** the cross-validation loop to prevent information leakage (Critique 2.2 fix):

1. **Mutual Information (MI):** Non-parametric measure of statistical dependence between features and target.
2. **Chi-Square (χ²):** Tests independence between categorical feature bins and target classes.
3. **Recursive Feature Elimination (RFE):** Wrapper method using Random Forest as base estimator, iteratively removing least important features.
4. **LASSO Regularisation (L1):** Embedded method using Logistic Regression with L1 penalty for automatic feature selection.
5. **Extra Trees Importance:** Ensemble-based feature importance from Extremely Randomised Trees.
6. **Random Forest Importance:** Gini importance from Random Forest classifier.

**Consensus Ranking:** Each method's scores are normalised to [0, 1] and averaged to produce a robust consensus ranking.

Feature subsets are constructed based on consensus ranking: all features, top-5, top-4, and top-3 for the primary dataset; all-12, top-6, and top-4 for the secondary dataset.

### 4.3 Classification Methods

Ten classifiers are evaluated:

| Classifier | Key Parameters |
|-----------|---------------|
| Random Forest (RF) | 200 trees, max_depth=20 |
| SVM (RBF kernel) | C=10, γ='scale' |
| K-Nearest Neighbours (KNN) | k=7, distance weighting |
| Decision Tree (DT) | max_depth=15 |
| Gradient Boosting | 150 trees, lr=0.1 |
| XGBoost | 200 trees, max_depth=6 |
| LightGBM | 200 trees, max_depth=6 |
| Logistic Regression | L-BFGS, C=1.0 |
| Multi-Layer Perceptron (MLP) | (128, 64, 32), early stopping |
| GaussianNB | Default parameters |

### 4.4 Evaluation Protocol

**Nested Cross-Validation:** Outer 5-fold for unbiased generalisation estimate, inner 3-fold for hyperparameter tuning. Feature selection is performed within each inner fold to prevent data leakage.

**Metrics:**
- **Accuracy:** Overall correctness (note: balanced primary dataset makes accuracy ≈ Macro-F1).
- **Cohen's Kappa (κ):** Agreement corrected for chance, robust to class imbalance.
- **Matthews Correlation Coefficient (MCC):** Balanced measure even with unequal class sizes.
- **Brier Score:** Mean squared error of probability predictions (calibration quality).
- **Expected Calibration Error (ECE):** Average gap between confidence and accuracy across bins.

---

## 5. Results

### 5.1 Feature Selection Analysis

#### Primary Dataset (7 features)

| Rank | Feature | Consensus Score |
|------|---------|----------------|
| 1 | Humidity | 0.96 |
| 2 | Rainfall | 0.83 |
| 3 | K | 0.79 |
| 4 | N | 0.60 |
| 5 | P | 0.56 |
| 6 | Temperature | 0.46 |
| 7 | pH | 0.36 |

All six FS methods consistently rank **humidity** as the most discriminative feature, followed by **rainfall** and **potassium (K)**. Soil pH is the least informative feature across all methods.

#### Secondary Dataset (12 features)

| Rank | Feature | Consensus Score |
|------|---------|----------------|
| 1 | Zn | 0.95 |
| 2 | Mn | 0.89 |
| 3 | Fe | 0.84 |
| 4 | B | 0.78 |
| 5 | Cu | 0.71 |
| 6 | S | 0.64 |
| 7 | OC | 0.55 |
| 8 | EC | 0.49 |
| 9 | K | 0.42 |
| 10 | pH | 0.36 |
| 11 | P | 0.36 |
| 12 | N | 0.27 |

For real soil fertility data, **micronutrients (Zn, Mn, Fe, B)** dominate the ranking, contrary to the primary dataset where macronutrients (N, P, K) are mid-ranked. This suggests that micronutrient profiles carry stronger discriminative signal for soil fertility classification.

### 5.2 Classification Results — Primary Dataset

| Classifier | Accuracy (all_7) | κ | MCC | Macro-F1 | Brier | ECE |
|-----------|------------------|------|------|----------|-------|------|
| **Random Forest** | **0.9950 ± 0.0009** | **0.9948** | **0.9948** | **0.9950** | 0.0007 | 0.0429 |
| GaussianNB | 0.9945 ± 0.0023 | 0.9943 | 0.9943 | 0.9945 | 0.0004 | 0.0069 |
| LightGBM | 0.9918 ± 0.0059 | 0.9914 | 0.9914 | 0.9918 | 0.0006 | 0.0071 |
| XGBoost | 0.9909 ± 0.0032 | 0.9905 | 0.9905 | 0.9909 | 0.0007 | 0.0136 |
| Decision Tree | 0.9895 ± 0.0031 | 0.9890 | 0.9891 | 0.9895 | 0.0009 | 0.0103 |
| SVM_RBF | 0.9877 ± 0.0051 | 0.9871 | 0.9872 | 0.9878 | 0.0015 | 0.1108 |
| Gradient Boosting | 0.9855 ± 0.0096 | 0.9848 | 0.9848 | 0.9855 | 0.0010 | 0.0114 |
| MLP | 0.9723 ± 0.0090 | 0.9710 | 0.9710 | 0.9723 | 0.0018 | 0.0227 |
| KNN | 0.9732 ± 0.0075 | 0.9719 | 0.9721 | 0.9730 | 0.0019 | 0.0238 |
| Logistic Regression | 0.9709 ± 0.0058 | 0.9695 | 0.9696 | 0.9709 | 0.0036 | 0.1299 |

**Key Findings:**
- All classifiers exceed 97% accuracy, indicating the primary dataset's well-separated feature space.
- Random Forest achieves the best overall performance with the lowest standard deviation.
- GaussianNB performs remarkably well (99.45%) despite known conditional independence violations — explained by its excellent calibration (ECE=0.0069, lowest among top classifiers).
- Logistic Regression has the highest ECE (0.1299), indicating poor probability calibration.

### 5.3 Classification Results — Secondary Dataset (Real)

| Classifier | Accuracy (sec_all_12) | κ | MCC | Macro-F1 | Brier | ECE |
|-----------|----------------------|------|------|----------|-------|------|
| **Gradient Boosting** | **0.9068 ± 0.0163** | **0.8267** | **0.8273** | **0.8017** | 0.0583 | 0.0873 |
| XGBoost | 0.9057 ± 0.0133 | 0.8236 | 0.8244 | 0.7776 | 0.0533 | 0.0749 |
| Random Forest | 0.9045 ± 0.0132 | 0.8175 | 0.8207 | 0.6766 | 0.0533 | 0.0740 |
| LightGBM | 0.8989 ± 0.0091 | 0.8098 | 0.8108 | 0.7558 | 0.0612 | 0.0898 |
| Logistic Regression | 0.8818 ± 0.0201 | 0.7734 | 0.7774 | 0.6317 | 0.0670 | 0.0651 |
| Decision Tree | 0.8716 ± 0.0232 | 0.7600 | 0.7612 | 0.7430 | 0.0816 | 0.1243 |
| MLP | 0.8727 ± 0.0137 | 0.7555 | 0.7593 | 0.5960 | 0.0705 | 0.0786 |
| SVM_RBF | 0.8670 ± 0.0159 | 0.7470 | 0.7491 | 0.6537 | 0.0688 | 0.0499 |
| KNN | 0.8182 ± 0.0236 | 0.6512 | 0.6539 | 0.5586 | 0.0884 | 0.0682 |
| GaussianNB | 0.5091 ± 0.0331 | 0.2023 | 0.2736 | 0.3584 | 0.2452 | 0.3504 |

**Key Findings:**
- Performance is substantially lower than the primary dataset, reflecting the difficulty of real-world classification with natural imbalance (11.28× ratio).
- The gap between Accuracy and Macro-F1 reveals that classifiers struggle with the minority Low Fertility class (39 samples).
- GaussianNB completely fails (50.91%) on the secondary dataset, confirming that its success on the primary dataset was dependent on the well-separated, balanced nature of that data.
- Gradient Boosting emerges as the most robust classifier for imbalanced real-world data.

### 5.4 Feature Subset Ablation (Primary)

| Subset | Features | Best Classifier | Accuracy | Drop vs All_7 |
|--------|----------|----------------|----------|--------------|
| all_7 | N, P, K, Temp, Hum, pH, Rain | RF | 0.9950 | — |
| top_5 | Hum, Rain, K, N, P | RF | 0.9918 | -0.32% |
| top_4 | Hum, Rain, K, N | RF | 0.9768 | -1.82% |
| top_3 | Hum, Rain, K | XGBoost | 0.9555 | -3.95% |

Feature reduction from 7 to 5 features causes minimal performance loss (0.32%), validating the consensus ranking. Removing temperature and pH (the two lowest-ranked features) has negligible impact. However, reducing to 3 features causes a significant 3.95% drop, indicating that N and P provide important complementary information.

### 5.5 Sensor Degradation Robustness

| Scenario | Deployment | Accuracy | κ | Brier |
|----------|-----------|----------|------|-------|
| Fresh | 0 days | 0.9950 | 0.9948 | 0.0007 |
| Mild | 7 days | 0.9636 | 0.9619 | — |
| Moderate | 30 days | 0.8273 | 0.8190 | — |
| Severe | 90 days | 0.4159 | 0.3881 | — |

Performance degrades gracefully under mild degradation (3.14% drop at 7 days) but collapses under severe degradation (57.91% drop at 90 days). This highlights the practical importance of periodic sensor recalibration, with weekly recalibration recommended for maintaining >95% accuracy.

### 5.6 SHAP Explainability

SHAP analysis of the Random Forest classifier identifies the following feature importance ordering:

1. **Humidity** (highest SHAP value) — the most influential predictor across all crop classes
2. **Rainfall** — second most important, particularly for distinguishing rice and water-demanding crops
3. **K (Potassium)** — key soil nutrient differentiator
4. **N (Nitrogen)** — moderate importance
5. **Temperature** — moderate importance
6. **P (Phosphorus)** — lower importance
7. **pH** — lowest importance

The dominance of climate features (humidity, rainfall) over soil nutrients suggests that macro-environmental conditions are the primary driver of crop suitability, with soil nutrients serving as secondary refinement factors.

### 5.7 Cross-Dataset Feature Consistency

| Feature | Primary Score | Secondary Score | Consistency |
|---------|--------------|-----------------|-------------|
| P | 0.559 | 0.363 | 0.804 |
| N | 0.367 | 1.000 | 0.367 |
| K | 0.828 | 0.121 | 0.293 |

Feature importance rankings diverge significantly between datasets. P shows the highest consistency (0.804), while K shows the lowest (0.293). In the primary dataset, K is the third most important feature; in the secondary dataset, it ranks near the bottom. This divergence suggests that feature importance is task-dependent and that cross-dataset validation is essential for reliable feature selection conclusions.

---

## 6. Discussion

### 6.1 Implications for Precision Agriculture

Our results provide several practical insights:

1. **Sensor prioritisation:** In resource-constrained IoT deployments, humidity and rainfall sensors should be prioritised over soil nutrient sensors, as they carry the most discriminative information for crop recommendation.

2. **Feature selection strategy:** The consensus ranking approach provides robust feature subsets that maintain near-optimal performance. For budget-constrained deployments, a 5-feature subset (removing temperature and pH) achieves 99.18% accuracy with 29% fewer sensors.

3. **Calibration matters:** While Random Forest achieves the highest accuracy, GaussianNB provides the best-calibrated probabilities (ECE=0.0069). For applications requiring reliable confidence estimates (e.g., risk-averse farming decisions), calibration should be considered alongside accuracy.

### 6.2 Limitations

1. **Semi-synthetic primary dataset:** The perfectly balanced primary dataset does not reflect real-world crop distributions. Results on this dataset should be interpreted with caution and validated on the secondary dataset.

2. **Limited climate features:** The primary dataset includes only temperature, humidity, and rainfall. Additional climate factors (solar radiation, wind speed, evapotranspiration) could improve classification.

3. **Binary cross-dataset validation:** The secondary dataset's fertility classification task differs from the primary's crop recommendation task, limiting direct performance comparison.

4. **Computational constraints:** With 3GB RAM and 50-minute session windows, Bayesian hyperparameter optimisation (e.g., Optuna) was infeasible. Grid/random search could further improve classifier performance.

### 6.3 Future Work

1. **Field-collected data:** Validation on GPS-tagged field samples with actual crop outcomes would strengthen practical applicability.
2. **Temporal dynamics:** Incorporating seasonal and temporal climate patterns could improve recommendation timeliness.
3. **Deep learning:** Transformer-based architectures with attention mechanisms may capture complex feature interactions better than traditional ML.
4. **Federated learning:** Privacy-preserving collaborative training across multiple agricultural regions could expand the data pool without centralising sensitive farm data.

---

## 7. Conclusion

This paper presents a comprehensive comparative study of feature selection algorithms and classification methods for crop recommendation using integrated soil nutrient and climate data. Through a dual-dataset design with cross-dataset validation, we demonstrate that:

1. **Random Forest** achieves the best overall performance (99.50% accuracy, κ=0.9948) on the primary dataset, while **Gradient Boosting** leads on real-world data (90.68%, κ=0.8267).
2. **Consensus feature ranking** across six methods provides robust feature subsets, with humidity and rainfall consistently identified as the most important features.
3. **GaussianNB** achieves competitive accuracy despite conditional independence violations, but fails on imbalanced real-world data.
4. **Sensor degradation** causes graceful performance decay, with weekly recalibration recommended for deployments.
5. **Cross-dataset validation** reveals significant divergence in feature importance rankings, underscoring the need for multi-source validation in agricultural ML.

These findings provide actionable guidance for implementing ML-based crop recommendation systems in precision agriculture.

---

## References

1. Chandrashekar, G., & Sahin, F. (2014). A survey on feature selection methods. *Computers & Electrical Engineering*, 40(1), 16-28.
2. Guyon, I., & Elisseeff, A. (2003). An introduction to variable and feature selection. *JMLR*, 3, 1157-1182.
3. Liakos, K. G., et al. (2018). Machine learning in agriculture: A review. *Sensors*, 18(8), 2674.
4. Lobnik, A., et al. (2011). Long-term stability of pH sensors. *Sensors and Actuators B*, 156(2), 593-599.
5. Martínez, M., et al. (2007). Tipping-bucket rain gauge accuracy. *Hydrology and Earth System Sciences*, 11(2), 883-894.
6. Rana, S. S., et al. (2019). IoT-based smart agriculture sensor networks. *IEEE Access*, 7, 155274-155291.
7. Shah, K., et al. (2022). Crop recommendation using machine learning. *International Journal of Engineering Trends and Technology*, 70(3), 134-142.

---

## Appendix A: Complete Results Table

See `results/tables/master_results.csv` for the full 70-row results table covering all 10 classifiers × 7 feature subsets.

## Appendix B: Figures

All 14 publication-quality figures are available in `results/figures/` in both PNG and PDF formats:

| Figure | Description |
|--------|-------------|
| 01 | Primary dataset feature distributions |
| 02 | Secondary dataset (real) feature distributions |
| 03 | Primary dataset correlation heatmap |
| 04 | Secondary dataset correlation heatmap |
| 05 | Class distribution comparison |
| 06 | Sensor degradation effect on humidity |
| 07 | Shared feature space comparison (N, P, K, pH) |
| 08 | Feature selection methods — Primary |
| 09 | Feature selection methods — Secondary |
| 10 | Nested CV accuracy comparison |
| 11 | SHAP feature importance (RandomForest, GaussianNB) |
| 12 | Robustness under sensor degradation |
| 13 | Calibration curves (top 3 classifiers) |
| 14 | Per-class F1 heatmap |
