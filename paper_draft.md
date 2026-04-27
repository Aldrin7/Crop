# A Comparative Study of Feature Selection Algorithms and Classification Methods for Crop Recommendation Using Integrated Soil Nutrient and Climate Data

---

**Anuradha Brijwal¹ · Praveena Chaturvedi²**

¹ Research Scholar, Department of Computer Science, Kanya Gurukul Campus Dehradun
² Professor, Department of Computer Science, Kanya Gurukul Campus Dehradun

Gurukul Kangri (Deemed to be University), Haridwar, Uttarakhand, India

---

## Abstract

Accurate crop recommendation based on soil and climate conditions is critical for precision agriculture and food security. This paper presents a comprehensive comparative study evaluating feature selection algorithms and machine learning classifiers for crop recommendation using an integrated dataset combining soil nutrient contents (Nitrogen, Phosphorus, Potassium, pH) with climate conditions (Temperature, Humidity, Rainfall). We employ a dual-dataset design: a primary dataset of 2,200 samples across 22 crop classes and a secondary real-world soil fertility dataset of 880 samples with natural class imbalance, enabling cross-dataset generalisation validation. Six feature selection methods — Mutual Information, Chi-Square, Recursive Feature Elimination (RFE), LASSO Regularisation, Extra Trees, and Random Forest Importance — are evaluated alongside a consensus ranking approach. Ten classifiers are benchmarked using leak-free 5-fold stratified cross-validation, where each fold independently applies a scikit-learn Pipeline (StandardScaler → SelectKBest with mutual information → Classifier) to prevent data leakage. Classifiers supporting it use `class_weight='balanced'` to address real-world imbalance. Metrics include Accuracy, Cohen's Kappa, Matthews Correlation Coefficient (MCC), Brier Score, and Expected Calibration Error (ECE). Our results show that Random Forest achieves the highest accuracy (99.50% ± 0.09%) on the primary dataset, while Random Forest with per-fold MI-selected features leads on the real secondary dataset (91.25% ± 0.77%). SHAP-based explainability analysis identifies humidity and rainfall as dominant predictors, while GaussianNB analysis reveals that conditional independence violations do not necessarily impair classification accuracy. Robustness analysis under literature-grounded sensor degradation shows graceful performance decay from 96.64% (7-day mild) to 43.82% (90-day severe drift), highlighting the need for periodic sensor recalibration. Cross-dataset feature consistency analysis reveals divergence in feature importance rankings between semi-synthetic and real-world data, underscoring the importance of multi-source validation.

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
- Benchmark of ten classifiers using leak-free 5-fold stratified cross-validation with per-fold Pipeline encapsulation (scaling, feature selection, and classification) and comprehensive metrics beyond simple accuracy.
- Application of `class_weight='balanced'` to classifiers that support it, addressing natural class imbalance in real-world data.
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
| Imbalance | 401 / 440 / 39 (natural distribution, ratio 11.28:1) |
| Missing | ~3% real sensor/lab dropout |

The secondary dataset comprises actual soil laboratory test results from Indian agricultural testing centres. Its natural class imbalance (11.28:1 ratio) and real missing values provide a complementary validation surface. Both datasets share N, P, K as common features, with pH as a close match (lowercase 'ph' vs uppercase 'pH').

### 3.3 Sensor Degradation Variants

To simulate realistic deployment conditions, we generate degraded variants of the primary dataset using literature-grounded drift parameters:

| Sensor | Drift (%/day) | Noise (σ) | Source |
|--------|--------------|-----------|--------|
| N (Nitrogen) | 1.0 | 2.0 | Rana et al. (2019) |
| P (Phosphorus) | 1.5 | 1.5 | Rana et al. (2019) |
| K (Potassium) | 1.2 | 1.5 | Rana et al. (2019) |
| Temperature | 0.2 | 0.5 | Sensirion SHT4x |
| Humidity | 0.5 | 1.0 | Sensirion SHT4x |
| pH | 0.1 | 0.1 | Lobnik et al. (2011) |
| Rainfall | 0.3 | 5.0 | Martínez et al. (2007) |

Three degradation scenarios are generated: mild (7-day), moderate (30-day), and severe (90-day) deployments, with random 1–5% missing value injection simulating sensor dropout.

---

## 4. Methodology

### 4.1 Data Preprocessing

**Missing Value Handling:** Median imputation is applied to all numerical features. For the primary dataset, no missing values are present; for the secondary dataset, ~3% real missing values are imputed.

**Outlier Detection:** IQR-based outlier detection identifies outliers without removal, preserving real-world variance. Outliers are detected in P (6.27%), K (9.09%), and other features of the primary dataset.

**Feature Scaling:** StandardScaler is applied within each cross-validation fold as part of the Pipeline, fitted exclusively on the training fold to prevent data leakage.

**Train-Test Split:** 80/20 stratified split with random state 42.

### 4.2 Leak-Free Pipeline Architecture

A critical methodological contribution of this work is the use of scikit-learn `Pipeline` objects to encapsulate the entire feature transformation and classification chain within each cross-validation fold. Previous approaches (and prior versions of this pipeline) suffered from data leakage, where feature scaling or feature selection was applied before the cross-validation split, allowing information from the test fold to influence training.

Our Pipeline per fold consists of three stages:

1. **StandardScaler** — fitted on the training fold only, applied to both training and validation fold.
2. **SelectKBest (mutual information)** — feature selection performed independently within each fold using `mutual_info_classif` as the scoring function. This ensures no information from the validation fold influences feature selection.
3. **Classifier** — the final estimator.

This architecture ensures that every preprocessing step observes only training data within each fold, providing an unbiased estimate of generalisation performance.

### 4.3 Feature Selection

Six feature selection methods are evaluated. For the primary analysis, a consensus ranking is computed from all methods applied to the full dataset for interpretability purposes. For the cross-validation training, per-fold mutual information selection is used (Section 4.2).

1. **Mutual Information (MI):** Non-parametric measure of statistical dependence between features and target.
2. **Chi-Square (χ²):** Tests independence between categorical feature bins and target classes.
3. **Recursive Feature Elimination (RFE):** Wrapper method using Random Forest as base estimator, iteratively removing least important features.
4. **LASSO Regularisation (L1):** Embedded method using Logistic Regression with L1 penalty for automatic feature selection.
5. **Extra Trees Importance:** Ensemble-based feature importance from Extremely Randomised Trees.
6. **Random Forest Importance:** Gini importance from Random Forest classifier.

**Consensus Ranking:** Each method's scores are normalised to [0, 1] and averaged to produce a robust consensus ranking. Feature subsets are constructed based on this ranking for the ablation study.

### 4.4 Classification Methods

Ten classifiers are evaluated. Where supported, `class_weight='balanced'` is applied to adjust for class imbalance in the secondary dataset:

| Classifier | Key Parameters | class_weight |
|-----------|---------------|-------------|
| Random Forest (RF) | 200 trees, max_depth=20 | balanced |
| SVM (RBF kernel) | C=10, γ='scale' | balanced |
| K-Nearest Neighbours (KNN) | k=7, distance weighting | N/A |
| Decision Tree (DT) | max_depth=15 | balanced |
| Gradient Boosting | 150 trees, lr=0.1 | N/A |
| XGBoost | 200 trees, max_depth=6 | N/A |
| LightGBM | 200 trees, max_depth=6 | N/A |
| Logistic Regression | L-BFGS, C=1.0 | balanced |
| Multi-Layer Perceptron (MLP) | (128, 64, 32), early stopping | N/A |
| GaussianNB | Default parameters | N/A |

Note: GaussianNB does not support `class_weight`, which has significant implications for its performance on imbalanced data (Section 5.3).

### 4.5 Evaluation Protocol

**Cross-Validation:** 5-fold stratified cross-validation with the Pipeline architecture described in Section 4.2. Feature selection and scaling are performed independently within each fold.

**Metrics:**
- **Accuracy:** Overall correctness. For the balanced primary dataset, accuracy ≈ Macro-F1.
- **Cohen's Kappa (κ):** Agreement corrected for chance, robust to class imbalance.
- **Matthews Correlation Coefficient (MCC):** Balanced measure even with unequal class sizes, considered the most informative single metric for multi-class problems.
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
| **Random Forest** | **0.9950 ± 0.0009** | **0.9948** | **0.9948** | **0.9950** | 0.0007 | 0.0430 |
| GaussianNB | 0.9945 ± 0.0023 | 0.9943 | 0.9943 | 0.9945 | 0.0004 | 0.0069 |
| LightGBM | 0.9918 ± 0.0053 | 0.9914 | 0.9914 | 0.9918 | 0.0006 | 0.0068 |
| XGBoost | 0.9909 ± 0.0032 | 0.9905 | 0.9905 | 0.9909 | 0.0007 | 0.0136 |
| Decision Tree | 0.9895 ± 0.0031 | 0.9890 | 0.9891 | 0.9895 | 0.0009 | 0.0103 |
| SVM (RBF) | 0.9877 ± 0.0051 | 0.9871 | 0.9872 | 0.9878 | 0.0015 | 0.1107 |
| Gradient Boosting | 0.9855 ± 0.0096 | 0.9848 | 0.9848 | 0.9855 | 0.0010 | 0.0114 |
| KNN | 0.9732 ± 0.0075 | 0.9719 | 0.9721 | 0.9730 | 0.0019 | 0.0238 |
| MLP | 0.9727 ± 0.0109 | 0.9714 | 0.9715 | 0.9727 | 0.0018 | 0.0222 |
| Logistic Regression | 0.9709 ± 0.0058 | 0.9695 | 0.9696 | 0.9709 | 0.0036 | 0.1298 |

**Key Findings:**
- All classifiers exceed 97% accuracy, indicating the primary dataset's well-separated feature space.
- Random Forest achieves the best overall performance with the lowest standard deviation (±0.09%).
- GaussianNB performs remarkably well (99.45%) despite known conditional independence violations — explained by its excellent calibration (ECE=0.0069, lowest among top classifiers). Its success on the primary dataset is attributable to the well-separated, balanced nature of the 22-class feature space.
- Logistic Regression has the highest ECE (0.1298), indicating poor probability calibration.

### 5.3 Classification Results — Secondary Dataset (Real, Imbalanced)

| Classifier | Accuracy (sec_mi_top_6) | κ | MCC | Macro-F1 | Brier | ECE |
|-----------|----------------------|------|------|----------|-------|------|
| **Random Forest** | **0.9125 ± 0.0077** | **0.8364** | **0.8371** | **0.8185** | 0.0483 | 0.0562 |
| XGBoost | 0.9034 ± 0.0200 | 0.8200 | 0.8205 | 0.7932 | 0.0565 | 0.0755 |
| LightGBM | 0.9034 ± 0.0183 | 0.8188 | 0.8197 | 0.7727 | 0.0582 | 0.0849 |
| Gradient Boosting | 0.8932 ± 0.0174 | 0.8002 | 0.8007 | 0.7659 | 0.0625 | 0.0954 |
| KNN | 0.8739 ± 0.0178 | 0.7574 | 0.7621 | 0.5964 | 0.0682 | 0.0666 |
| Decision Tree | 0.8705 ± 0.0170 | 0.7620 | 0.7632 | 0.7569 | 0.0814 | 0.1267 |
| MLP | 0.8875 ± 0.0170 | 0.7845 | 0.7885 | 0.6438 | 0.0731 | 0.1030 |
| SVM (RBF) | 0.8545 ± 0.0226 | 0.7410 | 0.7450 | 0.7229 | 0.0574 | 0.0510 |
| Logistic Regression | 0.7364 ± 0.0167 | 0.5626 | 0.5828 | 0.6101 | 0.1209 | 0.1107 |
| GaussianNB | 0.8011 ± 0.0711 | 0.6324 | 0.6398 | 0.5845 | 0.1005 | 0.0917 |

Results shown for the MI-selected top-6 feature subset (sec_mi_top_6), which yields the best performance. For comparison, results on the full 12-feature set (sec_all_12) are provided in Appendix A.

**Key Findings:**
- Performance is substantially lower than the primary dataset, reflecting the difficulty of real-world classification with natural imbalance (11.28:1 ratio).
- Random Forest with `class_weight='balanced'` achieves the best overall performance (91.25%, κ=0.8364), outperforming gradient boosting methods. The `class_weight` parameter helps the classifier attend to the minority Low Fertility class (39 samples).
- The gap between Accuracy and Macro-F1 reveals that classifiers struggle with the minority class. Even the best classifier (RF) shows a 9.40 percentage point gap between accuracy (91.25%) and macro-F1 (81.85%).
- GaussianNB performs poorly (80.11% with high variance ±7.11%) on the secondary dataset. Since GaussianNB does not support `class_weight`, it cannot compensate for the 11.28:1 imbalance ratio. This confirms that its success on the primary dataset was dependent on the well-separated, balanced nature of that data.
- Per-fold MI feature selection (top-6 from 12) improves RF performance from 90.68% (all-12 features) to 91.25%, demonstrating that removing noisy features benefits classification.

### 5.4 Feature Subset Ablation (Primary)

| Subset | Features | Best Classifier | Accuracy | Drop vs all_7 |
|--------|----------|----------------|----------|--------------|
| all_7 | N, P, K, Temp, Hum, pH, Rain | RF | 0.9950 | — |
| mi_top_5 | Hum, Rain, K, N, P | RF | 0.9905 | −0.45% |
| mi_top_4 | Hum, Rain, K, N | RF | 0.9782 | −1.68% |
| mi_top_3 | Hum, Rain, K | GB | 0.9555 | −3.95% |

Feature reduction from 7 to 5 features causes minimal performance loss (0.45%), validating the consensus ranking. Removing temperature and pH (the two lowest-ranked features) has negligible impact. However, reducing to 3 features causes a significant 3.95% drop, indicating that N and P provide important complementary information.

### 5.5 Sensor Degradation Robustness

| Scenario | Deployment Duration | Accuracy | κ | Brier |
|----------|-------------------|----------|------|-------|
| Fresh | 0 days | 0.9950 | 0.9948 | 0.0007 |
| Mild | 7 days | 0.9664 | 0.9648 | 0.0039 |
| Moderate | 30 days | 0.8177 | 0.8090 | 0.0138 |
| Severe | 90 days | 0.4382 | 0.4114 | 0.0331 |

Performance degrades gracefully under mild degradation (2.86% drop at 7 days) but collapses under severe degradation (55.68% drop at 90 days). This highlights the practical importance of periodic sensor recalibration, with weekly recalibration recommended for maintaining >95% accuracy.

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

| Feature | Primary Importance | Secondary Importance | Consistency |
|---------|-------------------|---------------------|-------------|
| P | 0.559 | 0.363 | 0.804 |
| N | 0.367 | 1.000 | 0.367 |
| K | 0.828 | 0.121 | 0.293 |

Feature importance rankings diverge significantly between datasets. P shows the highest consistency (0.804), while K shows the lowest (0.293). In the primary dataset, K is the third most important feature; in the secondary dataset, it ranks near the bottom. This divergence suggests that feature importance is task-dependent and that cross-dataset validation is essential for reliable feature selection conclusions.

---

## 6. Discussion

### 6.1 Implications for Precision Agriculture

Our results provide several practical insights:

1. **Sensor prioritisation:** In resource-constrained IoT deployments, humidity and rainfall sensors should be prioritised over soil nutrient sensors, as they carry the most discriminative information for crop recommendation.

2. **Feature selection strategy:** The consensus ranking approach provides robust feature subsets that maintain near-optimal performance. For budget-constrained deployments, a 5-feature subset (removing temperature and pH) achieves 99.05% accuracy with 29% fewer sensors.

3. **Calibration matters:** While Random Forest achieves the highest accuracy, GaussianNB provides the best-calibrated probabilities (ECE=0.0069). For applications requiring reliable confidence estimates (e.g., risk-averse farming decisions), calibration should be considered alongside accuracy.

4. **Class imbalance handling:** The application of `class_weight='balanced'` improves performance on the real-world imbalanced dataset. Practitioners deploying crop recommendation on real field data should prioritise classifiers with built-in imbalance handling or apply resampling techniques.

### 6.2 Methodological Lessons

This study's v3.1 revision corrected a critical data leakage issue present in earlier versions. Previous iterations applied StandardScaler and feature selection before cross-validation, allowing test fold information to influence training. The corrected Pipeline architecture ensures that all preprocessing occurs independently within each fold. This correction had a measurable impact on reported results, underscoring the importance of leak-free evaluation protocols in ML for agriculture.

### 6.3 Limitations

1. **Semi-synthetic primary dataset:** The perfectly balanced primary dataset does not reflect real-world crop distributions. Results on this dataset should be interpreted with caution and validated on the secondary dataset.

2. **Limited climate features:** The primary dataset includes only temperature, humidity, and rainfall. Additional climate factors (solar radiation, wind speed, evapotranspiration) could improve classification.

3. **Binary cross-dataset validation:** The secondary dataset's fertility classification task differs from the primary's crop recommendation task, limiting direct performance comparison.

4. **GaussianNB on imbalanced data:** GaussianNB does not support `class_weight='balanced'`, leading to its failure on the imbalanced secondary dataset (80.11% vs 91.25% for RF). This limits the applicability of Naive Bayes methods to real-world agricultural data with natural class imbalance.

5. **Computational constraints:** Hyperparameter optimisation (e.g., Optuna, Bayesian search) was not feasible within computational constraints. Further tuning could improve classifier performance, particularly for gradient boosting methods.

### 6.4 Future Work

1. **Field-collected data:** Validation on GPS-tagged field samples with actual crop outcomes would strengthen practical applicability.
2. **Temporal dynamics:** Incorporating seasonal and temporal climate patterns could improve recommendation timeliness.
3. **Deep learning:** Transformer-based architectures with attention mechanisms may capture complex feature interactions better than traditional ML.
4. **Federated learning:** Privacy-preserving collaborative training across multiple agricultural regions could expand the data pool without centralising sensitive farm data.
5. **Hyperparameter optimisation:** Systematic Bayesian optimisation (e.g., Optuna) for all classifiers could narrow performance gaps and identify optimal configurations.

---

## 7. Conclusion

This paper presents a comprehensive comparative study of feature selection algorithms and classification methods for crop recommendation using integrated soil nutrient and climate data. Through a dual-dataset design with cross-dataset validation and leak-free Pipeline-based evaluation, we demonstrate that:

1. **Random Forest** with `class_weight='balanced'` achieves the best overall performance on both datasets: 99.50% accuracy (κ=0.9948) on the primary dataset and 91.25% accuracy (κ=0.8364) on the real-world secondary dataset with natural class imbalance.
2. **Consensus feature ranking** across six methods provides robust feature subsets, with humidity and rainfall consistently identified as the most important features for crop recommendation.
3. **Per-fold feature selection** via scikit-learn Pipeline eliminates data leakage and provides honest generalisation estimates. The Pipeline architecture (StandardScaler → SelectKBest → Classifier) should be considered standard practice for agricultural ML studies.
4. **GaussianNB** achieves competitive accuracy on balanced, well-separated data despite conditional independence violations, but fails on imbalanced real-world data due to its lack of class weighting support.
5. **Sensor degradation** causes graceful performance decay, with weekly recalibration recommended for maintaining >95% accuracy in field deployments.
6. **Cross-dataset validation** reveals significant divergence in feature importance rankings between semi-synthetic and real-world data, underscoring the need for multi-source validation in agricultural ML.

These findings provide actionable guidance for implementing ML-based crop recommendation systems in precision agriculture, from sensor selection and feature engineering to classifier deployment and maintenance scheduling.

---

## References

1. Chandrashekar, G., & Sahin, F. (2014). A survey on feature selection methods. *Computers & Electrical Engineering*, 40(1), 16–28.
2. Guyon, I., & Elisseeff, A. (2003). An introduction to variable and feature selection. *Journal of Machine Learning Research*, 3, 1157–1182.
3. Liakos, K. G., et al. (2018). Machine learning in agriculture: A review. *Sensors*, 18(8), 2674.
4. Lobnik, A., et al. (2011). Long-term stability of pH sensors. *Sensors and Actuators B*, 156(2), 593–599.
5. Martínez, M., et al. (2007). Tipping-bucket rain gauge accuracy. *Hydrology and Earth System Sciences*, 11(2), 883–894.
6. Rana, S. S., et al. (2019). IoT-based smart agriculture sensor networks. *IEEE Access*, 7, 155274–155291.
7. Shah, K., et al. (2022). Crop recommendation using machine learning. *International Journal of Engineering Trends and Technology*, 70(3), 134–142.

---

## Appendix A: Complete Results Table

See `results/tables/master_results.csv` for the full 70-row results table covering all 10 classifiers × 7 feature subsets (primary: all_7, mi_top_5, mi_top_4, mi_top_3; secondary: sec_all_12, sec_mi_top_6, sec_mi_top_4).

### Primary Dataset — Full Results (all feature subsets)

| Features | Classifier | Accuracy | κ | MCC | Macro-F1 | Brier | ECE |
|----------|-----------|----------|------|------|----------|-------|------|
| all_7 | RandomForest | 0.9950±0.0009 | 0.9948 | 0.9948 | 0.9950 | 0.0007 | 0.0430 |
| all_7 | GaussianNB | 0.9945±0.0023 | 0.9943 | 0.9943 | 0.9945 | 0.0004 | 0.0069 |
| all_7 | LightGBM | 0.9918±0.0053 | 0.9914 | 0.9914 | 0.9918 | 0.0006 | 0.0068 |
| all_7 | XGBoost | 0.9909±0.0032 | 0.9905 | 0.9905 | 0.9909 | 0.0007 | 0.0136 |
| all_7 | DecisionTree | 0.9895±0.0031 | 0.9890 | 0.9891 | 0.9895 | 0.0009 | 0.0103 |
| all_7 | SVM_RBF | 0.9877±0.0051 | 0.9871 | 0.9872 | 0.9878 | 0.0015 | 0.1107 |
| all_7 | GradientBoosting | 0.9855±0.0096 | 0.9848 | 0.9848 | 0.9855 | 0.0010 | 0.0114 |
| all_7 | KNN | 0.9732±0.0075 | 0.9719 | 0.9721 | 0.9730 | 0.0019 | 0.0238 |
| all_7 | MLP | 0.9727±0.0109 | 0.9714 | 0.9715 | 0.9727 | 0.0018 | 0.0222 |
| all_7 | LogisticRegression | 0.9709±0.0058 | 0.9695 | 0.9696 | 0.9709 | 0.0036 | 0.1298 |
| mi_top_5 | RandomForest | 0.9905±0.0030 | 0.9900 | 0.9900 | 0.9904 | 0.0009 | 0.0371 |
| mi_top_5 | LightGBM | 0.9877±0.0040 | 0.9871 | 0.9872 | 0.9877 | 0.0008 | 0.0094 |
| mi_top_5 | SVM_RBF | 0.9814±0.0065 | 0.9805 | 0.9806 | 0.9814 | 0.0020 | 0.1139 |
| mi_top_5 | GradientBoosting | 0.9818±0.0085 | 0.9810 | 0.9810 | 0.9818 | 0.0014 | 0.0149 |
| mi_top_5 | DecisionTree | 0.9891±0.0063 | 0.9886 | 0.9886 | 0.9890 | 0.0009 | 0.0104 |
| mi_top_5 | GaussianNB | 0.9873±0.0037 | 0.9867 | 0.9867 | 0.9872 | 0.0009 | 0.0116 |
| mi_top_5 | XGBoost | 0.9868±0.0044 | 0.9862 | 0.9862 | 0.9868 | 0.0009 | 0.0133 |
| mi_top_5 | KNN | 0.9709±0.0084 | 0.9695 | 0.9697 | 0.9708 | 0.0020 | 0.0217 |
| mi_top_5 | MLP | 0.9695±0.0181 | 0.9681 | 0.9683 | 0.9691 | 0.0023 | 0.0389 |
| mi_top_5 | LogisticRegression | 0.9573±0.0068 | 0.9552 | 0.9554 | 0.9569 | 0.0060 | 0.1896 |
| mi_top_4 | RandomForest | 0.9782±0.0034 | 0.9771 | 0.9772 | 0.9780 | 0.0014 | 0.0264 |
| mi_top_4 | LightGBM | 0.9750±0.0072 | 0.9738 | 0.9738 | 0.9748 | 0.0021 | 0.0227 |
| mi_top_4 | GaussianNB | 0.9745±0.0030 | 0.9733 | 0.9734 | 0.9745 | 0.0016 | 0.0185 |
| mi_top_4 | DecisionTree | 0.9736±0.0053 | 0.9724 | 0.9724 | 0.9737 | 0.0023 | 0.0234 |
| mi_top_4 | XGBoost | 0.9723±0.0039 | 0.9710 | 0.9710 | 0.9723 | 0.0019 | 0.0132 |
| mi_top_4 | SVM_RBF | 0.9695±0.0042 | 0.9681 | 0.9682 | 0.9696 | 0.0028 | 0.1080 |
| mi_top_4 | GradientBoosting | 0.9664±0.0055 | 0.9648 | 0.9648 | 0.9668 | 0.0026 | 0.0297 |
| mi_top_4 | KNN | 0.9673±0.0023 | 0.9657 | 0.9658 | 0.9668 | 0.0022 | 0.0171 |
| mi_top_4 | MLP | 0.9559±0.0047 | 0.9538 | 0.9540 | 0.9558 | 0.0031 | 0.0411 |
| mi_top_4 | LogisticRegression | 0.9436±0.0124 | 0.9410 | 0.9411 | 0.9424 | 0.0080 | 0.2320 |
| mi_top_3 | RandomForest | 0.9645±0.0023 | 0.9629 | 0.9630 | 0.9646 | 0.0031 | 0.0607 |
| mi_top_3 | GradientBoosting | 0.9555±0.0047 | 0.9533 | 0.9534 | 0.9558 | 0.0036 | 0.0384 |
| mi_top_3 | LightGBM | 0.9545±0.0069 | 0.9524 | 0.9525 | 0.9547 | 0.0038 | 0.0419 |
| mi_top_3 | XGBoost | 0.9545±0.0073 | 0.9524 | 0.9525 | 0.9544 | 0.0033 | 0.0223 |
| mi_top_3 | GaussianNB | 0.9541±0.0049 | 0.9519 | 0.9523 | 0.9533 | 0.0031 | 0.0233 |
| mi_top_3 | DecisionTree | 0.9505±0.0092 | 0.9481 | 0.9482 | 0.9504 | 0.0041 | 0.0416 |
| mi_top_3 | SVM_RBF | 0.9445±0.0042 | 0.9419 | 0.9425 | 0.9431 | 0.0045 | 0.1162 |
| mi_top_3 | KNN | 0.9400±0.0146 | 0.9371 | 0.9375 | 0.9381 | 0.0040 | 0.0274 |
| mi_top_3 | MLP | 0.9364±0.0076 | 0.9333 | 0.9336 | 0.9357 | 0.0044 | 0.0403 |
| mi_top_3 | LogisticRegression | 0.9023±0.0080 | 0.8976 | 0.8983 | 0.8971 | 0.0139 | 0.3144 |

### Secondary Dataset — Full Results (all feature subsets)

| Features | Classifier | Accuracy | κ | MCC | Macro-F1 | Brier | ECE |
|----------|-----------|----------|------|------|----------|-------|------|
| sec_all_12 | RandomForest | 0.9068±0.0128 | 0.8224 | 0.8250 | 0.7021 | 0.0508 | 0.0755 |
| sec_all_12 | GradientBoosting | 0.9068±0.0163 | 0.8267 | 0.8273 | 0.8017 | 0.0584 | 0.0863 |
| sec_all_12 | XGBoost | 0.9057±0.0133 | 0.8236 | 0.8244 | 0.7776 | 0.0533 | 0.0749 |
| sec_all_12 | LightGBM | 0.9011±0.0133 | 0.8145 | 0.8153 | 0.7731 | 0.0588 | 0.0868 |
| sec_all_12 | MLP | 0.8750±0.0095 | 0.7596 | 0.7636 | 0.5972 | 0.0689 | 0.0808 |
| sec_all_12 | DecisionTree | 0.8716±0.0205 | 0.7649 | 0.7664 | 0.7365 | 0.0810 | 0.1222 |
| sec_all_12 | SVM_RBF | 0.8352±0.0152 | 0.6982 | 0.6988 | 0.6668 | 0.0691 | 0.0558 |
| sec_all_12 | KNN | 0.8170±0.0214 | 0.6488 | 0.6516 | 0.5578 | 0.0906 | 0.0719 |
| sec_all_12 | LogisticRegression | 0.7205±0.0056 | 0.5341 | 0.5509 | 0.5878 | 0.1227 | 0.0817 |
| sec_all_12 | GaussianNB | 0.5091±0.0331 | 0.2023 | 0.2736 | 0.3584 | 0.2452 | 0.3504 |
| sec_mi_top_6 | RandomForest | 0.9125±0.0077 | 0.8364 | 0.8371 | 0.8185 | 0.0483 | 0.0562 |
| sec_mi_top_6 | XGBoost | 0.9034±0.0200 | 0.8200 | 0.8205 | 0.7932 | 0.0565 | 0.0755 |
| sec_mi_top_6 | LightGBM | 0.9034±0.0183 | 0.8188 | 0.8197 | 0.7727 | 0.0582 | 0.0849 |
| sec_mi_top_6 | GradientBoosting | 0.8932±0.0174 | 0.8002 | 0.8007 | 0.7659 | 0.0625 | 0.0954 |
| sec_mi_top_6 | MLP | 0.8875±0.0170 | 0.7845 | 0.7885 | 0.6438 | 0.0731 | 0.1030 |
| sec_mi_top_6 | KNN | 0.8739±0.0178 | 0.7574 | 0.7621 | 0.5964 | 0.0682 | 0.0666 |
| sec_mi_top_6 | DecisionTree | 0.8705±0.0170 | 0.7620 | 0.7632 | 0.7569 | 0.0814 | 0.1267 |
| sec_mi_top_6 | SVM_RBF | 0.8545±0.0226 | 0.7410 | 0.7450 | 0.7229 | 0.0574 | 0.0510 |
| sec_mi_top_6 | GaussianNB | 0.8011±0.0711 | 0.6324 | 0.6398 | 0.5845 | 0.1005 | 0.0917 |
| sec_mi_top_6 | LogisticRegression | 0.7364±0.0167 | 0.5626 | 0.5828 | 0.6101 | 0.1209 | 0.1107 |
| sec_mi_top_4 | RandomForest | 0.9102±0.0110 | 0.8335 | 0.8339 | 0.8085 | 0.0483 | 0.0535 |
| sec_mi_top_4 | XGBoost | 0.9023±0.0154 | 0.8176 | 0.8180 | 0.7939 | 0.0541 | 0.0721 |
| sec_mi_top_4 | LightGBM | 0.8920±0.0219 | 0.7981 | 0.7986 | 0.7368 | 0.0620 | 0.0919 |
| sec_mi_top_4 | GradientBoosting | 0.8920±0.0203 | 0.7977 | 0.7985 | 0.7493 | 0.0649 | 0.0984 |
| sec_mi_top_4 | KNN | 0.8830±0.0214 | 0.7769 | 0.7793 | 0.6573 | 0.0645 | 0.0722 |
| sec_mi_top_4 | MLP | 0.8818±0.0208 | 0.7728 | 0.7770 | 0.6018 | 0.0827 | 0.1462 |
| sec_mi_top_4 | DecisionTree | 0.8773±0.0137 | 0.7741 | 0.7761 | 0.7696 | 0.0755 | 0.1181 |
| sec_mi_top_4 | SVM_RBF | 0.8636±0.0200 | 0.7573 | 0.7613 | 0.7471 | 0.0526 | 0.0496 |
| sec_mi_top_4 | GaussianNB | 0.7875±0.0724 | 0.6082 | 0.6243 | 0.5780 | 0.1027 | 0.0890 |
| sec_mi_top_4 | LogisticRegression | 0.7511±0.0308 | 0.5886 | 0.6098 | 0.6239 | 0.1203 | 0.1485 |

---

## Appendix B: Figures

All 14 publication-quality figures are available in `results/figures/` in both PNG and PDF formats:

| Figure | Description |
|--------|-------------|
| 01 | Primary dataset feature distributions (N, P, K, Temperature, Humidity, pH, Rainfall) |
| 02 | Secondary dataset feature distributions (12 soil fertility features) |
| 03 | Primary dataset correlation heatmap |
| 04 | Secondary dataset correlation heatmap |
| 05 | Class distribution comparison (22 crop classes vs 3 fertility levels) |
| 06 | Sensor degradation effect on humidity (fresh vs mild vs moderate vs severe) |
| 07 | Shared feature space comparison (N, P, K, pH across datasets) |
| 08 | Feature selection methods — Primary dataset (6 methods + consensus) |
| 09 | Feature selection methods — Secondary dataset (6 methods + consensus) |
| 10 | Cross-validation accuracy comparison across classifiers |
| 11 | SHAP feature importance (RandomForest, GaussianNB) |
| 12 | Robustness under sensor degradation (accuracy vs deployment days) |
| 13 | Calibration curves (top 3 classifiers) |
| 14 | Per-class F1 heatmap (22 crop classes × 10 classifiers) |

---

## Appendix C: Supplementary Tables

| Table | Description |
|-------|-------------|
| master_results.csv | Complete 70-row results (10 classifiers × 7 feature subsets) |
| nested_cv_results.csv | Per-fold cross-validation scores |
| per_class_f1.csv | Per-class F1 scores for all classifiers |
| fs_consensus.csv | Consensus feature ranking (primary dataset) |
| fs_sec_consensus.csv | Consensus feature ranking (secondary dataset) |
| cross_dataset_validation.csv | Cross-dataset feature consistency metrics |
| descriptive_stats_primary.csv | Descriptive statistics for primary dataset |
| descriptive_stats_secondary.csv | Descriptive statistics for secondary dataset |
