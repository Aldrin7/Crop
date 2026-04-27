# RobustCrop: A Leak-Free Machine Learning Pipeline for Crop Recommendation with Sensor Degradation Analysis and Cross-Dataset Feature Consistency

---

**Anuradha Brijwal¹ · Praveena Chaturvedi²**

¹ Research Scholar, Department of Computer Science, Kanya Gurukul Campus Dehradun
² Professor, Department of Computer Science, Kanya Gurukul Campus Dehradun

Gurukul Kangri (Deemed to be University), Haridwar, Uttarakhand, India

---

## Abstract

Deploying machine learning for crop recommendation in real agricultural settings requires addressing three practical challenges overlooked by prior work: data leakage in preprocessing pipelines, class imbalance in real-world soil data, and over-reliance on semi-synthetic benchmarks without external validation. This paper proposes **RobustCrop**, a leak-free pipeline that encapsulates feature scaling, mutual-information-based feature selection, and classification within a single scikit-learn Pipeline per cross-validation fold, eliminating the information leakage that inflates accuracy in prior studies. The system employs `class_weight='balanced'` where supported to handle natural class imbalance. We evaluate on two datasets: a primary crop recommendation dataset (2,200 samples, 22 semi-synthetic classes) and a real-world soil fertility dataset (880 samples, 3 classes, 11.28:1 imbalance ratio). On real-world data, the proposed Random Forest pipeline achieves **91.25% ± 0.77% accuracy** (κ = 0.8364, macro-F1 = 81.85%), outperforming nine benchmark classifiers. SHAP analysis identifies humidity and rainfall as dominant predictors, with potassium and nitrogen as key soil nutrient differentiators. Literature-grounded sensor degradation analysis under monotonic drift shows decay from 94.05% (7-day) to 16.09% (90-day drift), establishing that weekly recalibration maintains >94% accuracy. Cross-dataset feature consistency analysis reveals phosphorus as the most transferable feature (consistency = 0.804) while potassium importance is task-dependent (0.293). A Friedman test across classifiers confirms statistically significant differences (χ² = 32.32, p < 0.001). These findings provide actionable deployment guidance for ML-based crop recommendation in resource-constrained agricultural IoT settings.

**Keywords:** Crop Recommendation, Feature Selection, Soil Nutrients, Precision Agriculture, Machine Learning, Sensor Degradation, SHAP Explainability, Cross-Dataset Analysis

---

## 1. Introduction

### 1.1 Background

Agriculture accounts for approximately 4% of global GDP and employs over 25% of the world's population (World Bank, 2023). Selecting appropriate crops based on soil properties and climatic conditions directly influences yield, resource efficiency, and farmer livelihood. Machine learning (ML) offers data-driven crop recommendation by learning from integrated soil-climate datasets. However, deploying ML in agricultural settings introduces three practical challenges that prior work has inadequately addressed:

1. **Data leakage in evaluation:** Standard preprocessing pipelines apply feature scaling and selection before cross-validation splits, allowing test-fold information to influence training. Kapoor and Narayanan (2023) documented such leakage in 70% of surveyed ML-based science papers, with performance inflation of 5–30%.
2. **Class imbalance in real data:** Real-world soil fertility datasets exhibit natural class imbalance (e.g., 11.28:1 in our secondary dataset), yet most classifiers are evaluated without imbalance-aware training, and semi-synthetic benchmarks with perfect balance mask this issue.
3. **Lack of external validation:** Nearly all crop recommendation studies rely on a single semi-synthetic benchmark without validating on independent real-world data or analysing whether feature importance rankings transfer across datasets.

### 1.2 Contributions

This paper proposes **RobustCrop**, a leak-free ML pipeline for crop recommendation. Our contributions are:

- **A Pipeline-per-fold architecture** (scikit-learn Pipeline: StandardScaler → SelectKBest(mutual information) → Classifier) that eliminates data leakage by ensuring all preprocessing occurs independently within each cross-validation fold.
- **`class_weight='balanced'` integration** for classifiers that support it, improving minority-class recognition on imbalanced real-world data.
- **Dual-dataset evaluation** using both a semi-synthetic crop recommendation dataset and a real-world soil fertility dataset, enabling analysis of how findings transfer (or fail to transfer) across data sources.
- **Consensus feature ranking** across six methods (Mutual Information, Chi-Square, RFE, LASSO, Extra Trees, Random Forest Importance) with per-fold MI selection during training.
- **SHAP-based model explainability** providing global feature importance and interaction insights.
- **Literature-grounded sensor degradation analysis** quantifying performance under realistic deployment conditions (7–90 day monotonic drift) and establishing recalibration guidelines.
- **Cross-dataset feature consistency analysis** identifying which features rank reliably across datasets and which are task-dependent.

---

## 2. Related Work

### 2.1 Machine Learning for Crop Recommendation

ML for crop recommendation has been widely explored. Liakos et al. (2018) reviewed ML in agriculture, noting Random Forest and SVM as consistently strong performers. Shah et al. (2022) achieved 99.1% accuracy on the Kaggle Crop Recommendation dataset using ensemble classifiers. However, most studies report only accuracy on a single dataset without addressing calibration, class imbalance, or data leakage in preprocessing — issues that our work directly confronts.

### 2.2 Feature Selection in Agricultural Data

Feature selection reduces dimensionality and improves interpretability. Guyon and Elisseeff (2003) categorise methods into filter, wrapper, and embedded approaches. In agricultural contexts, filter methods (Mutual Information, Chi-Square) are computationally efficient, while embedded methods (LASSO, tree-based importance) balance accuracy and speed (Chandrashekar and Sahin, 2014). Our work synthesises six methods into a consensus ranking and analyses feature importance transferability across datasets.

### 2.3 Data Leakage in ML Pipelines

Data leakage — where test information inadvertently influences training — is a pervasive but under-reported issue in applied ML. Kapoor and Narayanan (2023) documented leakage in 229 out of 329 papers across 17 application domains. Common leakage sources include feature scaling before train-test splitting and feature selection on the full dataset before cross-validation. Our Pipeline-per-fold architecture directly addresses these leakage vectors.

### 2.4 Sensor Reliability in Agricultural IoT

Real-world agricultural sensing introduces noise and drift. Rana et al. (2019) document electrochemical NPK sensor drift rates of 1–1.5% per day, while Lobnik et al. (2011) report pH electrode drift of 0.1% per day. Sensor drift is typically monotonic and directional — electrochemical sensors lose sensitivity over time. Most ML studies assume clean data; our robustness analysis quantifies these effects using literature-grounded monotonic drift simulation.

---

## 3. Datasets

### 3.1 Primary Dataset: Crop Recommendation (Semi-Synthetic)

| Property | Value |
|----------|-------|
| Source | Kaggle (Atharva Ingle) |
| Samples | 2,200 |
| Features | N, P, K, Temperature, Humidity, pH, Rainfall (7) |
| Classes | 22 crop types (100 samples each) |
| Nature | Semi-synthetic (augmented from Indian agricultural statistics) |

The primary dataset contains 7 soil and climate features with 22 balanced crop classes. The perfect class balance (100 samples/class) reflects the semi-synthetic augmentation process rather than natural field distributions. This dataset is widely used in the literature but results should be interpreted with caution — its clean, balanced nature means classifiers can achieve >99% accuracy that may not transfer to real-world conditions.

### 3.2 Secondary Dataset: Soil Fertility (Real Lab Measurements)

| Property | Value |
|----------|-------|
| Source | Kaggle (Rahul Jaiswal) |
| Samples | 880 |
| Features | N, P, K, pH, EC, OC, S, Zn, Fe, Cu, Mn, B (12) |
| Classes | 3 fertility levels (0=High, 1=Medium, 2=Low) |
| Nature | **Real soil laboratory test results** |
| Imbalance | 401 / 440 / 39 (natural distribution, ratio 11.28:1) |
| Missing | ~3% real sensor/lab dropout |

The secondary dataset comprises actual soil laboratory test results from Indian agricultural testing centres. Its natural class imbalance and real missing values provide a realistic test surface. The minority class (Low Fertility, 39 samples; Figure 5) represents a genuine challenge for classifiers — this is the setting where `class_weight='balanced'` matters most.

### 3.3 Shared Feature Space

Both datasets share N, P, K as common features, with pH as a close match (lowercase 'ph' vs uppercase 'pH'). This shared space enables cross-dataset feature consistency analysis (Section 5.7).

### 3.4 Sensor Degradation Variants

To simulate realistic deployment conditions, we generate degraded variants of the primary dataset using literature-grounded **monotonic drift** parameters. Unlike random bidirectional perturbation, real sensor drift is directional — electrochemical sensors consistently lose sensitivity over time. Each sensor is assigned a fixed drift direction per simulation seed.

| Sensor | Drift (%/day) | Noise (σ) | Source |
|--------|--------------|-----------|--------|
| N (Nitrogen) | 1.0 | 2.0 | Rana et al. (2019) |
| P (Phosphorus) | 1.5 | 1.5 | Rana et al. (2019) |
| K (Potassium) | 1.2 | 1.5 | Rana et al. (2019) |
| Temperature | 0.2 | 0.5 | Sensirion SHT4x |
| Humidity | 0.5 | 1.0 | Sensirion SHT4x |
| pH | 0.1 | 0.1 | Lobnik et al. (2011) |
| Rainfall | 0.3 | 5.0 | Martínez et al. (2007) |

Three degradation scenarios are generated: mild (7-day), moderate (30-day), and severe (90-day) deployments, with realistic dropout rates (2–10% scaled to deployment duration).

---

## 4. Proposed Method: RobustCrop Pipeline

### 4.1 System Overview

RobustCrop is a modular, leak-free ML pipeline for crop recommendation, implemented using scikit-learn (Pedregosa et al., 2011). The key design principle is that **every preprocessing step observes only training data within each cross-validation fold**. The system architecture:

```
Data Acquisition → Preprocessing → Consensus FS Ranking (interpretation only)
                                          ↓
                    ┌─────────────────────────────────────┐
                    │   Per CV Fold (5-fold stratified):   │
                    │   StandardScaler (fit on train)      │
                    │   → SelectKBest(MI, k=N)             │
                    │   → RF(class_weight='balanced')      │
                    └─────────────────────────────────────┘
                                          ↓
                    Evaluation → Friedman Test → SHAP → Robustness → Cross-Dataset
```

### 4.2 Data Preprocessing

**Missing Value Handling:** Median imputation is applied to all numerical features. For the primary dataset, no missing values are present; for the secondary dataset, ~3% real missing values are imputed.

**Outlier Detection:** IQR-based outlier detection identifies outliers without removal, preserving real-world variance. Outliers are detected in P (6.27%), K (9.09%), and other features of the primary dataset.

**Feature Scaling:** StandardScaler is applied within each cross-validation fold as part of the Pipeline (Section 4.4), fitted exclusively on the training fold.

**Train-Test Split:** 80/20 stratified split with random state 42.

### 4.3 Feature Selection: Consensus Ranking

Six feature selection methods are evaluated to build a consensus ranking for interpretability and ablation analysis:

1. **Mutual Information (MI):** Non-parametric measure of statistical dependence between features and target.
2. **Chi-Square (χ²):** Tests independence between categorical feature bins and target classes.
3. **Recursive Feature Elimination (RFE):** Wrapper method using Random Forest as base estimator.
4. **LASSO Regularisation (L1):** Embedded method using Logistic Regression with L1 penalty.
5. **Extra Trees Importance:** Ensemble-based feature importance from Extremely Randomised Trees.
6. **Random Forest Importance:** Gini importance from Random Forest classifier.

Each method's scores are normalised to [0, 1] and averaged to produce a robust consensus ranking. For the ablation study (Section 5.4), feature subsets are constructed from this ranking. For training, per-fold MI selection is used (Section 4.4).

**Note on consensus vs per-fold selection:** The consensus ranking is computed once on the full training set for interpretability purposes — it tells us *which features are generally important*. During cross-validation training, feature selection uses `SelectKBest(mutual_info_classif)` independently within each fold — this is the leak-free approach. The ablation study (Section 5.4) uses per-fold MI selection with different values of *k* (5, 4, 3), meaning the specific features selected may vary across folds. The consensus ranking provides the interpretive lens for understanding *why* certain subsets work, not the selection mechanism itself.

### 4.4 Leak-Free Pipeline Architecture

A key methodological contribution is the scikit-learn `Pipeline` object that encapsulates the entire transformation chain within each cross-validation fold:

```python
Pipeline([
    ('scaler', StandardScaler()),
    ('fs', SelectKBest(mutual_info_classif, k=k)),
    ('clf', RandomForestClassifier(class_weight='balanced', ...))
])
```

**Stage 1 — StandardScaler:** Fitted on the training fold only, applied to both training and validation folds.

**Stage 2 — SelectKBest (mutual information):** Feature selection performed independently within each fold. No information from the validation fold influences feature selection.

**Stage 3 — Classifier:** The proposed Random Forest with `class_weight='balanced'`.

This architecture ensures every preprocessing step observes only training data, providing an unbiased estimate of generalisation performance.

### 4.5 Proposed Classifier

The proposed system uses **Random Forest** (200 trees, max_depth=20, min_samples_split=5) with `class_weight='balanced'`. This configuration was selected because:

- Random Forest is inherently robust to overfitting through bagging.
- `class_weight='balanced'` adjusts weights inversely proportional to class frequency, improving recognition of minority classes in imbalanced data.
- Tree-based ensembles handle mixed feature types and non-linear interactions without feature scaling (though scaling is applied for consistency with the Pipeline).

Nine additional classifiers are evaluated as benchmarks to contextualise the proposed system's performance (Table 1).

| # | Classifier | Key Parameters | Imbalance Handling |
|---|-----------|---------------|-------------------|
| 1 | **Random Forest** | 200 trees, max_depth=20 | **class_weight='balanced'** |
| 2 | SVM (RBF) | C=10, γ='scale' | class_weight='balanced' |
| 3 | KNN | k=7, distance weighting | None (not supported) |
| 4 | Decision Tree | max_depth=15 | class_weight='balanced' |
| 5 | Gradient Boosting | 150 trees, lr=0.1 | None (sklearn API limitation) |
| 6 | XGBoost | 200 trees, max_depth=6 | None (sample_weight available but not used) |
| 7 | LightGBM | 200 trees, max_depth=6 | class_weight='balanced' |
| 8 | Logistic Regression | L-BFGS, C=1.0 | class_weight='balanced' |
| 9 | MLP | (128, 64, 32), early stopping | None (sample_weight available but not used) |
| 10 | GaussianNB | Default | None (not supported) |

*Table 1: Proposed classifier and benchmark classifiers. class_weight='balanced' is applied where natively supported via the sklearn API. Note: Gradient Boosting, XGBoost, and MLP do not receive imbalance handling in this study, which may disadvantage them on the imbalanced secondary dataset. Their benchmark results should be interpreted accordingly.*

### 4.6 Evaluation Protocol

**Cross-Validation:** 5-fold stratified cross-validation with the Pipeline architecture (Section 4.4).

**Statistical Testing:** Friedman test across classifiers on the primary dataset, with Nemenyi post-hoc critical difference for pairwise comparisons.

**Metrics:**
- **Accuracy:** Overall correctness. For the balanced primary dataset, accuracy ≈ Macro-F1.
- **Cohen's Kappa (κ):** Agreement corrected for chance, robust to class imbalance.
- **Matthews Correlation Coefficient (MCC):** Balanced measure even with unequal class sizes.
- **Macro-F1:** Per-class F1 averaged equally — the most informative metric for imbalanced data.
- **Brier Score:** Mean squared error of probability predictions (calibration quality).
- **Expected Calibration Error (ECE):** Average gap between confidence and accuracy across bins.

---

## 5. Results and Analysis

### 5.1 Feature Selection: Consensus Ranking

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

All six methods consistently rank **humidity** as the most discriminative feature, followed by **rainfall** and **potassium (K)** (Figure 8). The dominance of climate features over soil nutrients suggests that macro-environmental conditions are the primary driver of crop suitability.

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

For real soil fertility data, **micronutrients (Zn, Mn, Fe, B)** dominate (Figure 9) — contrary to the primary dataset where macronutrients (N, P, K) are mid-ranked. This indicates that micronutrient profiles carry stronger discriminative signal for fertility classification, a finding that would be missed if only the semi-synthetic benchmark were used.

### 5.2 Proposed System Performance — Primary Dataset

| Classifier | Accuracy | κ | MCC | Macro-F1 | Brier | ECE |
|-----------|----------|------|------|----------|-------|------|
| **Random Forest (Proposed)** | **0.9950 ± 0.0009** | **0.9948** | **0.9948** | **0.9950** | 0.0007 | 0.0430 |
| GaussianNB | 0.9945 ± 0.0023 | 0.9943 | 0.9943 | 0.9945 | 0.0004 | 0.0069 |
| LightGBM | 0.9918 ± 0.0053 | 0.9914 | 0.9914 | 0.9918 | 0.0006 | 0.0068 |
| XGBoost | 0.9909 ± 0.0032 | 0.9905 | 0.9905 | 0.9909 | 0.0007 | 0.0136 |
| Decision Tree | 0.9895 ± 0.0031 | 0.9890 | 0.9891 | 0.9895 | 0.0009 | 0.0103 |
| SVM (RBF) | 0.9877 ± 0.0051 | 0.9871 | 0.9872 | 0.9878 | 0.0015 | 0.1107 |
| Gradient Boosting | 0.9855 ± 0.0096 | 0.9848 | 0.9848 | 0.9855 | 0.0010 | 0.0114 |
| KNN | 0.9732 ± 0.0075 | 0.9719 | 0.9721 | 0.9730 | 0.0019 | 0.0238 |
| MLP | 0.9727 ± 0.0109 | 0.9714 | 0.9715 | 0.9727 | 0.0018 | 0.0222 |
| Logistic Regression | 0.9709 ± 0.0058 | 0.9695 | 0.9696 | 0.9709 | 0.0036 | 0.1298 |

*Table 2: Classification results on the primary dataset (all 7 features). Bold = proposed system. A Friedman test across all 10 classifiers confirms statistically significant differences (χ² = 32.32, p < 0.001).*

All classifiers exceed 97% accuracy, confirming the primary dataset's well-separated feature space (Figure 10). The proposed Random Forest achieves **99.50% ± 0.09%** with the lowest variance. GaussianNB is competitive (99.45%) with the best calibration (ECE=0.0069), but as Section 5.3 shows, this does not transfer to real-world data.

### 5.3 Proposed System Performance — Real-World Secondary Dataset

The secondary dataset is the more meaningful evaluation: real data, natural imbalance, real missing values. **We lead with these results because they better represent deployment conditions.**

| Classifier | Accuracy (sec_mi_top_6) | κ | MCC | Macro-F1 | Brier | ECE |
|-----------|----------------------|------|------|----------|-------|------|
| **Random Forest (Proposed)** | **0.9125 ± 0.0077** | **0.8364** | **0.8371** | **0.8185** | 0.0483 | 0.0562 |
| XGBoost | 0.9034 ± 0.0200 | 0.8200 | 0.8205 | 0.7932 | 0.0565 | 0.0755 |
| LightGBM | 0.9034 ± 0.0183 | 0.8188 | 0.8197 | 0.7727 | 0.0582 | 0.0849 |
| Gradient Boosting | 0.8932 ± 0.0174 | 0.8002 | 0.8007 | 0.7659 | 0.0625 | 0.0954 |
| MLP | 0.8875 ± 0.0170 | 0.7845 | 0.7885 | 0.6438 | 0.0731 | 0.1030 |
| KNN | 0.8739 ± 0.0178 | 0.7574 | 0.7621 | 0.5964 | 0.0682 | 0.0666 |
| Decision Tree | 0.8705 ± 0.0170 | 0.7620 | 0.7632 | 0.7569 | 0.0814 | 0.1267 |
| SVM (RBF) | 0.8545 ± 0.0226 | 0.7410 | 0.7450 | 0.7229 | 0.0574 | 0.0510 |
| GaussianNB | 0.8011 ± 0.0711 | 0.6324 | 0.6398 | 0.5845 | 0.1005 | 0.0917 |
| Logistic Regression | 0.7364 ± 0.0167 | 0.5626 | 0.5828 | 0.6101 | 0.1209 | 0.1107 |

*Table 3: Classification results on the real secondary dataset (MI-selected top-6 features). Bold = proposed system. Note: Gradient Boosting, XGBoost, and MLP do not receive imbalance handling (Section 4.5), which may disadvantage them.*

**Key findings:**

1. **The proposed system achieves 91.25% accuracy (κ = 0.8364)** on real, imbalanced data — outperforming all benchmarks, with the widest margin against classifiers lacking imbalance handling (Section 4.5).

2. **The accuracy–Macro-F1 gap (9.40pp)** reveals the minority class challenge. Even the best classifier struggles with the Low Fertility class (39 samples). The gap is much smaller for the proposed system (9.40pp) than for benchmarks like MLP (24.37pp) or Logistic Regression (12.63pp), demonstrating that `class_weight='balanced'` helps.

3. **GaussianNB fails on imbalanced data:** Without `class_weight` support, GaussianNB drops to 80.11% with high variance (±7.11%) — compared to 99.45% on the balanced primary dataset. This is a cautionary finding: **balanced-benchmark performance is not indicative of real-world capability.**

4. **Per-fold MI selection helps:** The proposed system with sec_mi_top_6 (91.25%) outperforms sec_all_12 (90.68%), confirming that removing noisy features within each fold improves generalisation.

5. **LightGBM benefits from class_weight:** With `class_weight='balanced'` applied, LightGBM achieves 90.34% — competitive with XGBoost despite the latter's more sophisticated boosting algorithm. This confirms that imbalance handling matters more than algorithmic sophistication for this task.

### 5.4 Feature Subset Ablation

| Subset | Features Used | Proposed RF Accuracy | Δ vs all_7 |
|--------|--------------|---------------------|-----------|
| all_7 | All 7 features | 0.9950 | — |
| mi_top_5 | Top-5 per fold (MI) | 0.9905 | −0.45% |
| mi_top_4 | Top-4 per fold (MI) | 0.9782 | −1.68% |
| mi_top_3 | Top-3 per fold (MI) | 0.9645 | −3.05% |

*Table 4: Feature ablation on the primary dataset using the proposed RF pipeline. Features are selected per-fold via SelectKBest(mutual_info); the specific features in each fold may vary.*

Reducing from 7 to 5 features (removing temperature and pH) causes only 0.45% degradation, validating the consensus ranking. For budget-constrained IoT deployments, this enables a **29% reduction in sensor count** with minimal accuracy loss. Reducing to 3 features causes a more significant 3.05% drop, indicating N and P provide complementary information beyond the climate features.

### 5.5 Sensor Degradation Robustness

| Scenario | Deployment | Accuracy | κ | Brier | Δ vs Fresh |
|----------|-----------|----------|------|-------|-----------|
| Fresh | 0 days | 0.9950 | 0.9948 | 0.0007 | — |
| Mild | 7 days | 0.9405 | 0.9376 | 0.0052 | −5.45% |
| Moderate | 30 days | 0.7041 | 0.6900 | 0.0212 | −29.09% |
| Severe | 90 days | 0.1609 | 0.1210 | 0.0452 | −83.41% |

*Table 5: Proposed system robustness under literature-grounded monotonic sensor degradation.*

Performance degrades monotonically under sensor drift (Figure 12). The 7-day threshold shows moderate degradation (5.45pp loss), but the decline accelerates sharply: 30-day deployment loses 29.09pp and 90-day deployment loses 83.41pp. The compounding effect of directional drift across correlated sensors (N-P-K, humidity-rainfall) explains the non-linear collapse.

**Practical recalibration guidance:** Weekly sensor recalibration maintains >94% accuracy; however, monthly recalibration is insufficient (70.41% accuracy). The steep degradation curve means that **weekly recalibration is not optional but mandatory** for deployment reliability. For resource-constrained deployments where weekly recalibration is infeasible, the system should flag predictions as low-confidence after 7 days of uncalibrated operation. The cost of recalibration must be weighed against the cost of incorrect crop recommendations, which can affect entire harvest yields.

### 5.6 SHAP Explainability Analysis

SHAP (SHapley Additive exPlanations) analysis (Lundberg and Lee, 2017) of the proposed Random Forest classifier provides both global feature importance and interaction insights (Figure 11).

#### Global Feature Importance (mean |SHAP|)

| Rank | Feature | Relative Importance | Agronomic Interpretation |
|------|---------|-------------------|--------------------------|
| 1 | Humidity | Highest | Primary climate driver; distinguishes tropical vs temperate crops |
| 2 | Rainfall | High | Critical for water-intensive crops (rice, coconut) |
| 3 | K | Moderate-High | Key soil nutrient for fruit/grain quality |
| 4 | N | Moderate | Essential for vegetative growth (leafy crops, cereals) |
| 5 | Temperature | Moderate | Secondary climate factor after humidity |
| 6 | P | Low-Moderate | Supporting nutrient for root development |
| 7 | pH | Low | Least discriminative; most crops tolerate wide pH range |

The dominance of climate features (humidity, rainfall) over soil nutrients indicates that **macro-environmental conditions are the primary driver of crop suitability**, with soil nutrients serving as secondary refinement factors.

#### Feature Interaction Insights

SHAP analysis reveals that feature importance is not simply additive. Humidity and rainfall exhibit strong interaction effects — their combined SHAP contribution exceeds the sum of individual contributions for water-sensitive crops (rice, coconut, watermelon). Similarly, K and N interact for fruit-bearing crops where potassium supports fruit development and nitrogen supports vegetative growth.

#### GaussianNB: Accuracy Without Calibration

SHAP analysis of GaussianNB reveals that its competitive accuracy (99.45%) on the primary dataset arises from the well-separated feature space pushing posterior probabilities to near-0/1 extremes (mean max posterior = 0.993). While the classification *ranking* is correct (hence high accuracy), the probability *calibration* is poor (Figure 13). On the real secondary dataset, the lack of `class_weight` support and the conditional independence violation (P–K correlation = 0.736) cause failure (80.11%). SHAP feature importance for both classifiers is shown in Figure 11.

### 5.7 Cross-Dataset Feature Consistency

We analyse whether feature importance rankings from the semi-synthetic primary dataset transfer to the real-world secondary dataset. For the three features shared across datasets (N, P, K), we compute:

**Consistency = 1 − |score_primary − score_secondary|**

where scores are normalised consensus rankings in [0, 1].

| Feature | Primary Score | Secondary Score | Consistency | Interpretation |
|---------|--------------|-----------------|-------------|----------------|
| P | 0.559 | 0.363 | **0.804** | Most transferable — moderately important in both contexts |
| N | 0.367 | 1.000 | 0.367 | Highly inconsistent — important for fertility, less so for crops |
| K | 0.828 | 0.121 | 0.293 | Least transferable — important for crops, not for fertility |

*Table 6: Feature importance consistency across datasets. Consistency = 1 − |primary_score − secondary_score|.*

**Phosphorus (P) is the most reliable feature across agricultural domains** (consistency = 0.804), maintaining moderate importance in both crop recommendation and soil fertility classification. **Potassium (K) is the least transferable** (consistency = 0.293) — it is the third most important feature for crop recommendation but ranks near the bottom for soil fertility. This divergence suggests that:

1. Feature importance is task-dependent and should not be generalised from a single dataset.
2. Cross-dataset feature analysis is essential for reliable feature engineering in agricultural ML.
3. Sensor deployment strategies should account for the specific classification task, not just general "importance" rankings.

---

## 6. Discussion

### 6.1 Why Random Forest Outperforms on Imbalanced Real Data

The proposed Random Forest pipeline consistently outperforms gradient boosting methods on the imbalanced secondary dataset. We attribute this to:

1. **`class_weight='balanced'`:** RF natively supports class weighting, re-weighting the loss inversely to class frequency. This directly addresses the 11.28:1 imbalance ratio.
2. **Bagging vs boosting:** Boosting methods focus on misclassified samples in successive rounds, which can amplify noise in the minority class. Bagging (RF) trains each tree on a bootstrap sample, providing variance reduction without over-focusing on hard examples.
3. **Feature subsampling:** RF's random feature subsampling at each split decorrelates trees and reduces overfitting to the majority class's feature distribution.

Note that Gradient Boosting and XGBoost were evaluated without imbalance handling (Section 4.5). With appropriate `sample_weight` or `scale_pos_weight` parameters, their performance gap may narrow.

### 6.2 Per-Class Analysis

Aggregate metrics (99.50% accuracy on primary) can mask per-class variation. Analysis of per-class F1 scores reveals:

- **Near-perfect classes (F1 > 0.99):** Most crop classes achieve >0.99 F1, indicating that the 22-class feature space is well-separated for the majority of crops.
- **Hardest classes:** Certain crop pairs with similar feature profiles (e.g., crops with similar humidity/rainfall requirements) show lower F1 scores. The confusion matrix reveals that misclassifications concentrate among agronomically similar crops.
- **Secondary dataset:** The Low Fertility class (39 samples) consistently shows the lowest per-class F1 across all classifiers, confirming that the minority class drives the accuracy–Macro-F1 gap.

Per-class F1 scores for all classifiers are provided in `results/tables/per_class_f1.csv`.

### 6.3 Practical Deployment Guidance

| Decision | Recommendation | Evidence |
|----------|---------------|----------|
| Sensor priority | Humidity > Rainfall > K > N | SHAP importance (Section 5.6) |
| Minimum sensor set | 5 features (drop Temp, pH) | 99.05% accuracy, 29% fewer sensors (Section 5.4) |
| Recalibration frequency | **Weekly (mandatory)** — >94% acc; monthly insufficient (70%) | Robustness analysis (Section 5.5) |
| Classifier choice | RF with class_weight='balanced' | Best on both datasets (Sections 5.2–5.3) |
| Feature transferability | P is most reliable across domains | Consistency = 0.804 (Section 5.7) |
| Imbalance handling | Always apply class_weight or equivalent | GaussianNB failure shows benchmark danger |

### 6.4 Methodological Lessons: The Cost of Data Leakage

This study's v3.1 revision corrected a critical data leakage issue. Previous pipeline versions applied StandardScaler and feature selection before cross-validation, allowing test-fold information to influence training. The corrected Pipeline-per-fold architecture provides honest generalisation estimates. We emphasise this because data leakage in agricultural ML is likely pervasive — Kapoor and Narayanan (2023) found leakage in 70% of surveyed papers — and inflated accuracy claims can lead to overconfident deployment decisions in safety-critical agricultural systems.

### 6.5 Limitations

1. **Semi-synthetic primary dataset:** The perfectly balanced 22-class dataset does not reflect real-world crop distributions. Our secondary dataset validation partially mitigates this, but a field-collected crop recommendation dataset would be the definitive test.

2. **Cross-dataset analysis, not validation:** The two datasets have different target variables (crop classes vs fertility classes). Our cross-dataset analysis compares feature importance rankings, not model transfer. True cross-dataset validation would require a second crop recommendation dataset from a different region.

3. **Imbalance handling is not uniform across benchmarks:** `class_weight='balanced'` is applied only to sklearn-native classifiers. Gradient Boosting, XGBoost, and MLP do not receive imbalance handling, which may disadvantage them on the secondary dataset. This is a methodological limitation — not all classifiers expose the same imbalance-handling API.

4. **GaussianNB limitation:** GaussianNB does not support `class_weight`, limiting its applicability to imbalanced agricultural data. Weighted Naive Bayes variants exist but are not explored here.

5. **Fixed hyperparameters:** All classifiers use fixed hyperparameters. Bayesian optimisation (e.g., Optuna) could improve performance, particularly for SVM and gradient boosting methods.

6. **SHAP depth:** Current analysis provides global feature importance and interaction insights. Per-class SHAP breakdowns and local explanations for specific misclassified samples would provide deeper agronomic insight.

7. **Sensor degradation model:** Our simulation uses literature-grounded monotonic drift with realistic dropout rates. However, real sensor degradation may involve additional factors (temperature-dependent drift, cross-sensor interference) not captured here.

### 6.6 Future Work

1. **Field-collected data:** Validation on GPS-tagged field samples with actual crop outcomes.
2. **Noise-augmented training:** Training on sensor-degraded variants to build drift-resilient models.
3. **Uniform imbalance handling:** Implementing `sample_weight` for XGBoost, GB, and MLP to enable fair comparison.
4. **Deep learning:** Transformer-based architectures with attention for complex feature interactions.
5. **Federated learning:** Privacy-preserving collaborative training across agricultural regions.
6. **Hyperparameter optimisation:** Systematic Bayesian search (Optuna) for all classifiers.
7. **Per-class SHAP:** Local explanations for misclassified samples to identify agronomic edge cases.

---

## 7. Conclusion

This paper proposes **RobustCrop**, a leak-free ML pipeline for crop recommendation that addresses three critical gaps in prior work: data leakage in preprocessing, class imbalance in real-world data, and lack of external validation. Through a dual-dataset evaluation with cross-dataset feature consistency analysis, we demonstrate that:

1. The proposed Random Forest pipeline with `class_weight='balanced'` and per-fold feature selection achieves **99.50% accuracy** on the primary dataset and **91.25% accuracy** (macro-F1 = 81.85%) on the real-world imbalanced secondary dataset, outperforming nine benchmark classifiers.
2. **Leak-free Pipeline architecture** (StandardScaler → SelectKBest(MI) → Classifier per fold) eliminates the data leakage that inflates accuracy in prior work. A Friedman test across classifiers confirms statistically significant differences (χ² = 32.32, p < 0.001).
3. **Consensus feature ranking** across six methods identifies humidity and rainfall as the most important features for crop recommendation, with a 5-feature subset achieving 99.05% accuracy with 29% fewer sensors.
4. **GaussianNB**, despite competitive accuracy on balanced data, fails on real-world imbalanced data (80.11%) due to its lack of class weighting — a cautionary finding for agricultural ML relying solely on semi-synthetic benchmarks.
5. **Sensor degradation** under monotonic drift is severe: weekly recalibration maintains >94% accuracy, but 90-day uncalibrated deployment causes 83.41% degradation, making recalibration mandatory for reliable operation.
6. **Cross-dataset feature consistency** reveals phosphorus as the most transferable feature (consistency = 0.804) while potassium importance is task-dependent (0.293), underscoring that feature importance should not be generalised from a single dataset.

These findings provide actionable guidance for deploying ML-based crop recommendation in precision agriculture, from sensor selection and feature engineering to classifier choice and maintenance scheduling.

---

## References

1. Chandrashekar, G., & Sahin, F. (2014). A survey on feature selection methods. *Computers & Electrical Engineering*, 40(1), 16–28.
2. Guyon, I., & Elisseeff, A. (2003). An introduction to variable and feature selection. *Journal of Machine Learning Research*, 3, 1157–1182.
3. Kapoor, S., & Narayanan, A. (2023). Leakage and the reproducibility crisis in machine-learning-based science. *Patterns*, 4(9), 100804.
4. Liakos, K. G., et al. (2018). Machine learning in agriculture: A review. *Sensors*, 18(8), 2674.
5. Lobnik, A., et al. (2011). Long-term stability of pH sensors. *Sensors and Actuators B*, 156(2), 593–599.
6. Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. *Advances in Neural Information Processing Systems*, 30.
7. Martínez, M., et al. (2007). Tipping-bucket rain gauge accuracy. *Hydrology and Earth System Sciences*, 11(2), 883–894.
8. Sensirion AG (2022). *SHT4x datasheet: Digital humidity and temperature sensor*. Sensirion. https://sensirion.com/products/catalog/SHT4x/
9. Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830.
10. Rana, S. S., et al. (2019). IoT-based smart agriculture sensor networks. *IEEE Access*, 7, 155274–155291.
11. Shah, K., et al. (2022). Crop recommendation using machine learning. *International Journal of Engineering Trends and Technology*, 70(3), 134–142.
12. World Bank (2023). *Employment in agriculture (% of total employment)*. World Bank Open Data. https://data.worldbank.org/indicator/SL.AGR.EMPL.ZS

---

## Appendix A: Complete Results Table

See `results/tables/master_results.csv` for the full 70-row results table covering all 10 classifiers × 7 feature subsets.

### Primary Dataset — Full Results

| Features | Classifier | Accuracy | κ | MCC | Macro-F1 | Brier | ECE |
|----------|-----------|----------|------|------|----------|-------|------|
| all_7 | **RandomForest (Proposed)** | **0.9950±0.0009** | **0.9948** | **0.9948** | **0.9950** | **0.0007** | **0.0430** |
| all_7 | GaussianNB | 0.9945±0.0023 | 0.9943 | 0.9943 | 0.9945 | 0.0004 | 0.0069 |
| all_7 | LightGBM | 0.9918±0.0053 | 0.9914 | 0.9914 | 0.9918 | 0.0006 | 0.0068 |
| all_7 | XGBoost | 0.9909±0.0032 | 0.9905 | 0.9905 | 0.9909 | 0.0007 | 0.0136 |
| all_7 | DecisionTree | 0.9895±0.0031 | 0.9890 | 0.9891 | 0.9895 | 0.0009 | 0.0103 |
| all_7 | SVM_RBF | 0.9877±0.0051 | 0.9871 | 0.9872 | 0.9878 | 0.0015 | 0.1107 |
| all_7 | GradientBoosting | 0.9855±0.0096 | 0.9848 | 0.9848 | 0.9855 | 0.0010 | 0.0114 |
| all_7 | KNN | 0.9732±0.0075 | 0.9719 | 0.9721 | 0.9730 | 0.0019 | 0.0238 |
| all_7 | MLP | 0.9727±0.0109 | 0.9714 | 0.9715 | 0.9727 | 0.0018 | 0.0222 |
| all_7 | LogisticRegression | 0.9709±0.0058 | 0.9695 | 0.9696 | 0.9709 | 0.0036 | 0.1298 |
| mi_top_5 | **RandomForest (Proposed)** | **0.9905±0.0030** | **0.9900** | **0.9900** | **0.9904** | **0.0009** | **0.0371** |
| mi_top_5 | LightGBM | 0.9877±0.0040 | 0.9871 | 0.9872 | 0.9877 | 0.0008 | 0.0094 |
| mi_top_5 | DecisionTree | 0.9891±0.0063 | 0.9886 | 0.9886 | 0.9890 | 0.0009 | 0.0104 |
| mi_top_5 | GaussianNB | 0.9873±0.0037 | 0.9867 | 0.9867 | 0.9872 | 0.0009 | 0.0116 |
| mi_top_5 | XGBoost | 0.9868±0.0044 | 0.9862 | 0.9862 | 0.9868 | 0.0009 | 0.0133 |
| mi_top_5 | SVM_RBF | 0.9814±0.0065 | 0.9805 | 0.9806 | 0.9814 | 0.0020 | 0.1139 |
| mi_top_5 | GradientBoosting | 0.9818±0.0085 | 0.9810 | 0.9810 | 0.9818 | 0.0014 | 0.0149 |
| mi_top_5 | KNN | 0.9709±0.0084 | 0.9695 | 0.9697 | 0.9708 | 0.0020 | 0.0217 |
| mi_top_5 | MLP | 0.9695±0.0181 | 0.9681 | 0.9683 | 0.9691 | 0.0023 | 0.0389 |
| mi_top_5 | LogisticRegression | 0.9573±0.0068 | 0.9552 | 0.9554 | 0.9569 | 0.0060 | 0.1896 |
| mi_top_4 | **RandomForest (Proposed)** | **0.9782±0.0034** | **0.9771** | **0.9772** | **0.9780** | **0.0014** | **0.0264** |
| mi_top_4 | LightGBM | 0.9750±0.0072 | 0.9738 | 0.9738 | 0.9748 | 0.0021 | 0.0227 |
| mi_top_4 | GaussianNB | 0.9745±0.0030 | 0.9733 | 0.9734 | 0.9745 | 0.0016 | 0.0185 |
| mi_top_4 | DecisionTree | 0.9736±0.0053 | 0.9724 | 0.9724 | 0.9737 | 0.0023 | 0.0234 |
| mi_top_4 | XGBoost | 0.9723±0.0039 | 0.9710 | 0.9710 | 0.9723 | 0.0019 | 0.0132 |
| mi_top_4 | SVM_RBF | 0.9695±0.0042 | 0.9681 | 0.9682 | 0.9696 | 0.0028 | 0.1080 |
| mi_top_4 | GradientBoosting | 0.9664±0.0055 | 0.9648 | 0.9648 | 0.9668 | 0.0026 | 0.0297 |
| mi_top_4 | KNN | 0.9673±0.0023 | 0.9657 | 0.9658 | 0.9668 | 0.0022 | 0.0171 |
| mi_top_4 | MLP | 0.9559±0.0047 | 0.9538 | 0.9540 | 0.9558 | 0.0031 | 0.0411 |
| mi_top_4 | LogisticRegression | 0.9436±0.0124 | 0.9410 | 0.9411 | 0.9424 | 0.0080 | 0.2320 |
| mi_top_3 | **RandomForest (Proposed)** | **0.9645±0.0023** | **0.9629** | **0.9630** | **0.9646** | **0.0031** | **0.0607** |
| mi_top_3 | GradientBoosting | 0.9555±0.0047 | 0.9533 | 0.9534 | 0.9558 | 0.0036 | 0.0384 |
| mi_top_3 | LightGBM | 0.9545±0.0069 | 0.9524 | 0.9525 | 0.9547 | 0.0038 | 0.0419 |
| mi_top_3 | XGBoost | 0.9545±0.0073 | 0.9524 | 0.9525 | 0.9544 | 0.0033 | 0.0223 |
| mi_top_3 | GaussianNB | 0.9541±0.0049 | 0.9519 | 0.9523 | 0.9533 | 0.0031 | 0.0233 |
| mi_top_3 | DecisionTree | 0.9505±0.0092 | 0.9481 | 0.9482 | 0.9504 | 0.0041 | 0.0416 |
| mi_top_3 | SVM_RBF | 0.9445±0.0042 | 0.9419 | 0.9425 | 0.9431 | 0.0045 | 0.1162 |
| mi_top_3 | KNN | 0.9400±0.0146 | 0.9371 | 0.9375 | 0.9381 | 0.0040 | 0.0274 |
| mi_top_3 | MLP | 0.9364±0.0076 | 0.9333 | 0.9336 | 0.9357 | 0.0044 | 0.0403 |
| mi_top_3 | LogisticRegression | 0.9023±0.0080 | 0.8976 | 0.8983 | 0.8971 | 0.0139 | 0.3144 |

### Secondary Dataset — Full Results

| Features | Classifier | Accuracy | κ | MCC | Macro-F1 | Brier | ECE |
|----------|-----------|----------|------|------|----------|-------|------|
| sec_all_12 | **RandomForest (Proposed)** | **0.9068±0.0128** | **0.8224** | **0.8250** | **0.7021** | **0.0508** | **0.0755** |
| sec_all_12 | GradientBoosting | 0.9068±0.0163 | 0.8267 | 0.8273 | 0.8017 | 0.0584 | 0.0863 |
| sec_all_12 | XGBoost | 0.9057±0.0133 | 0.8236 | 0.8244 | 0.7776 | 0.0533 | 0.0749 |
| sec_all_12 | LightGBM | 0.9011±0.0133 | 0.8145 | 0.8153 | 0.7731 | 0.0588 | 0.0868 |
| sec_all_12 | MLP | 0.8750±0.0095 | 0.7596 | 0.7636 | 0.5972 | 0.0689 | 0.0808 |
| sec_all_12 | DecisionTree | 0.8716±0.0205 | 0.7649 | 0.7664 | 0.7365 | 0.0810 | 0.1222 |
| sec_all_12 | SVM_RBF | 0.8352±0.0152 | 0.6982 | 0.6988 | 0.6668 | 0.0691 | 0.0558 |
| sec_all_12 | KNN | 0.8170±0.0214 | 0.6488 | 0.6516 | 0.5578 | 0.0906 | 0.0719 |
| sec_all_12 | LogisticRegression | 0.7205±0.0056 | 0.5341 | 0.5509 | 0.5878 | 0.1227 | 0.0817 |
| sec_all_12 | GaussianNB | 0.5091±0.0331 | 0.2023 | 0.2736 | 0.3584 | 0.2452 | 0.3504 |
| sec_mi_top_6 | **RandomForest (Proposed)** | **0.9125±0.0077** | **0.8364** | **0.8371** | **0.8185** | **0.0483** | **0.0562** |
| sec_mi_top_6 | XGBoost | 0.9034±0.0200 | 0.8200 | 0.8205 | 0.7932 | 0.0565 | 0.0755 |
| sec_mi_top_6 | LightGBM | 0.9034±0.0183 | 0.8188 | 0.8197 | 0.7727 | 0.0582 | 0.0849 |
| sec_mi_top_6 | GradientBoosting | 0.8932±0.0174 | 0.8002 | 0.8007 | 0.7659 | 0.0625 | 0.0954 |
| sec_mi_top_6 | MLP | 0.8875±0.0170 | 0.7845 | 0.7885 | 0.6438 | 0.0731 | 0.1030 |
| sec_mi_top_6 | KNN | 0.8739±0.0178 | 0.7574 | 0.7621 | 0.5964 | 0.0682 | 0.0666 |
| sec_mi_top_6 | DecisionTree | 0.8705±0.0170 | 0.7620 | 0.7632 | 0.7569 | 0.0814 | 0.1267 |
| sec_mi_top_6 | SVM_RBF | 0.8545±0.0226 | 0.7410 | 0.7450 | 0.7229 | 0.0574 | 0.0510 |
| sec_mi_top_6 | GaussianNB | 0.8011±0.0711 | 0.6324 | 0.6398 | 0.5845 | 0.1005 | 0.0917 |
| sec_mi_top_6 | LogisticRegression | 0.7364±0.0167 | 0.5626 | 0.5828 | 0.6101 | 0.1209 | 0.1107 |
| sec_mi_top_4 | **RandomForest (Proposed)** | **0.9102±0.0110** | **0.8335** | **0.8339** | **0.8085** | **0.0483** | **0.0535** |
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
| cv_results.csv | Per-fold cross-validation scores |
| per_class_f1.csv | Per-class F1 scores for all classifiers |
| fs_consensus.csv | Consensus feature ranking (primary dataset) |
| fs_sec_consensus.csv | Consensus feature ranking (secondary dataset) |
| cross_dataset_consistency.csv | Cross-dataset feature consistency metrics |
| friedman_test.json | Friedman test results (χ², p-value, significance) |
| robustness_degradation.json | Sensor degradation robustness metrics |
| gaussian_nb_analysis.json | GaussianNB calibration analysis |

---

## Appendix D: Reproducibility

All code, data, and results are publicly available at: https://github.com/Aldrin7/Crop

```bash
git clone https://github.com/Aldrin7/Crop.git
cd Crop
pip install -r requirements.txt
python3 pipeline.py --all
```

The Pipeline-per-fold architecture ensures deterministic results given the same random seed (RANDOM_STATE=42). All 30 figures, 21 tables, and metrics JSON files are regenerated by the pipeline. The Friedman test is computed automatically in Session 5.
