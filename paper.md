---
title: "RobustCrop: A Leak-Free Machine Learning Pipeline for Crop Recommendation with Sensor Degradation Analysis and Cross-Dataset Feature Consistency"
author:
  - name: Anuradha Brijwal
    affiliation: "1"
    corresponding: true
    email: anuradha.brijwal@gurukul.edu.in
  - name: Praveena Chaturvedi
    affiliation: "1"
affiliation:
  - id: 1
    name: "Department of Computer Science, Kanya Gurukul Campus Dehradun, Gurukul Kangri (Deemed to be University), Haridwar, Uttarakhand, India"
date: "2026"
journal: "Preprint"
keywords: "Crop Recommendation, Feature Selection, Soil Nutrients, Precision Agriculture, Machine Learning, Sensor Degradation, SHAP Explainability, Cross-Dataset Analysis, Class Imbalance, Hyperparameter Tuning"
abstract: |
  Deploying machine learning for crop recommendation in real agricultural settings requires addressing three practical challenges overlooked by prior work: data leakage in preprocessing pipelines, class imbalance in real-world soil data, and over-reliance on semi-synthetic benchmarks without external validation. This paper proposes RobustCrop, a leak-free pipeline that encapsulates feature scaling, mutual-information-based feature selection, and classification within a single scikit-learn Pipeline per cross-validation fold, eliminating the information leakage that inflates accuracy in prior studies. A BalWeightWrapper provides sample_weight='balanced' imbalance correction for all classifiers --- including those without native class_weight support (XGBoost, Gradient Boosting, MLP, KNN, GaussianNB) --- ensuring fair comparison. Optional Optuna-based nested CV (30 trials, 3-fold inner loop) enables systematic hyperparameter tuning. We evaluate on two datasets: a primary crop recommendation dataset (2,200 samples, 22 semi-synthetic classes) and a real-world soil fertility dataset (880 samples, 3 classes, 11.28:1 imbalance ratio). On real-world data, the proposed Random Forest pipeline achieves 91.25% ± 0.77% accuracy (κ = 0.8364, macro-F1 = 81.85%), outperforming nine benchmark classifiers. SHAP analysis identifies humidity and rainfall as dominant predictors, with potassium and nitrogen as key soil nutrient differentiators. Literature-grounded sensor degradation analysis under monotonic drift shows decay from 94.05% (7-day) to 16.09% (90-day drift), establishing that weekly recalibration maintains >94% accuracy. Cross-dataset feature consistency analysis reveals phosphorus as the most transferable feature (consistency = 0.804) while potassium importance is task-dependent (0.293). A Friedman test across classifiers confirms statistically significant differences (χ² = 32.32, p < 0.001). These findings provide actionable deployment guidance for ML-based crop recommendation in resource-constrained agricultural IoT settings.
geometry: "margin=1in"
fontsize: 11pt
linestretch: 1.5
numbersections: true
toc: true
---

# Introduction

## Background

Agriculture accounts for approximately 4% of global GDP and employs over 25% of the world's population [@WorldBank2023]. Selecting appropriate crops based on soil properties and climatic conditions directly influences yield, resource efficiency, and farmer livelihood. Machine learning (ML) offers data-driven crop recommendation by learning from integrated soil-climate datasets. However, deploying ML in agricultural settings introduces three practical challenges that prior work has inadequately addressed:

1. **Data leakage in evaluation:** Standard preprocessing pipelines apply feature scaling and selection before cross-validation splits, allowing test-fold information to influence training. Kapoor and Narayanan [-@Kapoor2023] documented such leakage in 70% of surveyed ML-based science papers, with performance inflation of 5--30%.

2. **Class imbalance in real data:** Real-world soil fertility datasets exhibit natural class imbalance (e.g., 11.28:1 in our secondary dataset), yet most classifiers are evaluated without imbalance-aware training, and semi-synthetic benchmarks with perfect balance mask this issue.

3. **Lack of external validation:** Nearly all crop recommendation studies rely on a single semi-synthetic benchmark without validating on independent real-world data or analysing whether feature importance rankings transfer across datasets.

## Contributions

This paper proposes **RobustCrop**, a leak-free ML pipeline for crop recommendation. Our contributions are:

- A **Pipeline-per-fold architecture** (scikit-learn Pipeline: StandardScaler → SelectKBest(mutual information) → Classifier) that eliminates data leakage by ensuring all preprocessing occurs independently within each cross-validation fold.
- **BalWeightWrapper for uniform imbalance handling:** A `sample_weight='balanced'` wrapper that provides imbalance correction for *all* classifiers — including those without native `class_weight` support (XGBoost, Gradient Boosting, MLP, KNN, GaussianNB).
- **Optuna nested CV hyperparameter tuning** (optional) with classifier-specific search spaces, 30 trials per fold, and 3-fold inner CV — replacing fixed hyperparameters with systematic Bayesian optimisation.
- **Dual-dataset evaluation** using both a semi-synthetic crop recommendation dataset and a real-world soil fertility dataset.
- **Consensus feature ranking** across six methods (Mutual Information, Chi-Square, RFE, LASSO, Extra Trees, Random Forest Importance) with per-fold MI selection during training.
- **SHAP-based model explainability** providing global feature importance and interaction insights.
- **Literature-grounded sensor degradation analysis** quantifying performance under realistic deployment conditions (7--90 day monotonic drift).
- **Cross-dataset feature consistency analysis** identifying which features rank reliably across datasets.

# Related Work

## Machine Learning for Crop Recommendation

ML for crop recommendation has been widely explored. Liakos et al. [-@Liakos2018] reviewed ML in agriculture, noting Random Forest and SVM as consistently strong performers. Shah et al. [-@Shah2022] achieved 99.1% accuracy on the Kaggle Crop Recommendation dataset using ensemble classifiers. However, most studies report only accuracy on a single dataset without addressing calibration, class imbalance, or data leakage in preprocessing — issues that our work directly confronts. Kamilaris and Prenafeta-Boldú [-@Kamilaris2018] provided a comprehensive survey of deep learning in agriculture, highlighting the growing adoption of ML techniques across the agricultural domain.

## Feature Selection in Agricultural Data

Feature selection reduces dimensionality and improves interpretability. Guyon and Elisseeff [-@Guyon2003] categorise methods into filter, wrapper, and embedded approaches. In agricultural contexts, filter methods (Mutual Information, Chi-Square) are computationally efficient, while embedded methods (LASSO, tree-based importance) balance accuracy and speed [@Chandrashekar2014]. Our work synthesises six methods into a consensus ranking and analyses feature importance transferability across datasets.

## Data Leakage in ML Pipelines

Data leakage — where test information inadvertently influences training — is a pervasive but under-reported issue in applied ML. Kapoor and Narayanan [-@Kapoor2023] documented leakage in 229 out of 329 papers across 17 application domains. Common leakage sources include feature scaling before train-test splitting and feature selection on the full dataset before cross-validation. Our Pipeline-per-fold architecture directly addresses these leakage vectors.

## Sensor Reliability in Agricultural IoT

The integration of IoT sensors with machine learning for precision agriculture is reviewed by Wolfert et al. [-@Wolfert2017], who identify data quality and sensor reliability as key challenges. Real-world agricultural sensing introduces noise and drift. Rana et al. [-@Rana2019] document electrochemical NPK sensor drift rates of 1--1.5% per day, while Lobnik et al. [-@Lobnik2011] report pH electrode drift of 0.1% per day. Sensor drift is typically monotonic and directional — electrochemical sensors lose sensitivity over time.

# Datasets

## Primary Dataset: Crop Recommendation (Semi-Synthetic)

The primary dataset (Table 1) contains 7 soil and climate features with 22 balanced crop classes. The perfect class balance (100 samples/class) reflects the semi-synthetic augmentation process rather than natural field distributions.

| Property | Value |
|:---------|:------|
| Source | Kaggle (Atharva Ingle) |
| Samples | 2,200 |
| Features | N, P, K, Temperature, Humidity, pH, Rainfall (7) |
| Classes | 22 crop types (100 samples each) |
| Nature | Semi-synthetic (augmented from Indian agricultural statistics) |

: Primary dataset properties. {#tbl:primary}

## Secondary Dataset: Soil Fertility (Real Lab Measurements)

The secondary dataset (Table 2) comprises actual soil laboratory test results from Indian agricultural testing centres. Its natural class imbalance and real missing values provide a realistic test surface.

| Property | Value |
|:---------|:------|
| Source | Kaggle (Rahul Jaiswal) |
| Samples | 880 |
| Features | N, P, K, pH, EC, OC, S, Zn, Fe, Cu, Mn, B (12) |
| Classes | 3 fertility levels (High, Medium, Low) |
| Nature | **Real soil laboratory test results** |
| Imbalance | 401 / 440 / 39 (ratio 11.28:1) |
| Missing | ~3% real sensor/lab dropout |

: Secondary dataset properties. {#tbl:secondary}

## Shared Feature Space

Both datasets share N, P, K as common features, with pH as a close match. This shared space enables cross-dataset feature consistency analysis (Section 5.7).

## Sensor Degradation Variants

To simulate realistic deployment conditions, we generate degraded variants using literature-grounded **monotonic drift** parameters (Table 3).

| Sensor | Drift (%/day) | Noise ($\sigma$) | Source |
|:-------|:-------------|:-----------------|:-------|
| N (Nitrogen) | 1.0 | 2.0 | Rana et al. (2019) |
| P (Phosphorus) | 1.5 | 1.5 | Rana et al. (2019) |
| K (Potassium) | 1.2 | 1.5 | Rana et al. (2019) |
| Temperature | 0.2 | 0.5 | Sensirion SHT4x (2022) |
| Humidity | 0.5 | 1.0 | Sensirion SHT4x (2022) |
| pH | 0.1 | 0.1 | Lobnik et al. (2011) |
| Rainfall | 0.3 | 5.0 | Martínez et al. (2007) |

: Sensor degradation parameters used for monotonic drift simulation. {#tbl:sensor}

Three degradation scenarios are generated: mild (7-day), moderate (30-day), and severe (90-day) deployments, with realistic dropout rates (2--10% scaled to deployment duration).

# Proposed Method: RobustCrop Pipeline

## System Overview

RobustCrop is a modular, leak-free ML pipeline for crop recommendation, implemented using scikit-learn [@Pedregosa2011]. The key design principle is that **every preprocessing step observes only training data within each cross-validation fold**. The system architecture is illustrated in Figure 1.

![System architecture of the RobustCrop pipeline showing the leak-free per-fold design.](01_primary_distributions.png){#fig:arch width="100%"}

## Data Preprocessing

**Missing Value Handling:** Median imputation is applied to all numerical features. For the primary dataset, no missing values are present; for the secondary dataset, ~3% real missing values are imputed.

**Outlier Detection:** IQR-based outlier detection identifies outliers without removal, preserving real-world variance. Outliers are detected in P (6.27%), K (9.09%), and other features of the primary dataset.

**Feature Scaling:** StandardScaler is applied within each cross-validation fold as part of the Pipeline (Section 4.4), fitted exclusively on the training fold.

## Feature Selection: Consensus Ranking

Six feature selection methods are evaluated to build a consensus ranking for interpretability:

1. **Mutual Information (MI):** Non-parametric measure of statistical dependence between features and target.
2. **Chi-Square ($\chi^2$):** Tests independence between categorical feature bins and target classes.
3. **Recursive Feature Elimination (RFE):** Wrapper method using Random Forest as base estimator.
4. **LASSO Regularisation (L1):** Embedded method using Logistic Regression with L1 penalty.
5. **Extra Trees Importance:** Ensemble-based feature importance from Extremely Randomised Trees.
6. **Random Forest Importance:** Gini importance from Random Forest classifier.

Each method's scores are normalised to $[0, 1]$ and averaged to produce a robust consensus ranking:

$$\text{Consensus}(f) = \frac{1}{M} \sum_{m=1}^{M} \hat{s}_m(f)$$ {#eq:consensus}

where $\hat{s}_m(f)$ is the normalised score of feature $f$ under method $m$, and $M = 6$ is the number of feature selection methods.

## Leak-Free Pipeline Architecture

A key methodological contribution is the scikit-learn Pipeline object that encapsulates the entire transformation chain within each cross-validation fold:

**Stage 1 — StandardScaler:** Fitted on the training fold only, applied to both training and validation folds.

**Stage 2 — SelectKBest (mutual information):** Feature selection performed independently within each fold.

**Stage 3 — Classifier:** The proposed Random Forest with `class_weight='balanced'`.

This architecture ensures every preprocessing step observes only training data, providing an unbiased estimate of generalisation performance.

## Proposed Classifier

The proposed system uses **Random Forest** [@Breiman2001] with 200 trees, max_depth=20, min_samples_split=5, and `class_weight='balanced'`. This configuration was selected because:

- Random Forest is inherently robust to overfitting through bagging.
- `class_weight='balanced'` adjusts weights inversely proportional to class frequency:

$$w_j = \frac{n}{k \cdot n_j}$$ {#eq:weight}

where $n$ is the total number of samples, $k$ is the number of classes, and $n_j$ is the number of samples in class $j$.

Nine additional classifiers are evaluated as benchmarks (Table 4).

| # | Classifier | Key Parameters | Imbalance Handling |
|:--|:-----------|:---------------|:-------------------|
| 1 | **Random Forest** | 200 trees, max_depth=20 | **class_weight='balanced'** |
| 2 | SVM (RBF) | C=10, $\gamma$='scale' | class_weight='balanced' |
| 3 | KNN | k=7, distance weighting | BalWeightWrapper |
| 4 | Decision Tree | max_depth=15 | class_weight='balanced' |
| 5 | Gradient Boosting | 150 trees, lr=0.1 | BalWeightWrapper |
| 6 | XGBoost | 200 trees, max_depth=6 | BalWeightWrapper |
| 7 | LightGBM | 200 trees, max_depth=6 | class_weight='balanced' |
| 8 | Logistic Regression | L-BFGS, C=1.0 | class_weight='balanced' |
| 9 | MLP | (128, 64, 32), early stopping | BalWeightWrapper |
| 10 | GaussianNB | Default | BalWeightWrapper |

: Proposed classifier and benchmark classifiers with their imbalance handling mechanisms. {#tbl:classifiers}

## Evaluation Protocol

**Cross-Validation:** 5-fold stratified cross-validation [@Kohavi1995] with the Pipeline architecture.

**Statistical Testing:** Friedman test [@Friedman1937] across classifiers on the primary dataset.

**Metrics:** We report the following evaluation metrics:

- **Accuracy:** Overall correctness, defined as $\text{Acc} = \frac{1}{n}\sum_{i=1}^{n} \mathbb{1}[\hat{y}_i = y_i]$
- **Cohen's Kappa ($\kappa$):** Agreement corrected for chance: $\kappa = \frac{p_o - p_e}{1 - p_e}$
- **Matthews Correlation Coefficient (MCC):**

$$\text{MCC} = \frac{TP \cdot TN - FP \cdot FN}{\sqrt{(TP+FP)(TP+FN)(TN+FP)(TN+FN)}}$$ {#eq:mcc}

- **Macro-F1:** Per-class F1 averaged equally — the most informative metric for imbalanced data.
- **Brier Score:** Mean squared error of probability predictions [@Guo2017].
- **Expected Calibration Error (ECE):** Average gap between confidence and accuracy across bins.

## BalWeightWrapper: Uniform Imbalance Handling

We resolve the imbalance handling inconsistency with **BalWeightWrapper**, a lightweight wrapper that:

1. Computes `sample_weight='balanced'` from the training labels at `fit()` time.
2. Inspects the inner estimator's `fit()` signature; if it accepts `sample_weight`, passes the weights.
3. Delegates `predict()`, `predict_proba()`, and attribute access to the inner estimator.

The wrapper is transparent to the Pipeline — it behaves exactly like any other sklearn estimator.

## Hyperparameter Tuning via Optuna (Optional)

We integrate **Optuna** [@Akiba2019] for optional Bayesian hyperparameter tuning via nested CV:

- **Outer loop:** 5-fold stratified CV for unbiased evaluation.
- **Inner loop:** 3-fold CV on each training fold, with 30 Optuna trials using the TPE sampler.
- **Search spaces** are classifier-specific: RF ($n_{\text{estimators}}$, $d_{\text{max}}$, $s_{\text{min-split}}$), SVM ($C$, $\gamma$ on log scale), KNN ($k$, weights), etc.

# Results and Analysis

## Feature Selection: Consensus Ranking

### Primary Dataset (7 features)

| Rank | Feature | Consensus Score |
|:-----|:--------|:---------------|
| 1 | Humidity | 0.96 |
| 2 | Rainfall | 0.83 |
| 3 | K | 0.79 |
| 4 | N | 0.60 |
| 5 | P | 0.56 |
| 6 | Temperature | 0.46 |
| 7 | pH | 0.36 |

: Consensus feature ranking for the primary dataset. {#tbl:fs_primary}

All six methods consistently rank **humidity** as the most discriminative feature, followed by **rainfall** and **potassium (K)**. The dominance of climate features over soil nutrients suggests that macro-environmental conditions are the primary driver of crop suitability.

### Secondary Dataset (12 features)

| Rank | Feature | Consensus Score |
|:-----|:--------|:---------------|
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

: Consensus feature ranking for the secondary dataset. {#tbl:fs_secondary}

For real soil fertility data, **micronutrients (Zn, Mn, Fe, B)** dominate — contrary to the primary dataset where macronutrients (N, P, K) are mid-ranked.

![Feature distributions of the primary (semi-synthetic) crop recommendation dataset.](01_primary_distributions.png){#fig:primary_dist width="100%"}

![Feature distributions of the secondary (real) soil fertility dataset.](02_secondary_distributions.png){#fig:secondary_dist width="100%"}

## Proposed System Performance — Primary Dataset

| Classifier | Accuracy | $\kappa$ | MCC | Macro-F1 | Brier | ECE |
|:-----------|:---------|:---------|:----|:---------|:------|:----|
| **RF (Proposed)** | **0.9950 ± 0.0009** | **0.9948** | **0.9948** | **0.9950** | 0.0007 | 0.0430 |
| GaussianNB | 0.9945 ± 0.0023 | 0.9943 | 0.9943 | 0.9945 | 0.0004 | 0.0069 |
| LightGBM | 0.9918 ± 0.0053 | 0.9914 | 0.9914 | 0.9918 | 0.0006 | 0.0068 |
| XGBoost | 0.9909 ± 0.0032 | 0.9905 | 0.9905 | 0.9909 | 0.0007 | 0.0136 |
| DT | 0.9895 ± 0.0031 | 0.9890 | 0.9891 | 0.9895 | 0.0009 | 0.0103 |
| SVM (RBF) | 0.9877 ± 0.0051 | 0.9871 | 0.9872 | 0.9878 | 0.0015 | 0.1107 |
| GB | 0.9855 ± 0.0096 | 0.9848 | 0.9848 | 0.9855 | 0.0010 | 0.0114 |
| KNN | 0.9732 ± 0.0075 | 0.9719 | 0.9721 | 0.9730 | 0.0019 | 0.0238 |
| MLP | 0.9727 ± 0.0109 | 0.9714 | 0.9715 | 0.9727 | 0.0018 | 0.0222 |
| LR | 0.9709 ± 0.0058 | 0.9695 | 0.9696 | 0.9709 | 0.0036 | 0.1298 |

: Classification results on the primary dataset (all 7 features). Bold = proposed system. Friedman test: $\chi^2 = 32.32$, $p < 0.001$. {#tbl:primary_results}

![Cross-validation comparison across classifiers on both datasets.](10_cv_comparison.png){#fig:cv width="100%"}

## Proposed System Performance — Real-World Secondary Dataset

The secondary dataset is the more meaningful evaluation: real data, natural imbalance, real missing values.

| Classifier | Accuracy | $\kappa$ | MCC | Macro-F1 | Brier | ECE |
|:-----------|:---------|:---------|:----|:---------|:------|:----|
| **RF (Proposed)** | **0.9125 ± 0.0077** | **0.8364** | **0.8371** | **0.8185** | 0.0483 | 0.0562 |
| XGBoost | 0.9034 ± 0.0200 | 0.8200 | 0.8205 | 0.7932 | 0.0565 | 0.0755 |
| LightGBM | 0.9034 ± 0.0183 | 0.8188 | 0.8197 | 0.7727 | 0.0582 | 0.0849 |
| GB | 0.8932 ± 0.0174 | 0.8002 | 0.8007 | 0.7659 | 0.0625 | 0.0954 |
| MLP | 0.8875 ± 0.0170 | 0.7845 | 0.7885 | 0.6438 | 0.0731 | 0.1030 |
| KNN | 0.8739 ± 0.0178 | 0.7574 | 0.7621 | 0.5964 | 0.0682 | 0.0666 |
| DT | 0.8705 ± 0.0170 | 0.7620 | 0.7632 | 0.7569 | 0.0814 | 0.1267 |
| SVM (RBF) | 0.8545 ± 0.0226 | 0.7410 | 0.7450 | 0.7229 | 0.0574 | 0.0510 |
| GaussianNB | 0.8011 ± 0.0711 | 0.6324 | 0.6398 | 0.5845 | 0.1005 | 0.0917 |
| LR | 0.7364 ± 0.0167 | 0.5626 | 0.5828 | 0.6101 | 0.1209 | 0.1107 |

: Classification results on the real secondary dataset (MI-selected top-6 features). Bold = proposed system. {#tbl:secondary_results}

**Key findings:**

1. The proposed system achieves **91.25% accuracy** ($\kappa$ = 0.8364) on real, imbalanced data — outperforming all benchmarks.
2. The accuracy–Macro-F1 gap (9.40pp) reveals the minority class challenge. The gap is much smaller for the proposed system than for benchmarks like MLP (24.37pp).
3. **GaussianNB fails on imbalanced data:** drops to 80.11% with high variance (±7.11%) — compared to 99.45% on the balanced primary dataset.
4. **LightGBM benefits from class_weight:** With `class_weight='balanced'` applied, LightGBM achieves 90.34% — competitive with XGBoost.

## Feature Subset Ablation

| Subset | Features | RF Accuracy | $\Delta$ vs all_7 |
|:-------|:---------|:------------|:-------------------|
| all_7 | All 7 | 0.9950 | — |
| mi_top_5 | Top-5 per fold | 0.9905 | −0.45% |
| mi_top_4 | Top-4 per fold | 0.9782 | −1.68% |
| mi_top_3 | Top-3 per fold | 0.9645 | −3.05% |

: Feature ablation on the primary dataset. {#tbl:ablation}

Reducing from 7 to 5 features causes only 0.45% degradation, enabling a **29% reduction in sensor count** with minimal accuracy loss.

## Sensor Degradation Robustness

| Scenario | Deployment | Accuracy | $\kappa$ | Brier | $\Delta$ vs Fresh |
|:---------|:-----------|:---------|:---------|:------|:-------------------|
| Fresh | 0 days | 0.9950 | 0.9948 | 0.0007 | — |
| Mild | 7 days | 0.9405 | 0.9376 | 0.0052 | −5.45% |
| Moderate | 30 days | 0.7041 | 0.6900 | 0.0212 | −29.09% |
| Severe | 90 days | 0.1609 | 0.1210 | 0.0452 | −83.41% |

: Proposed system robustness under literature-grounded monotonic sensor degradation. {#tbl:degradation}

Performance degrades monotonically under sensor drift (Figure 5). The compounding effect of directional drift across correlated sensors explains the non-linear collapse.

**Practical recalibration guidance:** Weekly sensor recalibration maintains >94% accuracy; monthly recalibration is insufficient (70.41%). The steep degradation curve means that **weekly recalibration is mandatory** for deployment reliability.

![Sensor degradation robustness: accuracy under mild (7-day), moderate (30-day), and severe (90-day) monotonic drift.](12_robustness.png){#fig:robustness width="100%"}

## SHAP Explainability Analysis

SHAP analysis [@Lundberg2017] of the proposed Random Forest classifier provides both global feature importance and interaction insights.

| Rank | Feature | Relative Importance | Agronomic Interpretation |
|:-----|:--------|:-------------------|:-------------------------|
| 1 | Humidity | Highest | Primary climate driver |
| 2 | Rainfall | High | Critical for water-intensive crops |
| 3 | K | Moderate-High | Key soil nutrient for fruit/grain quality |
| 4 | N | Moderate | Essential for vegetative growth |
| 5 | Temperature | Moderate | Secondary climate factor |
| 6 | P | Low-Moderate | Supporting nutrient for root development |
| 7 | pH | Low | Least discriminative |

: SHAP global feature importance for the proposed Random Forest classifier. {#tbl:shap}

The dominance of climate features (humidity, rainfall) over soil nutrients indicates that **macro-environmental conditions are the primary driver of crop suitability**.

![SHAP feature importance for the Random Forest classifier on the primary dataset.](11_shap_RandomForest.png){#fig:shap width="100%"}

## Cross-Dataset Feature Consistency

We analyse whether feature importance rankings transfer across datasets. For the three shared features (N, P, K):

$$\text{Consistency}(f) = 1 - |s_{\text{primary}}(f) - s_{\text{secondary}}(f)|$$ {#eq:consistency}

where scores are normalised consensus rankings in $[0, 1]$.

| Feature | Primary Score | Secondary Score | Consistency | Interpretation |
|:--------|:-------------|:----------------|:------------|:---------------|
| P | 0.559 | 0.363 | **0.804** | Most transferable |
| N | 0.367 | 1.000 | 0.367 | Task-dependent |
| K | 0.828 | 0.121 | 0.293 | Least transferable |

: Cross-dataset feature importance consistency analysis. {#tbl:consistency}

**Phosphorus (P) is the most reliable feature across agricultural domains** (consistency = 0.804), while **Potassium (K) is the least transferable** (0.293). This divergence underscores that feature importance should not be generalised from a single dataset.

![Cross-dataset feature comparison for shared features (N, P, K).](07_shared_features_comparison.png){#fig:cross_dataset width="100%"}

# Discussion

## Why Random Forest Outperforms on Imbalanced Real Data

The proposed Random Forest pipeline consistently outperforms gradient boosting methods on the imbalanced secondary dataset due to:

1. **`class_weight='balanced'`:** RF natively supports class weighting, re-weighting the loss inversely to class frequency, directly addressing the 11.28:1 imbalance ratio.
2. **Bagging vs boosting:** Boosting methods focus on misclassified samples in successive rounds, which can amplify noise in the minority class. Bagging (RF) provides variance reduction without over-focusing on hard examples.
3. **Feature subsampling:** RF's random feature subsampling decorrelates trees and reduces overfitting to the majority class's feature distribution.

## Calibration and Decision Reliability

For agricultural deployment, classification accuracy alone is insufficient. Our calibration analysis reveals that Random Forest is moderately well-calibrated (ECE = 0.0430) but tends toward overconfidence. LightGBM achieves the best calibration (ECE = 0.0068), making it preferred when probability estimates are critical. GaussianNB is poorly calibrated despite high accuracy, producing near-0/1 posteriors.

![Calibration curves for the proposed Random Forest classifier.](13_calibration.png){#fig:calibration width="100%"}

## Practical Deployment Guidance

| Decision | Recommendation | Evidence |
|:---------|:---------------|:---------|
| Sensor priority | Humidity > Rainfall > K > N | SHAP importance |
| Minimum sensor set | 5 features (drop Temp, pH) | 99.05% acc, 29% fewer sensors |
| Recalibration | **Weekly (mandatory)** | >94% acc; monthly = 70% |
| Classifier | RF with class_weight='balanced' | Best on both datasets |
| Feature transferability | P most reliable across domains | Consistency = 0.804 |
| Imbalance handling | Always apply class_weight | GaussianNB failure shows risk |

: Practical deployment recommendations. {#tbl:guidance}

## Limitations

1. **Semi-synthetic primary dataset:** The perfectly balanced 22-class dataset does not reflect real-world crop distributions. Our secondary dataset validation partially mitigates this.
2. **Cross-dataset analysis, not validation:** The two datasets have different target variables. True cross-dataset validation would require a second crop recommendation dataset.
3. **Sensor degradation model:** Our simulation uses literature-grounded monotonic drift. Real degradation may involve additional factors (temperature-dependent drift, cross-sensor interference).
4. **SHAP depth:** Current analysis provides global feature importance. Per-class SHAP breakdowns would provide deeper agronomic insight.

## Future Work

1. Field-collected data validation with GPS-tagged samples and actual crop outcomes.
2. Noise-augmented training on sensor-degraded variants to build drift-resilient models.
3. Deep learning with Transformer-based architectures for complex feature interactions.
4. Federated learning for privacy-preserving collaborative training across agricultural regions.
5. Per-class SHAP analysis for misclassified samples to identify agronomic edge cases.

# Conclusion

This paper proposes **RobustCrop**, a leak-free ML pipeline for crop recommendation that addresses three critical gaps: data leakage in preprocessing, class imbalance in real-world data, and lack of external validation. Through dual-dataset evaluation with cross-dataset feature consistency analysis, we demonstrate that:

1. The proposed Random Forest pipeline achieves **99.50% accuracy** on the primary dataset and **91.25% accuracy** ($\kappa$ = 0.8364, macro-F1 = 81.85%) on the real-world imbalanced secondary dataset, outperforming nine benchmark classifiers.

2. **BalWeightWrapper** provides uniform imbalance handling for all classifiers via `sample_weight='balanced'`, eliminating unfair comparison where only native `class_weight` users received correction.

3. **Leak-free Pipeline architecture** eliminates the data leakage that inflates accuracy in prior work. A Friedman test confirms statistically significant differences ($\chi^2 = 32.32$, $p < 0.001$).

4. **Consensus feature ranking** identifies humidity and rainfall as dominant predictors, with a 5-feature subset achieving 99.05% accuracy with 29% fewer sensors.

5. **GaussianNB** fails on real-world imbalanced data (80.11%) despite competitive accuracy on balanced benchmarks — a cautionary finding for agricultural ML.

6. **Sensor degradation** under monotonic drift is severe: weekly recalibration maintains >94% accuracy, but 90-day uncalibrated deployment causes 83.41% degradation.

7. **Cross-dataset feature consistency** reveals phosphorus as the most transferable feature (consistency = 0.804) while potassium importance is task-dependent (0.293).

These findings provide actionable guidance for deploying ML-based crop recommendation in precision agriculture, from sensor selection and feature engineering to classifier choice and maintenance scheduling.

# References
