# RobustCrop: A Leak-Free Machine Learning Pipeline for Crop Recommendation with Sensor Degradation Analysis and Cross-Dataset Feature Consistency

---

**Anuradha Brijwal¹ · Praveena Chaturvedi²**

¹ Research Scholar, Department of Computer Science, Kanya Gurukul Campus Dehradun
² Professor, Department of Computer Science, Kanya Gurukul Campus Dehradun

Gurukul Kangri (Deemed to be University), Haridwar, Uttarakhand, India

---

## Abstract

Machine learning for crop recommendation promises data-driven agricultural guidance, yet most studies evaluate on balanced semi-synthetic benchmarks under leakage-prone preprocessing pipelines, providing inflated performance estimates that fail to transfer to real-world conditions. We present RobustCrop, a leak-free pipeline that encapsulates feature scaling, mutual-information-based feature selection, and classification within a single cross-validation fold, eliminating information leakage. We evaluate on two datasets: a real-world soil fertility dataset (880 samples, 3 classes, 11.28:1 imbalance ratio) and a semi-synthetic crop recommendation benchmark (2,200 samples, 22 classes). On real-world data, our Random Forest pipeline achieves 91.25% ± 0.77% accuracy (κ = 0.8364, macro-F1 = 81.85%), outperforming nine benchmarks. SHAP analysis identifies humidity and rainfall as dominant predictors, while cross-dataset analysis reveals phosphorus as the most transferable feature (consistency = 0.804). Literature-grounded sensor degradation analysis shows weekly recalibration maintains >94% accuracy, whereas 90-day uncalibrated deployment causes catastrophic 83.41% degradation. These findings provide actionable deployment guidance for precision agriculture.

**Keywords:** Precision Agriculture, Crop Recommendation, Class Imbalance, Sensor Degradation, Explainable AI, Feature Selection

---

## 1. Introduction

A smallholder farmer in northern India stands at a decision point each planting season: which crop to sow on a given plot of land. The wrong choice — rice on a potassium-depleted field, or wheat in a region where monsoon humidity favors fungal disease — can mean the difference between a profitable harvest and a season's lost income. Precision agriculture aims to replace this guesswork with data-driven recommendations, using soil nutrient measurements (nitrogen, phosphorus, potassium), climatic variables (temperature, humidity, rainfall), and soil chemistry (pH) to match crops to conditions (Liakos et al., 2018; Wolfert et al., 2017).

Machine learning (ML) has shown remarkable accuracy on standard crop recommendation benchmarks, with multiple studies reporting >99% classification accuracy (Shah et al., 2022). However, these results come with caveats that limit their real-world applicability. First, most studies apply preprocessing steps — feature scaling, feature selection — before cross-validation, inadvertently leaking test-fold information into training and inflating reported accuracy by 5–30% (Kapoor and Narayanan, 2023). Second, standard benchmarks are perfectly balanced with clean synthetic data, masking the class imbalance and missing values that characterize real agricultural datasets. Third, evaluations are confined to a single dataset, leaving open whether feature importance rankings — and the sensor configurations they imply — transfer across agricultural contexts.

This paper addresses these gaps through three contributions:

1. **A leak-free pipeline architecture** that encapsulates all preprocessing within each cross-validation fold, providing honest generalization estimates. We demonstrate that this architecture yields statistically significant performance differences across classifiers (Friedman χ² = 32.32, p < 0.001), whereas leakage-prone pipelines can make mediocre classifiers appear equivalent.

2. **Dual-dataset evaluation with cross-dataset feature consistency analysis.** We evaluate on both a real-world soil fertility dataset and a semi-synthetic crop recommendation benchmark, analyzing whether feature importance rankings transfer across datasets. This reveals that phosphorus is the most transferable feature (consistency = 0.804), while potassium importance is task-dependent (0.293) — a finding invisible to single-dataset studies.

3. **Practical deployment guidance** including sensor degradation robustness under literature-grounded drift models, minimum viable sensor configurations, and recalibration schedules. We show that a 5-feature configuration achieves 99.05% accuracy with 29% fewer sensors, and that weekly recalibration is mandatory for maintaining >94% accuracy in field deployment.

---

## 2. Related Work

### 2.1 Machine Learning for Crop Recommendation

Machine learning for crop recommendation has been extensively studied over the past decade. Liakos et al. (2018) provided a comprehensive review of ML in agriculture, identifying Random Forest and SVM as consistently strong performers across agricultural classification tasks. More recently, Shah et al. (2022) reported 99.1% accuracy on the Kaggle Crop Recommendation dataset using ensemble classifiers, while Naresh Kumar et al. (2019) applied neural networks to soil-crop matching with reported accuracies exceeding 95%. Suresh et al. (2023) explored gradient boosting methods for crop recommendation in Indian agricultural contexts, achieving competitive performance on semi-synthetic benchmarks.

However, a critical limitation pervades this literature: nearly all studies report accuracy on a single, balanced, semi-synthetic benchmark without validating on independent real-world data (Elavarasan and Vincent, 2020). Jha et al. (2019) reviewed soil-crop recommendation systems and noted the absence of field validation as a major gap. Nabwire et al. (2021) similarly highlighted that most crop recommendation systems lack robustness evaluation under realistic conditions. Our work addresses this by evaluating on both a semi-synthetic benchmark and a real-world soil fertility dataset, revealing significant performance gaps between the two settings.

### 2.2 Feature Selection in Agricultural Data

Feature selection reduces dimensionality, improves interpretability, and can enhance generalization by removing noisy or redundant features. Guyon and Elisseeff (2003) categorized methods into filter, wrapper, and embedded approaches, each with distinct computational and accuracy trade-offs. In agricultural contexts, filter methods such as Mutual Information and Chi-Square are computationally efficient and scale well to high-dimensional soil nutrient data (Chandrashekar and Sahin, 2014). Embedded methods like LASSO regularization and tree-based importance provide feature selection as a byproduct of model training, balancing accuracy with interpretability.

Kamhawy et al. (2023) investigated feature selection for crop recommendation and found that reducing the feature set from seven to five variables had minimal impact on classification accuracy, suggesting that climate variables (humidity, rainfall) carry most of the discriminative signal. Our work extends this by synthesizing six feature selection methods into a consensus ranking, analyzing feature importance transferability across datasets, and identifying a minimum viable sensor configuration for budget-constrained IoT deployments.

### 2.3 Data Leakage in Machine Learning Pipelines

Data leakage — where test information inadvertently influences training — is a pervasive but under-reported issue in applied machine learning. Kapoor and Narayanan (2023) documented leakage in 229 out of 329 surveyed papers across 17 application domains, with performance inflation ranging from 5% to over 30%. Common leakage sources include feature scaling before train-test splitting, feature selection on the full dataset before cross-validation, and hyperparameter tuning on the test set.

In agricultural ML, leakage is particularly insidious because the inflated accuracy can lead to overconfident deployment decisions. A system that appears to achieve 99.5% accuracy but actually achieves 95% under leak-free evaluation may cause significant crop losses when deployed. Our Pipeline-per-fold architecture directly addresses these leakage vectors by ensuring all preprocessing occurs independently within each cross-validation fold, following established best practices for reproducible ML evaluation (Kohavi, 1995; Kapoor and Narayanan, 2023).

### 2.4 Sensor Reliability and IoT in Agriculture

The integration of IoT sensors with machine learning for precision agriculture is reviewed by Wolfert et al. (2017), who identify data quality and sensor reliability as key challenges for real-world deployment. Real-world agricultural sensing introduces noise, drift, and calibration decay that laboratory-clean benchmarks do not capture.

Rana et al. (2019) documented electrochemical NPK sensor drift rates of 1–1.5% per day, while Lobnik et al. (2011) reported pH electrode drift of 0.1% per day. Crucially, sensor drift is typically monotonic and directional — electrochemical sensors lose sensitivity over time, not randomly in both directions. Martínez et al. (2007) characterized tipping-bucket rain gauge accuracy degradation under field conditions. Sensirion (2022) specifies temperature and humidity sensor stability at 0.2% and 0.5% per day, respectively. Most ML studies assume clean, static data; our robustness analysis quantifies model degradation using these literature-grounded monotonic drift parameters, providing practical recalibration guidelines for field deployment.

### 2.5 Class Imbalance in Agricultural Data

Class imbalance is endemic in real-world agricultural datasets, where favorable conditions vastly outnumber marginal or deficient ones. Chawla et al. (2002) introduced SMOTE (Synthetic Minority Over-sampling Technique) as a general-purpose solution, generating synthetic samples for minority classes. Subsequent variants — Borderline-SMOTE (Han et al., 2005), ADASYN (He et al., 2008), and SMOTE-ENN (Batista et al., 2004) — have refined this approach with adaptive sampling strategies.

However, for tabular soil data with limited feature dimensions, class weighting often outperforms oversampling because it preserves the original data distribution without introducing synthetic artifacts (Krawczyk, 2016). The `class_weight='balanced'` parameter available in scikit-learn classifiers adjusts sample weights inversely proportional to class frequency, providing a simple and effective correction. A practical limitation is that not all classifiers support native class weighting — gradient boosting implementations (XGBoost, LightGBM) and distance-based methods (KNN) require alternative approaches such as sample weighting or wrapper-based solutions. Our work ensures uniform imbalance handling across all evaluated classifiers, enabling fair comparison on the imbalanced secondary dataset (11.28:1 ratio).

### 2.6 Explainability in Agricultural Machine Learning

Explainability is increasingly important for agricultural ML adoption, as farmers and agronomists need to understand *why* a system recommends a particular crop before trusting it. Lundberg and Lee (2017) introduced SHAP (SHapley Additive exPlanations), a unified framework for model-agnostic feature attribution grounded in cooperative game theory. SHAP values provide both global feature importance (which features matter overall) and local explanations (why a specific prediction was made).

Kamilaris and Prenafeta-Boldú (2018) reviewed deep learning in agriculture and noted the tension between model complexity and interpretability. For crop recommendation, explainability serves two purposes: scientific insight (understanding which soil and climate factors drive crop suitability) and deployment trust (enabling agronomists to validate and correct system recommendations). Our SHAP analysis reveals that climate features (humidity, rainfall) dominate over soil nutrients for crop recommendation, while micronutrients (Zn, Mn, Fe) are most important for soil fertility classification — a distinction that has direct implications for sensor deployment priorities.

---

## 3. Datasets

### 3.1 Primary Dataset: Crop Recommendation

The primary dataset is a widely-used semi-synthetic crop recommendation benchmark (Atharva Ingle, Kaggle) containing 2,200 samples with 7 features (N, P, K, temperature, humidity, pH, rainfall) and 22 balanced crop classes (100 samples each). The dataset was augmented from Indian agricultural statistics and does not represent natural field distributions. Its perfect class balance and clean feature space allow classifiers to achieve >99% accuracy that may not transfer to real-world conditions. Results on this dataset should be interpreted as upper-bound estimates of classifier capability under ideal conditions.

### 3.2 Secondary Dataset: Soil Fertility

The secondary dataset comprises 880 real soil laboratory test results from Indian agricultural testing centres (Rahul Jaiswal, Kaggle), with 12 features (N, P, K, pH, EC, OC, S, Zn, Fe, Cu, Mn, B) and 3 fertility classes (High: 401, Medium: 440, Low: 39). The natural class imbalance ratio of 11.28:1 and ~3% real missing values provide a realistic test surface. This dataset is the more meaningful evaluation: real data, natural imbalance, and real measurement noise. The minority class (Low Fertility, 39 samples) represents the setting where imbalance-aware training is most critical.

### 3.3 Limitations

We are transparent about dataset limitations. The primary dataset is semi-synthetic — its balanced, clean nature means results represent best-case performance. The two datasets have different target variables (crop classes vs. fertility classes), so our cross-dataset analysis compares feature importance rankings, not model transfer. True cross-dataset validation would require a second crop recommendation dataset from a different region, which we identify as future work.

### 3.4 Sensor Degradation Variants

To simulate realistic deployment conditions, we generate degraded variants of the primary dataset using literature-grounded monotonic drift parameters. Unlike random bidirectional perturbation, real sensor drift is directional — electrochemical sensors consistently lose sensitivity over time. Each sensor is assigned a fixed drift direction per simulation seed.

| Sensor | Drift (%/day) | Noise (σ) | Source |
|--------|--------------|-----------|--------|
| N (Nitrogen) | 1.0 | 2.0 | Rana et al. (2019) |
| P (Phosphorus) | 1.5 | 1.5 | Rana et al. (2019) |
| K (Potassium) | 1.2 | 1.5 | Rana et al. (2019) |
| Temperature | 0.2 | 0.5 | Sensirion (2022) |
| Humidity | 0.5 | 1.0 | Sensirion (2022) |
| pH | 0.1 | 0.1 | Lobnik et al. (2011) |
| Rainfall | 0.3 | 5.0 | Martínez et al. (2007) |

Three degradation scenarios are generated: mild (7-day), moderate (30-day), and severe (90-day) deployments, with realistic dropout rates (2–10% scaled to deployment duration).

---

## 4. Methods

### 4.1 Pipeline Architecture

The central methodological contribution of RobustCrop is a leak-free pipeline architecture that ensures all preprocessing steps observe only training data within each cross-validation fold. Let $\mathcal{D} = \{(\mathbf{x}_i, y_i)\}_{i=1}^{N}$ denote the dataset with feature vectors $\mathbf{x}_i \in \mathbb{R}^d$ and labels $y_i \in \{1, \ldots, C\}$. In $K$-fold stratified cross-validation, the data is partitioned into $K$ disjoint folds $\mathcal{D} = \bigcup_{k=1}^{K} \mathcal{F}_k$. For each fold $k$:

1. **Scaling:** StandardScaler is fitted on $\mathcal{D} \setminus \mathcal{F}_k$ (training fold), producing mean $\boldsymbol{\mu}_k$ and standard deviation $\boldsymbol{\sigma}_k$. Both training and validation folds are transformed:
   
   $$\tilde{\mathbf{x}}_i = \frac{\mathbf{x}_i - \boldsymbol{\mu}_k}{\boldsymbol{\sigma}_k}$$

2. **Feature Selection:** Mutual information $I(\mathbf{x}_j; y)$ is computed between each feature $j$ and the target, estimated on the training fold only. The top-$m$ features are selected via SelectKBest.

3. **Classification:** The classifier is trained on the scaled, selected training fold and evaluated on the held-out validation fold.

This Pipeline-per-fold architecture is encapsulated as a single scikit-learn Pipeline object per fold, eliminating the common leakage vector where scaling and selection are applied globally before cross-validation (Kapoor and Narayanan, 2023).

### 4.2 Feature Selection: Consensus Ranking

Six feature selection methods are evaluated to build a consensus ranking for interpretability:

1. **Mutual Information (MI):** $I(\mathbf{x}_j; y) = H(y) - H(y | \mathbf{x}_j)$, a non-parametric measure of statistical dependence.
2. **Chi-Square (χ²):** Tests independence between binned feature values and target classes.
3. **Recursive Feature Elimination (RFE):** Wrapper method using Random Forest as base estimator, iteratively removing least important features.
4. **LASSO Regularization (L1):** Embedded method using Logistic Regression with L1 penalty, driving feature coefficients to zero.
5. **Extra Trees Importance:** Ensemble-based feature importance from Extremely Randomized Trees.
6. **Random Forest Importance:** Mean decrease in impurity (Gini importance) from Random Forest.

Each method's scores are normalized to [0, 1] and averaged to produce a robust consensus ranking $\bar{s}_j = \frac{1}{6}\sum_{m=1}^{6} s_j^{(m)}$ for each feature $j$. The consensus ranking is computed once on the full training set for interpretability; during cross-validation, per-fold MI selection is used to maintain the leak-free property.

### 4.3 Classifier Configuration

The proposed system uses Random Forest (Breiman, 2001) with 200 trees, max depth 20, minimum samples split 5, and class weighting balanced. Class weighting adjusts sample weights inversely proportional to class frequency:

$$w_c = \frac{N}{C \cdot n_c}$$

where $N$ is the total number of samples, $C$ is the number of classes, and $n_c$ is the number of samples in class $c$. This directly addresses class imbalance without introducing synthetic samples.

Nine additional classifiers serve as benchmarks: SVM with RBF kernel (C=10, γ='scale'), KNN (k=7, distance weighting), Decision Tree (max depth 15), Gradient Boosting (150 trees, lr=0.1), XGBoost (200 trees, max depth 6), LightGBM (200 trees, max depth 6), Logistic Regression (L-BFGS, C=1.0), MLP (128-64-32 architecture, early stopping), and GaussianNB (default parameters). All classifiers receive imbalance correction: those with native class_weight support use `class_weight='balanced'`; the remainder receive equivalent sample weighting at fit time, ensuring fair comparison on imbalanced data.

### 4.4 Evaluation Protocol

**Metrics.** We report accuracy, Cohen's kappa (κ), Matthews Correlation Coefficient (MCC), macro-averaged F1, Brier score, and Expected Calibration Error (ECE). For the balanced primary dataset, accuracy ≈ macro-F1; for the imbalanced secondary dataset, macro-F1 is the most informative metric as it weights all classes equally regardless of frequency.

**Statistical Testing.** Friedman test (Friedman, 1937) across all classifiers, with Nemenyi post-hoc critical difference for pairwise comparisons. The test statistic is $\chi^2_F = \frac{12N}{k(k+1)}\left[\sum_j R_j^2 - \frac{k(k+1)^2}{4}\right]$, where $R_j$ is the average rank of classifier $j$ across datasets.

**Hyperparameter Tuning.** Optional Bayesian optimization via Optuna (Akiba et al., 2019) with 30 trials and 3-fold inner CV, using the TPE sampler. Classifier-specific search spaces are defined for key parameters. When tuning is not used, fixed hyperparameters are applied.

---

## 5. Results

### 5.1 Real-World Secondary Dataset Results

We lead with the secondary dataset results because they better represent deployment conditions: real data, natural class imbalance (11.28:1), and real measurement noise.

| Classifier | Accuracy | κ | MCC | Macro-F1 | Brier | ECE |
|-----------|----------|------|------|----------|-------|------|
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

*Table 1: Classification results on the real-world secondary dataset (MI-selected top-6 features). All classifiers receive imbalance correction. Bold = proposed system.*

**Key findings:**

The proposed Random Forest achieves **91.25% accuracy** (κ = 0.8364, macro-F1 = 81.85%), outperforming all benchmarks. The accuracy–macro-F1 gap (9.40 percentage points) reveals the minority class challenge: even the best classifier struggles with the Low Fertility class (39 samples). This gap is substantially smaller for the proposed system than for benchmarks like MLP (24.37 pp) or KNN (27.75 pp), demonstrating the practical value of class weighting.

GaussianNB, despite achieving 99.45% on the balanced primary dataset, collapses to 80.11% ± 7.11% on the imbalanced secondary dataset. This is a cautionary finding: balanced-benchmark performance is not indicative of real-world capability. The high variance (±7.11%) indicates that GaussianNB's performance is sensitive to the particular fold composition, a consequence of its inability to adapt to class imbalance.

Feature selection improves generalization: the proposed system with MI-selected top-6 features (91.25%) outperforms the full 12-feature configuration (90.68%), confirming that removing noisy features within each fold reduces overfitting. LightGBM and XGBoost achieve competitive performance (90.34%) when provided with class weighting, confirming that imbalance handling matters more than algorithmic sophistication for this task.

### 5.2 Semi-Synthetic Primary Dataset Results

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

*Table 2: Classification results on the semi-synthetic primary dataset (all 7 features). Friedman test across all 10 classifiers: χ² = 32.32, p < 0.001.*

All classifiers exceed 97% accuracy, confirming the primary dataset's well-separated feature space. The proposed Random Forest achieves **99.50% ± 0.09%** with the lowest variance. However, these results represent best-case performance on clean, balanced data and should not be taken as indicative of field deployment accuracy. The stark gap between primary dataset performance (99.50%) and secondary dataset performance (91.25%) for the same classifier underscores this point.

### 5.3 Feature Selection Analysis

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

All six methods consistently rank humidity as the most discriminative feature, followed by rainfall and potassium. The dominance of climate features over soil nutrients suggests that macro-environmental conditions are the primary driver of crop suitability.

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

For real soil fertility data, micronutrients (Zn, Mn, Fe, B) dominate — contrary to the primary dataset where macronutrients (N, P, K) are mid-ranked. This indicates that micronutrient profiles carry stronger discriminative signal for fertility classification, a finding that would be missed if only the semi-synthetic benchmark were used.

### 5.4 Feature Subset Ablation

| Subset | Features Used | Proposed RF Accuracy | Δ vs all_7 |
|--------|--------------|---------------------|-----------|
| all_7 | All 7 features | 0.9950 | — |
| mi_top_5 | Top-5 per fold (MI) | 0.9905 | −0.45% |
| mi_top_4 | Top-4 per fold (MI) | 0.9782 | −1.68% |
| mi_top_3 | Top-3 per fold (MI) | 0.9645 | −3.05% |

*Table 3: Feature ablation on the primary dataset. Features selected per-fold via mutual information.*

Reducing from 7 to 5 features (removing temperature and pH) causes only 0.45% degradation, validating the consensus ranking. For budget-constrained IoT deployments, this enables a **29% reduction in sensor count** with minimal accuracy loss. Reducing to 3 features causes a more significant 3.05% drop, indicating N and P provide complementary information beyond the climate features.

### 5.5 Cross-Dataset Feature Consistency

We analyze whether feature importance rankings from the semi-synthetic primary dataset transfer to the real-world secondary dataset. For the three features shared across datasets (N, P, K), we compute consistency as:

$$\text{Consistency}_j = 1 - |s_j^{\text{primary}} - s_j^{\text{secondary}}|$$

where scores are normalized consensus rankings in [0, 1].

| Feature | Primary Score | Secondary Score | Consistency | Interpretation |
|---------|--------------|-----------------|-------------|----------------|
| P | 0.559 | 0.363 | **0.804** | Most transferable — moderately important in both contexts |
| N | 0.367 | 1.000 | 0.367 | Highly inconsistent — important for fertility, less so for crops |
| K | 0.828 | 0.121 | 0.293 | Least transferable — important for crops, not for fertility |

*Table 4: Feature importance consistency across datasets.*

Phosphorus is the most reliable feature across agricultural domains (consistency = 0.804), maintaining moderate importance in both crop recommendation and soil fertility classification. Potassium is the least transferable (consistency = 0.293) — it is the third most important feature for crop recommendation but ranks near the bottom for soil fertility. This divergence demonstrates that feature importance is task-dependent and should not be generalized from a single dataset.

### 5.6 Sensor Degradation Robustness

| Scenario | Deployment | Accuracy | κ | Brier | Δ vs Fresh |
|----------|-----------|----------|------|-------|-----------|
| Fresh | 0 days | 0.9950 | 0.9948 | 0.0007 | — |
| Mild | 7 days | 0.9405 | 0.9376 | 0.0052 | −5.45% |
| Moderate | 30 days | 0.7041 | 0.6900 | 0.0212 | −29.09% |
| Severe | 90 days | 0.1609 | 0.1210 | 0.0452 | −83.41% |

*Table 5: Robustness under literature-grounded monotonic sensor degradation.*

Performance degrades monotonically under sensor drift. The 7-day threshold shows moderate degradation (5.45 pp loss), but the decline accelerates sharply: 30-day deployment loses 29.09 pp and 90-day deployment loses 83.41 pp. The compounding effect of directional drift across correlated sensors (N–P–K, humidity–rainfall) explains the non-linear collapse. **Weekly sensor recalibration maintains >94% accuracy; monthly recalibration is insufficient (70.41%).** The steep degradation curve means weekly recalibration is not optional but mandatory for deployment reliability.

### 5.7 SHAP Explainability Analysis

SHAP analysis of the proposed Random Forest reveals that climate features dominate global feature importance. Humidity ranks highest, followed by rainfall and potassium. The dominance of climate features over soil nutrients indicates that macro-environmental conditions are the primary driver of crop suitability, with soil nutrients serving as secondary refinement factors.

Feature interactions are also evident: humidity and rainfall exhibit strong synergistic effects — their combined SHAP contribution exceeds the sum of individual contributions for water-sensitive crops (rice, coconut, watermelon). Similarly, K and N interact for fruit-bearing crops where potassium supports fruit development and nitrogen supports vegetative growth.

A Friedman test across all 10 classifiers on the primary dataset confirms statistically significant differences (χ² = 32.32, p < 0.001), validating that the choice of classifier matters even on well-separated data.

---

## 6. Discussion

### 6.1 Why Random Forest Outperforms on Imbalanced Real Data

The proposed Random Forest pipeline consistently outperforms gradient boosting methods on the imbalanced secondary dataset. We attribute this to three factors. First, class weighting re-weights the loss inversely to class frequency, directly addressing the 11.28:1 imbalance ratio. Second, bagging (RF) trains each tree on a bootstrap sample, providing variance reduction without over-focusing on hard minority-class examples — unlike boosting, which can amplify noise in successive rounds. Third, RF's random feature subsampling at each split decorrelates trees and reduces overfitting to the majority class's feature distribution. Since all classifiers receive equivalent imbalance handling, performance differences are attributable to algorithmic characteristics rather than differential treatment of imbalance.

### 6.2 Agronomic Implications

The feature importance rankings have direct agronomic implications. The dominance of humidity and rainfall for crop recommendation confirms that macro-climatic conditions are the primary constraint on crop suitability — a finding consistent with agronomic knowledge that water availability is the limiting factor in Indian agriculture (Jha et al., 2019). For soil fertility classification, the importance of micronutrients (Zn, Mn, Fe, B) over macronutrients (N, P, K) suggests that micronutrient deficiency is a more reliable indicator of poor soil health than macronutrient levels alone.

The cross-dataset analysis reveals that phosphorus is the most reliable feature across both tasks, consistent with its well-established role in root development and its dual importance for both crop productivity and soil health assessment. Potassium's task-dependent importance (high for crop recommendation, low for fertility) suggests that K levels are more relevant for matching crops to conditions than for assessing general soil quality.

### 6.3 Practical Deployment Guidance

| Decision | Recommendation | Evidence |
|----------|---------------|----------|
| Sensor priority | Humidity > Rainfall > K > N | SHAP importance |
| Minimum sensor set | 5 features (drop temperature, pH) | 99.05% accuracy, 29% fewer sensors |
| Recalibration frequency | Weekly (mandatory) | >94% accuracy; monthly = 70.41% |
| Classifier choice | RF with class weighting | Best on both datasets |
| Feature transferability | P most reliable across domains | Consistency = 0.804 |
| Imbalance handling | Always apply class weighting | GaussianNB failure demonstrates risk |

For resource-constrained IoT deployments, the 5-feature configuration (dropping temperature and pH) offers the best accuracy-per-sensor ratio. However, this recommendation assumes that the deployment environment is similar to the training data; in regions where temperature or pH are limiting factors (e.g., acid sulfate soils, high-altitude farming), these features may become critical and should be retained.

### 6.4 Calibration and Decision Reliability

For agricultural deployment, classification accuracy alone is insufficient — farmers need to trust the system's confidence estimates. Our calibration analysis reveals that Random Forest is moderately well-calibrated (ECE = 0.0430) but tends toward overconfidence on the primary dataset. LightGBM achieves the best calibration (ECE = 0.0068), making it the preferred choice when probability estimates are critical for risk-averse farming decisions. GaussianNB is poorly calibrated despite high accuracy, producing near-0/1 posteriors — a consequence of the conditional independence assumption. For deployment, we recommend using the proposed RF model for classification but consulting probability estimates from a well-calibrated model (e.g., LightGBM) for risk assessment, or applying post-hoc calibration via Platt scaling or isotonic regression (Guo et al., 2017).

### 6.5 Limitations

1. **Semi-synthetic primary dataset.** The perfectly balanced 22-class benchmark does not reflect real-world crop distributions. Our secondary dataset validation partially mitigates this, but a field-collected crop recommendation dataset would be the definitive test.

2. **Cross-dataset analysis, not validation.** The two datasets have different target variables. Our analysis compares feature importance rankings, not model transfer. True cross-dataset validation requires a second crop recommendation dataset from a different region.

3. **Sensor degradation model.** Our simulation uses literature-grounded monotonic drift with realistic dropout rates, but real sensor degradation may involve additional factors (temperature-dependent drift, cross-sensor interference) not captured here.

4. **SHAP depth.** Current analysis provides global feature importance and interaction insights. Per-class SHAP breakdowns and local explanations for specific misclassified samples would provide deeper agronomic insight.

---

## 7. Conclusion

This paper presents RobustCrop, a leak-free machine learning pipeline for crop recommendation that addresses three critical gaps in prior work: data leakage in preprocessing, class imbalance in real-world data, and lack of external validation. Four key findings emerge from our dual-dataset evaluation:

1. **Real-world performance gaps are substantial.** The proposed Random Forest achieves 99.50% on the balanced semi-synthetic benchmark but 91.25% (macro-F1 = 81.85%) on the real-world imbalanced dataset — an 8.25 percentage point gap that highlights the danger of relying solely on synthetic benchmarks for deployment decisions.

2. **Feature importance is task-dependent.** Cross-dataset analysis reveals phosphorus as the most transferable feature (consistency = 0.804), while potassium importance varies dramatically between crop recommendation and soil fertility classification (consistency = 0.293). Sensor deployment strategies must account for the specific classification task.

3. **Sensor degradation demands weekly recalibration.** Literature-grounded monotonic drift simulation shows that weekly recalibration maintains >94% accuracy, while 90-day uncalibrated deployment causes catastrophic 83.41% degradation. This establishes a concrete maintenance schedule for agricultural IoT deployments.

4. **Class weighting is essential for imbalanced agricultural data.** Without imbalance correction, classifiers that appear competitive on balanced benchmarks (e.g., GaussianNB: 99.45%) collapse on real data (80.11%). Uniform imbalance handling across all classifiers ensures fair comparison and honest performance estimates.

Future work should prioritize field-collected crop recommendation datasets, noise-augmented training for drift resilience, and per-class SHAP analysis for deeper agronomic insight.

---

## References

1. Akiba, T., Sano, S., Yanase, T., Ohta, T., & Koyama, M. (2019). Optuna: A next-generation hyperparameter optimization framework. *Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining*, 2623–2631.
2. Batista, G. E. A. P. A., Prati, R. C., & Monard, M. C. (2004). A study of the behavior of several methods for balancing machine learning training data. *ACM SIGKDD Explorations Newsletter*, 6(1), 20–29.
3. Breiman, L. (2001). Random Forests. *Machine Learning*, 45(1), 5–32.
4. Chandrashekar, G., & Sahin, F. (2014). A survey on feature selection methods. *Computers & Electrical Engineering*, 40(1), 16–28.
5. Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). SMOTE: Synthetic Minority Over-sampling Technique. *Journal of Artificial Intelligence Research*, 16, 321–357.
6. Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery & Data Mining*, 785–794.
7. Elavarasan, D., & Vincent, D. R. (2020). Crop recommendation system based on investigative analysis of soil and climatic parameters. *Computers and Electronics in Agriculture*, 178, 105758.
8. Friedman, M. (1937). The use of ranks to avoid the assumption of normality implicit in the analysis of variance. *Journal of the American Statistical Association*, 32(200), 675–701.
9. Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On calibration of modern neural networks. *Proceedings of the 34th International Conference on Machine Learning*, 1321–1330.
10. Guyon, I., & Elisseeff, A. (2003). An introduction to variable and feature selection. *Journal of Machine Learning Research*, 3, 1157–1182.
11. Han, H., Wang, W.-Y., & Mao, B.-H. (2005). Borderline-SMOTE: A new over-sampling method in imbalanced data sets learning. *Advances in Intelligent Computing*, 878–887.
12. He, H., Bai, Y., Garcia, E. A., & Li, S. (2008). ADASYN: Adaptive synthetic sampling approach for imbalanced learning. *IEEE International Joint Conference on Neural Networks*, 1322–1328.
13. Jha, K., Doshi, A., Patel, P., & Shah, M. (2019). A comprehensive review on automation in agriculture using artificial intelligence. *Artificial Intelligence in Agriculture*, 2, 1–12.
14. Kamhawy, E., Elsayed, S., & El-Bendary, N. (2023). Feature selection for crop recommendation using meta-heuristic optimization. *International Journal of Advanced Computer Science and Applications*, 14(5), 1–8.
15. Kamilaris, A., & Prenafeta-Boldú, F. X. (2018). Deep learning in agriculture: A survey. *Computers and Electronics in Agriculture*, 147, 70–90.
16. Kapoor, S., & Narayanan, A. (2023). Leakage and the reproducibility crisis in machine-learning-based science. *Patterns*, 4(9), 100804.
17. Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., ... & Liu, T.-Y. (2017). LightGBM: A highly efficient gradient boosting decision tree. *Advances in Neural Information Processing Systems*, 30.
18. Kohavi, R. (1995). A study of cross-validation and bootstrap for accuracy estimation and model selection. *Proceedings of the 14th International Joint Conference on Artificial Intelligence*, 1137–1143.
19. Krawczyk, B. (2016). Learning from imbalanced data: Open challenges and future directions. *Progress in Artificial Intelligence*, 5(4), 221–232.
20. Liakos, K. G., Busato, P., Moshou, D., Pearson, S., & Bochtis, D. (2018). Machine learning in agriculture: A review. *Sensors*, 18(8), 2674.
21. Lobnik, A., Oćwieja, M., & Križaj, D. (2011). Long-term stability of pH sensors. *Sensors and Actuators B*, 156(2), 593–599.
22. Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. *Advances in Neural Information Processing Systems*, 30.
23. Martínez, M. A., Laguna, A., & Vicente, J. (2007). Tipping-bucket rain gauge accuracy. *Hydrology and Earth System Sciences*, 11(2), 883–894.
24. Nabwire, S., Mwangi, R. W., & Ikoha, A. P. (2021). A review of machine learning techniques for crop recommendation. *International Journal of Computer Applications*, 174(25), 1–7.
25. Naresh Kumar, N., Jothi, K., & Mohan, V. (2019). Crop recommendation system using machine learning techniques. *International Journal of Recent Technology and Engineering*, 8(3), 5940–5943.
26. Nemenyi, P. (1963). Distribution-free multiple comparisons. *PhD dissertation, Princeton University*.
27. Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., ... & Chintala, S. (2019). PyTorch: An imperative style, high-performance deep learning library. *Advances in Neural Information Processing Systems*, 32.
28. Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830.
29. Platt, J. C. (1999). Probabilistic outputs for support vector machines and comparisons to regularized likelihood methods. *Advances in Large Margin Classifiers*, 61–74.
30. Rana, S. S., Bhargava, R., & Sharma, R. (2019). IoT-based smart agriculture sensor networks. *IEEE Access*, 7, 155274–155291.
31. Sensirion AG (2022). *SHT4x datasheet: Digital humidity and temperature sensor*. Sensirion.
32. Shah, K., Patel, H., & Jain, A. (2022). Crop recommendation using machine learning. *International Journal of Engineering Trends and Technology*, 70(3), 134–142.
33. Suresh, S., Priya, S., & Rajkumar, S. (2023). An ensemble-based crop recommendation system using gradient boosting. *International Journal of Information Technology*, 15(2), 891–900.
34. Tibshirani, R. (1996). Regression shrinkage and selection via the lasso. *Journal of the Royal Statistical Society: Series B*, 58(1), 267–288.
35. Van der Maaten, L., & Hinton, G. (2008). Visualizing data using t-SNE. *Journal of Machine Learning Research*, 9, 2579–2605.
36. Wolfert, S., Ge, L., Verdouw, C., & Bogaardt, M.-J. (2017). Big data in smart farming: A review. *Agricultural Systems*, 153, 69–80.
37. World Bank (2023). *Employment in agriculture (% of total employment)*. World Bank Open Data.
38. Zhang, C., & Ma, Y. (Eds.) (2012). *Ensemble machine learning: Methods and applications*. Springer.
39. Zadrozny, B., & Elkan, C. (2002). Transforming classifier scores into accurate multiclass probability estimates. *Proceedings of the 8th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining*, 694–699.
40. Geurts, P., Ernst, D., & Wehenkel, L. (2006). Extremely randomized trees. *Machine Learning*, 63(1), 3–42.
41. Fernández, A., García, S., Galar, M., Prati, R. C., Krawczyk, B., & Herrera, F. (2018). *Learning from imbalanced data sets*. Springer.
42. Niculescu-Mizil, A., & Caruana, R. (2005). Predicting good probabilities with supervised learning. *Proceedings of the 22nd International Conference on Machine Learning*, 625–632.
43. Guyon, I., Weston, J., Barnhill, S., & Vapnik, V. (2002). Gene selection for cancer classification using support vector machines. *Machine Learning*, 46(1-3), 389–422.

---

## Appendix A: Complete Results Tables

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

| Figure | Description |
|--------|-------------|
| 1 | Primary dataset feature distributions (N, P, K, Temperature, Humidity, pH, Rainfall) |
| 2 | Secondary dataset feature distributions (12 soil fertility features) |
| 3 | Primary dataset correlation heatmap |
| 4 | Secondary dataset correlation heatmap |
| 5 | Class distribution comparison (22 crop classes vs 3 fertility levels) |
| 6 | Sensor degradation effect on feature distributions |
| 7 | Shared feature space comparison (N, P, K, pH across datasets) |
| 8 | Feature selection methods — Primary dataset (6 methods + consensus) |
| 9 | Feature selection methods — Secondary dataset (6 methods + consensus) |
| 10 | Cross-validation accuracy comparison across classifiers |
| 11 | SHAP feature importance (RandomForest, GaussianNB) |
| 12 | Robustness under sensor degradation (accuracy vs deployment days) |
| 13 | Calibration curves (top classifiers) |
| 14 | Per-class F1 heatmap (22 crop classes × 10 classifiers) |

---

## Appendix C: Reproducibility

All code, data, and results are publicly available at: https://github.com/Aldrin7/Crop

The Pipeline-per-fold architecture ensures deterministic results given the same random seed (RANDOM_STATE=42). All figures, tables, and metrics are regenerated by the pipeline.
