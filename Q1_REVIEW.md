# Q1 Journal Peer-Review Report

**Manuscript Title:** Robustness-Aware Crop Recommendation Using Soil-Climate Data: A Comparative Study of Feature Selection and Classification Methods
**Recommendation:** Major Revision / Resubmit
**Journal Target Level:** Q1 (e.g., *Computers and Electronics in Agriculture*, *IEEE Transactions on AgriFood Electronics*)

## 1. Summary of the Work
The manuscript presents a machine learning pipeline for crop recommendation utilizing soil and climate data. The authors attempt to pivot away from the standard (and often flawed) pure accuracy benchmarking prevalent in this domain, focusing instead on model robustness under simulated degradation, feature importance via SHAP, and statistical validation (Friedman/Nemenyi). While the narrative shift toward "robustness" is commendable and necessary for precision agriculture, several critical methodological and technical flaws prevent this work from being publishable in a top-tier journal in its current state.

## 2. Major Methodological Critiques (Must Fix)

### 2.1 The "Kaggle Dataset" Illusion and External Validity
The fundamental flaw of this paper is its reliance on the "Crop Recommendation Dataset" by Atharva Ingle. It is an open secret in the agronomic ML community that this dataset is heavily curated, artificially balanced (exactly 100 samples per class), and likely semi-synthetic.
* **The Critique:** Claiming "Robustness-Aware Evaluation" on a dataset that contains no natural environmental variance, no sensor noise, and no class imbalance is paradoxical. The >99% accuracy is an artifact of the data generation process, not a reflection of agricultural reality.
* **Required Action:** The authors *must* explicitly acknowledge the semi-synthetic nature of this dataset in the abstract and methodology. To achieve Q1 publication, the pipeline must be validated on an independent, real-world agronomic dataset (e.g., real IoT sensor data with temporal drift and missing values) to prove the robustness framework generalizes.

### 2.2 Feature Selection Data Leakage (Cross-Validation Flaw)
In Phase 2.2 of `pipeline.py`, feature selection (Chi-Square, Mutual Information, RFE, LASSO, Boruta) is performed on the entire `X_train` dataset *before* the cross-validation loop in Session 3.
* **The Critique:** This is a classic form of data leakage. Feature selection must be performed *inside* the cross-validation loop. By selecting features based on the entire `X_train`, the CV scores in Session 3 are upwardly biased because the model evaluates features that were selected using the validation folds.
* **Required Action:** The authors must rewrite Session 3 to use `sklearn.pipeline.Pipeline` where the feature selection step (e.g., `SelectKBest`) is a step in the pipeline, ensuring it is fit independently on the training folds of each CV iteration.

### 2.3 Sensor Degradation Simulation Validity
While the shift from generic Gaussian noise to "Sensor-Specific Noise Models" in the recent update is a massive improvement, the implementation remains highly theoretical.
* **The Critique:** The combined degradation (mild/moderate/severe) is arbitrary. How does a "moderate" drift of a pH sensor compare to a "moderate" failure of a soil moisture probe in the field?
* **Required Action:** The authors should ground their noise simulations in empirical literature. Cite specific IoT sensor reliability studies (e.g., typical drift rates for NPK electrochemical sensors over a 30-day deployment) to justify the perturbation parameters (σ values).

## 3. Technical & Software Engineering Critiques (The "Logo" Level)

### 3.1 Hardcoded State and Monolithic Scripting
* **The Critique:** `pipeline.py` is an 1,800+ line monolith. The use of `.flag` files and pickling global state (`save_checkpoint`) in a single procedural script is brittle. It makes unit testing the individual components (like the custom noise injection logic) impossible.
* **Required Action:** Refactor the codebase. Break `pipeline.py` into modular components (e.g., `src/data_processing.py`, `src/models.py`, `src/evaluation.py`). Use a proper pipeline orchestrator (like `dvc` or `make`) instead of manual file flag checking.

### 3.2 Evaluation Metrics Consistency
* **The Critique:** The script calculates Precision, Recall, and F1-Score using `average='weighted'`. In a perfectly balanced dataset (100 samples per class), `macro`, `micro`, and `weighted` averages will mathematically yield the exact same result as accuracy.
* **Required Action:** The authors should drop the redundant columns in their tables. When classes are perfectly balanced and every prediction is made, Accuracy = Micro F1 = Macro F1 = Weighted F1. Presenting them as distinct achievements betrays a lack of deep statistical understanding.

### 3.3 Interpretability Contradictions
* **The Critique:** The authors state that GaussianNB assumes conditional independence, yet their own feature correlation heatmap (Figure 3) and independence analysis (Phase 8.4) show moderate to high correlations (e.g., Potassium and Phosphorus).
* **Required Action:** The authors must explain *why* GaussianNB succeeds despite the violation of its core assumption. (Hint: Naive Bayes only requires the decision boundary to be correct, not the actual probability estimates. If the classes are far apart in the feature space, Naive Bayes pushes the probabilities to 1.0/0.0, completely destroying calibration but maintaining accuracy). The Brier score analysis in Phase 8.6 should reflect this poor calibration.

## 4. Final Recommendation
The codebase demonstrates excellent effort in artifact generation and reproducibility. However, evaluating "robustness" on a synthetic dataset using a leaky cross-validation pipeline undermines the scientific validity of the conclusions. The authors must fix the CV pipeline and, ideally, validate their claims on a secondary, real-world dataset.