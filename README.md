# Crop-Research-v2 — Q1 Journal-Grade ML Pipeline

## Fixes Applied per Peer Review

### 2.1 Dataset Limitations ✅
- Explicit semi-synthetic acknowledgement in paper draft & code docstring
- Secondary real-world dataset with natural variance, missing values, class imbalance
- Cross-dataset validation framework

### 2.2 Feature Selection Data Leakage ✅
- Feature selection performed INSIDE cross-validation loop via sklearn Pipeline
- No pre-selection on full X_train before CV
- Nested CV: outer=evaluation, inner=FS+HP tuning

### 2.3 Sensor Degradation Grounded in Literature ✅
- Empirical drift rates from IoT sensor reliability studies
- NPK electrochemical sensor drift: 0.5-2% per day (Rana et al., 2019)
- pH glass electrode drift: 0.01-0.05 pH units per month (Lobnik et al., 2011)
- SHT-series humidity/temp sensor: ±2% RH drift over 6 months

### 3.1 Modular Codebase ✅
- `src/data_loader.py` — dataset acquisition
- `src/preprocessing.py` — scaling, encoding, outlier handling
- `src/feature_selection.py` — FS methods (inside CV)
- `src/models.py` — classifier definitions
- `src/evaluation.py` — metrics, statistical tests, calibration
- `src/noise_injection.py` — empirically-grounded sensor degradation
- `src/explainability.py` — SHAP analysis
- `pipeline.py` — orchestrator only

### 3.2 Metrics Consistency ✅
- Dropped redundant weighted/micro for balanced datasets
- Added: Cohen's Kappa, Matthews Correlation Coefficient, Brier Score, Log Loss
- Per-class metrics table with class-specific F1 only

### 3.3 Interpretability ✅
- SHAP TreeExplainer for all tree-based models
- Explicit discussion of GaussianNB conditional independence violation
- Brier score calibration analysis showing NB overconfidence
- Expected Calibration Error (ECE) for all models

## Usage
```bash
cd Crop-Research-v2
python pipeline.py --session 1  # Data & EDA
python pipeline.py --session 2  # Preprocessing + FS (leak-free)
python pipeline.py --session 3  # Training with nested CV
python pipeline.py --session 4  # Evaluation + SHAP + Calibration
python pipeline.py --session 5  # Compilation & paper artifacts
python pipeline.py --all        # All sessions
```

## Target Journal
**Heliyon** (Cell Press / Elsevier) — Free APC, SCI indexed, ~1-2 month review
**Backup:** IEEE Access, Computers and Electronics in Agriculture
