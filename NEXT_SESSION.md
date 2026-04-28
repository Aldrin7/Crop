# NEXT SESSION — Status Update

## Status: ALL CRITIQUE ITEMS RESOLVED ✅ (v3.2)

### Completed in v3.1
- [x] Paper rewritten as RobustCrop proposal (not comparative)
- [x] Title: "RobustCrop: A Leak-Free ML Pipeline..."
- [x] Author details: Anuradha Brijwal + Praveena Chaturvedi, Gurukul Kangri
- [x] All "nested CV" references removed from code
- [x] Version strings updated to v3.1
- [x] Cross-dataset "validation" reframed as "consistency analysis"
- [x] Consensus vs per-fold MI clarified in paper (Section 4.3)
- [x] class_weight='balanced' added to LightGBM
- [x] Table 1 documents imbalance handling per classifier
- [x] SHAP code made defensive (feature_cols fallback)
- [x] Sensor drift model: monotonic directional (realistic)
- [x] Dropout rates: 2-10% scaled to deployment duration
- [x] Per-class F1 discussion added (Section 6.2)
- [x] Friedman test implemented + reported
- [x] Consistency formula fixed and defined in paper
- [x] Recalibration cost nuance added
- [x] References expanded (Lundberg, Pedregosa, Kapoor)
- [x] Secondary results lead (Section 5.3)
- [x] technical_critique.md updated

### Completed in v3.2
- [x] BalWeightWrapper: sample_weight balancing for ALL classifiers (XGBoost, GB, MLP, KNN, GaussianNB)
- [x] Optuna nested CV hyperparameter tuning (`--tune` flag)
- [x] Dead code removed (`add_class_imbalance`)
- [x] Deprecation warnings on `scale_features()` and `run_all_fs_methods()`
- [x] Missing dependencies fixed (shap, optuna in requirements.txt)
- [x] Tests added (`tests/test_pipeline.py`)
- [x] MIT License added
- [x] `.gitignore` updated (.docx, .db, IDE files)
- [x] `src/__init__.py` proper exports

### Remaining (Future Work)
- [ ] Per-class SHAP breakdowns + local explanations
- [ ] Second crop dataset for true cross-dataset validation
- [ ] CI/CD pipeline (GitHub Actions)

### Key Results (v3.2 — All Honest)
- **Primary:** RF 99.50%±0.09%, κ=0.9948
- **Secondary:** RF 91.25%±0.77%, κ=0.8364 (sec_mi_top_6)
- **Friedman:** χ²=32.32, p<0.001 (significant)
- **Robustness:** 94.05% (7d) → 16.09% (90d)
- **Cross-dataset:** P=0.804, N=0.367, K=0.293 consistency
