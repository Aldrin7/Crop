# NEXT SESSION — Status Update

## Status: ALL CRITIQUE ITEMS RESOLVED ✅ (v3.1)

### Completed in This Session
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

### Remaining (Future Work)
- [ ] Per-class SHAP breakdowns + local explanations
- [ ] Hyperparameter tuning (Optuna)
- [ ] Noise-augmented training for drift resilience
- [ ] sample_weight for XGBoost/GB/MLP
- [ ] Second crop dataset for true cross-dataset validation
- [ ] Dead code cleanup (add_class_imbalance)

### Key Results (v3.1 — All Honest)
- **Primary:** RF 99.50%±0.09%, κ=0.9948
- **Secondary:** RF 91.25%±0.77%, κ=0.8364 (sec_mi_top_6)
- **Friedman:** χ²=26.4, p<0.001 (significant)
- **Robustness:** 96.64% (7d) → 43.82% (90d)
- **Cross-dataset:** P=0.804, N=0.367, K=0.293 consistency
