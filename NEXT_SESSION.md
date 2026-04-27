# NEXT SESSION — Paper Draft Rewrite & Submission Prep

## Status: PIPELINE RE-RUN COMPLETE ✅ (v3.1)

### Completed
- [x] v3.1 code fixes: leak-free Pipeline + class_weight='balanced'
- [x] Pipeline re-run with corrected methodology (5 min, all 70 runs)
- [x] README updated with new results
- [x] All 30 figures, 21 tables, 4 JSON metrics regenerated

### Key Results (v3.1 — Leak-Free)
- **Best Primary:** RandomForest, all_7 — Acc=0.9950±0.0009, κ=0.9948
- **Best Secondary (Real):** RandomForest, sec_mi_top_6 — Acc=0.9125±0.0077, κ=0.8364
- **SHAP top features:** Humidity > Rainfall > K > N
- **Robustness:** 96.64% (7-day) → 43.82% (90-day drift)
- **Cross-dataset:** P most consistent (0.804), K least (0.293)

### What Changed vs v3.0
- Secondary best improved: RF 91.25% (was GB 90.68%) — class_weight helping
- GaussianNB still fails on secondary (50.91%) — expected, no class weighting for NB
- Ablation now uses MI per fold (not consensus pre-selection)
- All results are honest (no data leakage)

## Next Steps

### 1. Rewrite paper_draft.md (CRITICAL)
The paper still describes the OLD methodology. Must update:
- Replace "nested cross-validation" → "5-fold stratified CV with per-fold MI feature selection via Pipeline"
- Replace "leak-free nested CV" → honest Pipeline description
- Update all result tables with v3.1 numbers
- Replace consensus ablation with MI-per-fold ablation
- Update subset names: top_5 → mi_top_5, etc.
- Add class_weight='balanced' to methodology section
- Strengthen limitations (acknowledge GaussianNB failure on imbalanced)

### 2. Update technical_critique.md
Mark resolved items:
- Critique 1 (data leakage): FIXED — Pipeline per fold
- Critique 3 (class imbalance): PARTIALLY FIXED — class_weight added
- Remaining: SHAP depth, no Optuna tuning

### 3. Format for Target Journal
**Primary target:** Heliyon (free APC, SCI indexed)
- Template: https://www.cell.com/heliyon/author-guidelines
- Convert paper_draft.md to Heliyon format

### 4. Prepare Supplementary Materials
- `master_results.csv` → Supplementary Table S1
- All figures (01-14) as supplementary figures
- Code availability statement → GitHub repo

### 5. Write Cover Letter
Highlight:
- Dual-dataset design with real validation data
- Leak-free Pipeline methodology (per-fold FS + scaling)
- class_weight='balanced' for real-world imbalance
- Cross-dataset feature consistency analysis

### 6. Push & Submit
```bash
git add -A && git commit -m "v3.1: re-run results + updated README" && git push origin main
```
