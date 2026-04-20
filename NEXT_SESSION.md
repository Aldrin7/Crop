# NEXT SESSION — Paper Draft & Submission Prep

## Status: ALL PIPELINE SESSIONS COMPLETE ✅

### Completed
- [x] Session 1: Data & EDA (both datasets) — 7 figures
- [x] Session 2: Preprocessing + 6 FS methods on both datasets
- [x] Session 3: Nested CV training — 70 runs (10 clf × 7 subsets)
- [x] Session 4: SHAP + Calibration + Robustness + Cross-dataset validation
- [x] Session 5: Paper artifacts (master table, summary JSON)
- [x] Paper draft written (`paper_draft.md`)

### Key Results
- **Best Primary:** RandomForest, all_7 — Acc=0.9950±0.0009, κ=0.9948
- **Best Secondary (Real):** GradientBoosting, sec_all_12 — Acc=0.9068±0.0163, κ=0.8267
- **SHAP top features:** Humidity > Rainfall > K > N
- **Robustness:** 96.36% (7-day) → 41.59% (90-day drift)
- **14 figures, 21 tables, 4 JSON metrics**

## Next Steps

### 1. Review Paper Draft
```bash
cat paper_draft.md
```
Review for accuracy, add author details, affiliations.

### 2. Add Author Info
Edit `paper_draft.md` — add:
- Author names and affiliations
- Corresponding author email
- Acknowledgements

### 3. Format for Target Journal
**Primary target:** Computers and Electronics in Agriculture (Elsevier)
- Convert to LaTeX or Word using Elsevier template
- Template: https://www.elsevier.com/authors/policies-and-guidelines/latex-instructions

**Fast backup:** Heliyon (free APC, SCI indexed)
- Template: https://www.cell.com/heliyon/author-guidelines

### 4. Prepare Supplementary Materials
- `master_results.csv` → Supplementary Table S1
- All figures (01-14) as supplementary figures
- Code availability statement pointing to GitHub repo

### 5. Write Cover Letter
Brief summary highlighting:
- Dual-dataset design with real validation data
- Cross-dataset feature consistency analysis
- Leak-free nested CV methodology

### 6. Push & Submit
```bash
git add -A && git commit -m "Add paper draft" && git push origin main
```
