#!/usr/bin/env python3
"""
Crop Recommendation Pipeline v3.1
============================================
Dual-dataset design with cross-dataset feature consistency analysis:
  Primary:   Crop Recommendation (semi-synthetic, 2200 samples, 22 classes)
  Secondary: Soil Fertility (real lab measurements, 880 samples, 3 classes)

Fixes all peer-review critiques:
  2.1 — Semi-synthetic dataset acknowledged + real secondary + degradation variants
  2.2 — Feature selection INSIDE CV loop (no leakage)
  2.3 — Sensor degradation grounded in empirical literature
  3.1 — Modular codebase (src/*.py)
  3.2 — Cohen's Kappa, MCC, Brier, ECE (no redundant balanced metrics)
  3.3 — SHAP + GaussianNB calibration analysis
  4.1 — NEW: Cross-dataset validation on shared feature space (N, P, K, pH)

Usage:
  python pipeline.py --session 1   # Data & EDA (both datasets)
  python pipeline.py --session 2   # Preprocessing + descriptive FS
  python pipeline.py --session 3   # Training (5-fold stratified CV, leak-free Pipeline)
  python pipeline.py --session 4   # Evaluation + SHAP + calibration + cross-dataset consistency
  python pipeline.py --session 5   # Paper artifacts
  python pipeline.py --all
"""
import sys, os, time, json, argparse, logging, warnings
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))
from src.config import *
from src.utils import *
from src.data_loader import (
    load_primary, load_secondary, load_secondary_variants,
    get_shared_features, get_dataset_summary,
)
from src.preprocessing import prepare_data, detect_outliers, handle_missing, encode_target
from src.feature_selection import run_all_fs_methods, TopKFromScores, RFESelector
from src.models import all_classifiers
from src.evaluation import compute_metrics, friedman_test, nemenyi_critical_difference
from src.explainability import (compute_shap_values, analyze_gaussian_nb_calibration,
                                 correlation_violation_report)
from src.noise_injection import degrade_dataset

warnings.filterwarnings('ignore')
sns.set_theme(style='whitegrid', palette='husl', font_scale=1.1)

log = None  # set in main()


# ═══════════════════════════════════════════════════════════════════════════════
# SESSION 1 — DATA ACQUISITION & EDA (BOTH DATASETS)
# ═══════════════════════════════════════════════════════════════════════════════
def session1():
    log.info("="*60 + "\nSESSION 1: DATA ACQUISITION & EDA\n" + "="*60)

    if ckpt_exists('s1'):
        log.info("Session 1 checkpoint exists, loading...")
        return load_ckpt('s1')

    # ── 1.1 Load PRIMARY dataset ──────────────────────────────────────────
    df_primary = load_primary()
    summary_primary = get_dataset_summary(df_primary, TARGET)
    log.info(f"PRIMARY: {summary_primary['n_samples']} samples, "
             f"{summary_primary['n_classes']} classes")
    log.info(f"  NOTE: Semi-synthetic (augmented from Indian agricultural statistics). "
             f"Perfect balance ({summary_primary['min_class']}/class) — does NOT "
             f"reflect real-world distribution.")

    # ── 1.2 Load SECONDARY dataset (REAL) ─────────────────────────────────
    df_secondary = load_secondary()
    summary_secondary = get_dataset_summary(df_secondary, SECONDARY_TARGET)
    log.info(f"SECONDARY (REAL): {summary_secondary['n_samples']} samples, "
             f"{summary_secondary['n_classes']} fertility classes")
    log.info(f"  Imbalance ratio: {summary_secondary['imbalance_ratio']}x "
             f"(min={summary_secondary['min_class']}, max={summary_secondary['max_class']})")
    log.info(f"  Missing values: {summary_secondary['missing']} (real lab dropout)")

    # ── 1.3 Shared feature space ──────────────────────────────────────────
    primary_shared, secondary_shared, shared_feats = get_shared_features(
        df_primary, df_secondary)
    log.info(f"\nShared feature space: {shared_feats}")

    # ── 1.4 Degradation variants (primary) ────────────────────────────────
    log.info("\nGenerating literature-grounded sensor degradation variants...")
    variants = load_secondary_variants(n_variants=3)
    for name, vdf in variants.items():
        missing = vdf[FEATURES].isnull().sum().sum()
        log.info(f"  Variant '{name}': {missing} injected missing values")

    # ── 1.5 EDA ───────────────────────────────────────────────────────────
    log.info("\n--- EDA ---")

    # Descriptive stats — PRIMARY
    desc_primary = df_primary[FEATURES].describe().T
    desc_primary['skewness'] = df_primary[FEATURES].skew()
    desc_primary['kurtosis'] = df_primary[FEATURES].kurtosis()
    save_table(desc_primary, 'descriptive_stats_primary')

    # Descriptive stats — SECONDARY
    desc_secondary = df_secondary[SECONDARY_FEATURES].describe().T
    desc_secondary['skewness'] = df_secondary[SECONDARY_FEATURES].skew()
    desc_secondary['kurtosis'] = df_secondary[SECONDARY_FEATURES].kurtosis()
    save_table(desc_secondary, 'descriptive_stats_secondary')

    # ── Figures ───────────────────────────────────────────────────────────

    # Figure 1: Primary distributions
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    for i, feat in enumerate(FEATURES):
        ax = axes[i // 4, i % 4]
        ax.hist(df_primary[feat], bins=40, density=True, alpha=0.7,
                color='steelblue', edgecolor='white')
        df_primary[feat].plot.kde(ax=ax, color='red', linewidth=2)
        ax.set_title(feat, fontweight='bold')
    axes[1, 3].set_visible(False)
    fig.suptitle('Primary Dataset — Feature Distributions (Semi-Synthetic)',
                 fontsize=16, fontweight='bold')
    fig.tight_layout(); save_fig(fig, '01_primary_distributions')

    # Figure 2: Secondary distributions (REAL)
    fig, axes = plt.subplots(3, 4, figsize=(22, 14))
    for i, feat in enumerate(SECONDARY_FEATURES):
        ax = axes[i // 4, i % 4]
        ax.hist(df_secondary[feat].dropna(), bins=40, density=True,
                alpha=0.7, color='#E91E63', edgecolor='white')
        df_secondary[feat].dropna().plot.kde(ax=ax, color='darkred', linewidth=2)
        ax.set_title(feat, fontweight='bold')
    fig.suptitle('Secondary Dataset — Real Soil Lab Measurements',
                 fontsize=16, fontweight='bold')
    fig.tight_layout(); save_fig(fig, '02_secondary_distributions')

    # Figure 3: Correlation heatmap — PRIMARY
    fig, ax = plt.subplots(figsize=(9, 7))
    corr = df_primary[FEATURES].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt='.3f', cmap='RdBu_r',
                center=0, square=True, linewidths=0.5, ax=ax)
    ax.set_title('Primary — Feature Correlation\n(Naive Bayes independence check)',
                 fontweight='bold')
    fig.tight_layout(); save_fig(fig, '03_primary_correlation')

    # Figure 4: Correlation heatmap — SECONDARY
    fig, ax = plt.subplots(figsize=(11, 9))
    corr_s = df_secondary[SECONDARY_FEATURES].corr()
    mask_s = np.triu(np.ones_like(corr_s, dtype=bool))
    sns.heatmap(corr_s, mask=mask_s, annot=True, fmt='.2f', cmap='RdBu_r',
                center=0, square=True, linewidths=0.5, ax=ax, annot_kws={'fontsize': 8})
    ax.set_title('Secondary — Real Soil Feature Correlations', fontweight='bold')
    fig.tight_layout(); save_fig(fig, '04_secondary_correlation')

    # Figure 5: Class distributions (side by side)
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    df_primary[TARGET].value_counts().plot.bar(ax=axes[0], color='steelblue',
                                                edgecolor='white')
    axes[0].set_title(f'Primary — {N_CLASSES_PRIMARY} Crop Classes (Balanced)',
                      fontweight='bold')
    axes[0].set_ylabel('Samples')

    df_secondary[SECONDARY_TARGET].value_counts().sort_index().plot.bar(
        ax=axes[1], color=['#4CAF50', '#FF9800', '#F44336'], edgecolor='white')
    axes[1].set_title('Secondary — Soil Fertility Classes (Real Imbalance)',
                      fontweight='bold')
    axes[1].set_ylabel('Samples')
    axes[1].set_xticklabels(['High (0)', 'Medium (1)', 'Low (2)'], rotation=0)
    fig.suptitle('Class Distribution Comparison', fontsize=14, fontweight='bold')
    fig.tight_layout(); save_fig(fig, '05_class_distributions')

    # Figure 6: Degradation comparison
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for idx, (scenario, vdf) in enumerate(variants.items()):
        ax = axes[idx]
        ax.hist(df_primary['humidity'], bins=30, alpha=0.5, label='Original', density=True)
        ax.hist(vdf['humidity'].dropna(), bins=30, alpha=0.5,
                label=f'Degraded ({scenario})', density=True)
        ax.set_title(f'Humidity: {scenario} degradation', fontweight='bold')
        ax.legend()
    fig.suptitle('Sensor Degradation Effect on Humidity', fontsize=14, fontweight='bold')
    fig.tight_layout(); save_fig(fig, '06_degradation_comparison')

    # Figure 7: Shared feature space comparison (N, P, K, pH)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for i, feat in enumerate(shared_feats):
        ax = axes[i // 2, i % 2]
        # Map 'ph' → 'pH' for secondary; also try exact + variants
        sec_candidates = [feat, feat.upper(), feat.lower(), feat.capitalize(), 'pH', 'PH']
        sec_feat = None
        for cand in sec_candidates:
            if cand in df_secondary.columns:
                sec_feat = cand
                break
        if sec_feat is None:
            continue  # skip if no matching column
        ax.hist(df_primary[feat].dropna(), bins=40, alpha=0.5, density=True,
                label='Primary', color='steelblue')
        ax.hist(df_secondary[sec_feat].dropna(),
                bins=40, alpha=0.5, density=True, label='Secondary (Real)', color='#E91E63')
        ax.set_title(f'{feat} — Distribution Comparison', fontweight='bold')
        ax.legend(); ax.grid(True, alpha=0.3)
    fig.suptitle('Shared Feature Space: Primary vs Secondary Dataset',
                 fontsize=14, fontweight='bold')
    fig.tight_layout(); save_fig(fig, '07_shared_features_comparison')

    # Save all data
    data = {
        'primary': df_primary,
        'secondary': df_secondary,
        'primary_shared': primary_shared,
        'secondary_shared': secondary_shared,
        'shared_features': shared_feats,
        'variants': variants,          # raw degraded DataFrames (for leak-free pipeline)
        'variants_raw': variants,      # alias: unscaled degraded data
        'summary_primary': summary_primary,
        'summary_secondary': summary_secondary,
    }
    save_ckpt('s1', data)

    # Summary JSON
    save_json({
        'session': 1,
        'primary': summary_primary,
        'secondary': summary_secondary,
        'shared_features': shared_feats,
        'degradation_variants': list(variants.keys()),
    }, 'session1_summary')

    mark_done(1)
    log.info("SESSION 1 COMPLETE ✓")
    return data


# ═══════════════════════════════════════════════════════════════════════════════
# SESSION 2 — PREPROCESSING + DESCRIPTIVE FEATURE SELECTION
# ═══════════════════════════════════════════════════════════════════════════════
def session2():
    log.info("="*60 + "\nSESSION 2: PREPROCESSING + DESCRIPTIVE FS\n" + "="*60)

    if ckpt_exists('s2'):
        return load_ckpt('s2')
    s1 = load_ckpt('s1')
    if s1 is None:
        raise RuntimeError("Run session 1 first")

    # ── 2.1 Preprocess PRIMARY ────────────────────────────────────────────
    df_primary = s1['primary']
    prep_primary = prepare_data(df_primary)
    log.info(f"Primary preprocessing: {prep_primary['outlier_report']}")

    # ── 2.2 Preprocess SECONDARY (REAL — with missing value handling) ─────
    df_secondary = s1['secondary']
    log.info("\nPreprocessing secondary (REAL) dataset...")

    # Handle missing values first
    df_sec_clean = handle_missing(df_secondary.copy(), feature_cols=SECONDARY_FEATURES)
    prep_secondary = prepare_data(df_sec_clean, target_col=SECONDARY_TARGET,
                                   feature_cols=SECONDARY_FEATURES)
    log.info(f"Secondary preprocessing: {prep_secondary['outlier_report']}")
    log.info(f"  Class imbalance detected: "
             f"{dict(zip(*np.unique(prep_secondary['y_train'], return_counts=True)))}")

    # ── 2.3 Descriptive FS on PRIMARY ─────────────────────────────────────
    log.info("\nDescriptive Feature Selection — Primary (analysis only)")
    X_tr = prep_primary['X_train']
    y_tr = prep_primary['y_train']
    fs_results = run_all_fs_methods(X_tr.values, y_tr, FEATURES)

    # Figure 8: FS comparison
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    methods = ['mutual_info', 'chi2', 'rf_importance', 'extratrees', 'lasso', 'consensus']
    for idx, method in enumerate(methods):
        ax = axes[idx // 3, idx % 3]
        if method == 'consensus':
            df_plot = fs_results['consensus'][['feature', 'mean_score']].sort_values('mean_score')
            ax.barh(df_plot['feature'], df_plot['mean_score'], color='#607D8B')
            ax.set_title('Consensus Ranking', fontweight='bold')
        else:
            col = 'score' if 'score' in fs_results[method].columns else 'ranking'
            df_plot = fs_results[method].sort_values(col)
            ax.barh(df_plot['feature'], df_plot[col], color=f'C{idx}')
            ax.set_title(method.replace('_', ' ').title(), fontweight='bold')
    fig.suptitle('Feature Selection Methods — Primary Dataset (Descriptive)',
                 fontsize=14, fontweight='bold')
    fig.tight_layout(); save_fig(fig, '08_feature_selection')

    # Save FS tables
    for method, df_fs in fs_results.items():
        save_table(df_fs, f'fs_{method}')

    # ── 2.4 Descriptive FS on SECONDARY ───────────────────────────────────
    log.info("\nDescriptive Feature Selection — Secondary (analysis only)")
    X_tr_s = prep_secondary['X_train']
    y_tr_s = prep_secondary['y_train']
    fs_secondary = run_all_fs_methods(X_tr_s.values, y_tr_s, SECONDARY_FEATURES)

    # Figure 9: FS comparison — secondary
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    for idx, method in enumerate(methods):
        ax = axes[idx // 3, idx % 3]
        if method == 'consensus':
            df_plot = fs_secondary['consensus'][['feature', 'mean_score']].sort_values('mean_score')
            ax.barh(df_plot['feature'], df_plot['mean_score'], color='#E91E63')
            ax.set_title('Consensus Ranking', fontweight='bold')
        else:
            col = 'score' if 'score' in fs_secondary[method].columns else 'ranking'
            df_plot = fs_secondary[method].sort_values(col)
            ax.barh(df_plot['feature'], df_plot[col], color=f'C{idx}')
            ax.set_title(method.replace('_', ' ').title(), fontweight='bold')
    fig.suptitle('Feature Selection Methods — Secondary Dataset (Real Lab Data)',
                 fontsize=14, fontweight='bold')
    fig.tight_layout(); save_fig(fig, '09_feature_selection_secondary')

    for method, df_fs in fs_secondary.items():
        save_table(df_fs, f'fs_sec_{method}')

    # ── 2.5 Cross-dataset FS consistency ──────────────────────────────────
    log.info("\nCross-dataset FS consistency (N, P, K shared features)")
    shared = ['N', 'P', 'K']
    primary_consensus = fs_results['consensus']
    secondary_consensus = fs_secondary['consensus']

    consistency = pd.DataFrame({'feature': shared})
    for label, cons_df in [('primary', primary_consensus), ('secondary', secondary_consensus)]:
        sub = cons_df[cons_df['feature'].isin(shared)][['feature', 'mean_score']].copy()
        sub.columns = ['feature', f'{label}_score']
        consistency = consistency.merge(sub, on='feature', how='left')
    consistency['score_diff'] = abs(consistency['primary_score'] - consistency['secondary_score'])
    consistency = consistency.sort_values('score_diff')
    save_table(consistency, 'cross_dataset_fs_consistency')
    log.info(f"FS consistency:\n{consistency.to_string(index=False)}")

    # ── 2.6 Process degradation variants ──────────────────────────────────
    variant_preps = {}
    for vname, vdf in s1['variants'].items():
        variant_preps[vname] = prepare_data(vdf)

    data = {
        'prep_primary': prep_primary,
        'prep_secondary': prep_secondary,
        'fs_results': fs_results,
        'fs_secondary': fs_secondary,
        'variant_preps': variant_preps,
        'consensus': fs_results['consensus'],
        'consensus_secondary': fs_secondary['consensus'],
        'shared_features': shared,
    }
    save_ckpt('s2', data)
    mark_done(2)
    log.info("SESSION 2 COMPLETE ✓")
    return data


# ═══════════════════════════════════════════════════════════════════════════════
# SESSION 3 — LEAK-FREE CV TRAINING + SECONDARY
# ═══════════════════════════════════════════════════════════════════════════════
def session3():
    """
    Critique 2.2 fix: Feature selection INSIDE the CV loop.
    5-fold stratified CV with leak-free Pipeline per fold.
    Trains on BOTH primary and secondary datasets separately.
    """
    log.info("="*60 + "\nSESSION 3: LEAK-FREE CV TRAINING\n" + "="*60)

    if ckpt_exists('s3'):
        return load_ckpt('s3')
    s2 = load_ckpt('s2')
    if s2 is None:
        raise RuntimeError("Run session 2 first")

    from sklearn.base import clone
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.feature_selection import SelectKBest, mutual_info_classif

    classifiers = all_classifiers()
    all_results = {}
    best_models = {}

    # ── 3.1 PRIMARY dataset training (LEAK-FREE: FS inside CV via Pipeline) ──
    prep = s2['prep_primary']
    X_raw = pd.concat([prep['X_train_raw'], prep['X_test_raw']])  # unscaled
    y = np.concatenate([prep['y_train'], prep['y_test']])

    # Ablation: MI-based top-k selection PER FOLD (no global pre-selection)
    primary_k_subsets = {
        'all_7': None,   # no selection — use all 7 features
        'mi_top_5': 5,
        'mi_top_4': 4,
        'mi_top_3': 3,
    }

    log.info("═══ PRIMARY DATASET (leak-free: MI selection per fold) ═══")
    for subset_name, k in primary_k_subsets.items():
        log.info(f"\n{'='*40}\nFeature subset: {subset_name} (k={k or 'all'})\n{'='*40}")
        _train_classifiers(X_raw, y, classifiers, subset_name,
                           all_results, best_models, le=prep['label_encoder'],
                           feature_cols=FEATURES, k=k)

    # ── 3.2 SECONDARY dataset training (LEAK-FREE) ─────────────────────────
    log.info("\n═══ SECONDARY DATASET (REAL, leak-free) ═══")
    prep_s = s2['prep_secondary']
    X_sec_raw = pd.concat([prep_s['X_train_raw'], prep_s['X_test_raw']])  # unscaled
    y_sec = np.concatenate([prep_s['y_train'], prep_s['y_test']])

    secondary_k_subsets = {
        'sec_all_12': None,
        'sec_mi_top_6': 6,
        'sec_mi_top_4': 4,
    }

    for subset_name, k in secondary_k_subsets.items():
        log.info(f"\n{'='*40}\nFeature subset: {subset_name} (k={k or 'all'})\n{'='*40}")
        _train_classifiers(X_sec_raw, y_sec, classifiers, subset_name,
                           all_results, best_models, le=prep_s['label_encoder'],
                           feature_cols=SECONDARY_FEATURES, k=k)

    # ── Summary ───────────────────────────────────────────────────────────
    all_k_subsets = {**primary_k_subsets, **secondary_k_subsets}
    save_ckpt('s3', {'all_results': all_results, 'best_models': best_models,
                     'feature_subsets': {k: f'mi_top_{v}' if v else 'all' for k, v in all_k_subsets.items()}})

    _save_training_summary(all_results)
    mark_done(3)
    log.info("SESSION 3 COMPLETE ✓")


def _train_classifiers(X_raw, y, classifiers, subset_name, all_results, best_models,
                       le=None, feature_cols=None, k=None):
    """Train all classifiers with LEAK-FREE CV: scaler + feature selection inside each fold.

    Args:
        X_raw: Unscaled feature DataFrame/array (scaling done per-fold via Pipeline).
        y: Target labels.
        classifiers: Dict of {name: estimator}.
        subset_name: Label for this ablation run (e.g. 'all_7', 'mi_top_5').
        all_results: Mutable dict to accumulate results.
        best_models: Mutable dict to store fitted models.
        le: Label encoder (for reporting).
        feature_cols: Column names for the full feature set.
        k: If set, use SelectKBest(mutual_info) to pick top-k features per fold.
            If None, use all features (no selection inside CV).
    """
    from sklearn.model_selection import StratifiedKFold
    from sklearn.base import clone
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.feature_selection import SelectKBest, mutual_info_classif

    n_total = X_raw.shape[1]
    if k is None or k >= n_total:
        k_eff = n_total  # no selection
    else:
        k_eff = k

    subset_results = {}
    for clf_name, clf in classifiers.items():
        log.info(f"  Training {clf_name} (k={k_eff}/{n_total})...")
        t0 = time.time()
        try:
            # Build leak-free pipeline: scaler → feature selection → classifier
            steps = [('scaler', StandardScaler())]
            if k_eff < n_total:
                steps.append(('selector', SelectKBest(mutual_info_classif, k=k_eff)))
            steps.append(('clf', clone(clf)))
            pipe = Pipeline(steps)

            outer_cv = StratifiedKFold(n_splits=CV_OUTER, shuffle=True,
                                       random_state=RANDOM_STATE)
            outer_scores, fold_metrics = [], []

            for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X_raw, y)):
                X_tr = X_raw.iloc[train_idx] if hasattr(X_raw, 'iloc') else X_raw[train_idx]
                X_te = X_raw.iloc[test_idx] if hasattr(X_raw, 'iloc') else X_raw[test_idx]
                y_tr, y_te = y[train_idx], y[test_idx]

                fold_pipe = clone(pipe)
                fold_pipe.fit(X_tr, y_tr)
                y_pred = fold_pipe.predict(X_te)
                y_proba = (fold_pipe.predict_proba(X_te)
                           if hasattr(fold_pipe, 'predict_proba') else None)

                m = compute_metrics(y_te, y_pred, y_proba)
                fold_metrics.append(m)
                outer_scores.append(m['accuracy'])

            elapsed = time.time() - t0
            result = {
                'accuracy_mean': float(np.mean(outer_scores)),
                'accuracy_std': float(np.std(outer_scores)),
                'fold_scores': outer_scores,
                'cohens_kappa': float(np.mean([f['cohens_kappa'] for f in fold_metrics])),
                'mcc': float(np.mean([f['mcc'] for f in fold_metrics])),
                'macro_f1': float(np.mean([f['macro_f1'] for f in fold_metrics])),
                'brier_mean': float(np.mean([f.get('brier_mean', 0) for f in fold_metrics])),
                'ece': float(np.mean([f.get('ece', 0) for f in fold_metrics])),
                'train_time': elapsed,
                'subset': subset_name,
                'n_features': k_eff,
                'selection_method': 'MI-per-fold' if k_eff < n_total else 'none',
            }
            subset_results[clf_name] = result

            # Train final pipeline on full data (for downstream use)
            final_pipe = clone(pipe)
            final_pipe.fit(X_raw, y)
            best_models[f"{subset_name}__{clf_name}"] = final_pipe

            log.info(f"    Acc={result['accuracy_mean']:.4f}±{result['accuracy_std']:.4f} "
                     f"Kappa={result['cohens_kappa']:.4f} MCC={result['mcc']:.4f} "
                     f"[{elapsed:.1f}s]")
        except Exception as e:
            log.error(f"    FAILED: {e}")
            subset_results[clf_name] = {'error': str(e)}

    all_results[subset_name] = subset_results


def _save_training_summary(all_results):
    """Save training results table."""
    rows = []
    for subset, res_dict in all_results.items():
        for clf, r in res_dict.items():
            if 'error' not in r:
                rows.append({
                    'Features': subset, 'Classifier': clf,
                    'Accuracy': f"{r['accuracy_mean']:.4f}±{r['accuracy_std']:.4f}",
                    'Kappa': f"{r['cohens_kappa']:.4f}", 'MCC': f"{r['mcc']:.4f}",
                    'Macro-F1': f"{r['macro_f1']:.4f}", 'Brier': f"{r['brier_mean']:.4f}",
                    'ECE': f"{r['ece']:.4f}", 'Time(s)': f"{r['train_time']:.1f}",
                })
    save_table(pd.DataFrame(rows), 'cv_results')

    # Figure: accuracy comparison
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    primary_subsets = [s for s in all_results if not s.startswith('sec_')]
    secondary_subsets = [s for s in all_results if s.startswith('sec_')]

    for idx, (subsets, title) in enumerate([(primary_subsets, 'Primary'),
                                            (secondary_subsets, 'Secondary (Real)')]):
        ax = axes[idx]
        for subset in subsets:
            clfs, vals = [], []
            for clf, r in all_results[subset].items():
                if 'error' not in r:
                    clfs.append(clf); vals.append(r['accuracy_mean'])
            if vals:
                ax.plot(clfs, vals, 'o-', label=subset, markersize=6)
        ax.set_xticklabels(clfs, rotation=45, ha='right', fontsize=8)
        ax.set_ylabel('Accuracy'); ax.set_title(f'{title} — Accuracy', fontweight='bold')
        ax.legend(); ax.grid(True, alpha=0.3)
    fig.suptitle('Leak-Free CV Results', fontsize=14, fontweight='bold')
    fig.tight_layout(); save_fig(fig, '10_cv_comparison')


# ═══════════════════════════════════════════════════════════════════════════════
# SESSION 4 — EVALUATION + SHAP + CALIBRATION + CROSS-DATASET
# ═══════════════════════════════════════════════════════════════════════════════
def session4():
    log.info("="*60 + "\nSESSION 4: EVALUATION + SHAP + CALIBRATION\n" + "="*60)

    s1 = load_ckpt('s1'); s2 = load_ckpt('s2'); s3 = load_ckpt('s3')
    if any(x is None for x in [s1, s2, s3]):
        raise RuntimeError("Run sessions 1-3 first")

    prep = s2['prep_primary']
    all_results = s3['all_results']
    best_models = s3['best_models']
    X_train_raw = prep['X_train_raw']; X_test_raw = prep['X_test_raw']
    y_train = prep['y_train']; y_test = prep['y_test']
    le = prep['label_encoder']

    # ── 4.1 SHAP Analysis ────────────────────────────────────────────────
    log.info("\nPhase 4.1: SHAP Explainability")
    top3 = sorted(all_results['all_7'].items(),
                  key=lambda x: x[1].get('accuracy_mean', 0), reverse=True)[:3]

    for clf_name, _ in top3:
        if 'error' in all_results['all_7'].get(clf_name, {}):
            continue
        model_key = f"all_7__{clf_name}"
        if model_key not in best_models:
            continue
        pipe = best_models[model_key]  # Pipeline(scaler → [selector] → clf)
        log.info(f"  Computing SHAP for {clf_name}...")

        # Extract scaler + selector from pipeline, apply to raw data
        from sklearn.pipeline import Pipeline as SkPipeline
        X_train_scaled = pipe.named_steps['scaler'].transform(X_train_raw)
        X_test_scaled = pipe.named_steps['scaler'].transform(X_test_raw)
        if 'selector' in pipe.named_steps:
            X_train_scaled = pipe.named_steps['selector'].transform(X_train_scaled)
            X_test_scaled = pipe.named_steps['selector'].transform(X_test_scaled)
            sel_idx = pipe.named_steps['selector'].get_support(indices=True)
            shap_features = [FEATURES[i] for i in sel_idx if i < len(FEATURES)]
        else:
            shap_features = FEATURES
        clf_model = pipe.named_steps['clf']

        shap_vals, _ = compute_shap_values(clf_model, X_train_scaled, X_test_scaled, shap_features)
        if shap_vals is not None:
            try:
                # Handle various SHAP output shapes
                if isinstance(shap_vals, list):
                    # Multi-class: list of arrays (n_samples, n_features) per class
                    mean_shap = np.mean([np.abs(sv).mean(axis=0) for sv in shap_vals], axis=0)
                elif shap_vals.ndim == 3:
                    # Shape: (n_samples, n_features, n_classes)
                    mean_shap = np.abs(shap_vals).mean(axis=(0, 2))
                elif shap_vals.ndim == 2:
                    # Shape: (n_samples, n_features)
                    mean_shap = np.abs(shap_vals).mean(axis=0)
                else:
                    log.warning(f"  Unexpected SHAP shape: {shap_vals.shape}")
                    continue
                mean_shap = np.atleast_1d(mean_shap).flatten()[:len(shap_features)]
                feat_imp = pd.DataFrame({'feature': shap_features[:len(mean_shap)], 'mean_|SHAP|': mean_shap}
                                        ).sort_values('mean_|SHAP|')
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.barh(feat_imp['feature'], feat_imp['mean_|SHAP|'], color='#E91E63')
                ax.set_xlabel('Mean |SHAP value|')
                ax.set_title(f'SHAP — {clf_name}', fontweight='bold')
                fig.tight_layout(); save_fig(fig, f'11_shap_{clf_name}')
            except Exception as e:
                log.warning(f"  SHAP plot failed for {clf_name}: {e}")

    # ── 4.2 GaussianNB Calibration ────────────────────────────────────────
    log.info("\nPhase 4.2: GaussianNB Calibration & Independence Violation")
    nb_key = 'all_7__GaussianNB'
    if nb_key in best_models:
        nb_pipe = best_models[nb_key]
        nb_analysis = analyze_gaussian_nb_calibration(nb_pipe, X_test_raw.values, y_test, le)
        violations = correlation_violation_report(X_train_raw.values, FEATURES)
        nb_analysis['correlation_violations'] = violations
        log.info(f"  Independence violations: {len(violations)} pairs")
        save_json(nb_analysis, 'gaussian_nb_analysis')

    # ── 4.3 Robustness under sensor degradation ──────────────────────────
    log.info("\nPhase 4.3: Robustness Under Sensor Degradation")
    best_clf_name = top3[0][0]
    best_model = best_models.get(f"all_7__{best_clf_name}")

    best_pipe = best_models.get(f"all_7__{best_clf_name}")
    if best_pipe is not None:
        variants_raw = s1['variants']  # raw degraded DataFrames (unscaled)
        robustness = {}

        # Fresh results
        fresh_res = all_results['all_7'][best_clf_name]
        robustness['fresh'] = {
            'accuracy': fresh_res['accuracy_mean'],
            'cohens_kappa': fresh_res['cohens_kappa'],
            'brier_mean': fresh_res['brier_mean'],
        }

        for scenario, vdf in variants_raw.items():
            # Pipeline handles scaling internally — pass raw degraded data
            X_v = vdf[FEATURES].copy()
            y_v = le.transform(vdf[TARGET])
            X_v_clean = handle_missing(X_v, feature_cols=FEATURES)
            y_pred = best_pipe.predict(X_v_clean)
            y_proba = (best_pipe.predict_proba(X_v_clean)
                       if hasattr(best_pipe, 'predict_proba') else None)
            m = compute_metrics(y_v, y_pred, y_proba)
            robustness[scenario] = {
                'accuracy': m['accuracy'], 'cohens_kappa': m['cohens_kappa'],
                'brier_mean': m.get('brier_mean', 0),
            }
            log.info(f"  {scenario}: Acc={m['accuracy']:.4f} Kappa={m['cohens_kappa']:.4f}")

        save_json(robustness, 'robustness_degradation')

        # Figure: Robustness
        scenarios = ['fresh'] + list(variants_raw.keys())
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        for idx, metric in enumerate(['accuracy', 'cohens_kappa', 'brier_mean']):
            ax = axes[idx]
            vals = [robustness.get(s, {}).get(metric, 0) for s in scenarios]
            colors = ['#4CAF50', '#FF9800', '#F44336', '#9C27B0'][:len(scenarios)]
            ax.bar(scenarios, vals, color=colors, edgecolor='white')
            ax.set_title(metric.replace('_', ' ').title(), fontweight='bold')
            for i, v in enumerate(vals):
                ax.text(i, v + 0.002, f'{v:.3f}', ha='center', fontsize=9)
        fig.suptitle(f'Robustness: {best_clf_name}', fontsize=14, fontweight='bold')
        fig.tight_layout(); save_fig(fig, '12_robustness')

    # ── 4.4 Calibration curves ────────────────────────────────────────────
    log.info("\nPhase 4.4: Calibration Curves")
    from sklearn.calibration import calibration_curve
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for idx, (clf_name, _) in enumerate(top3):
        ax = axes[idx]
        model_key = f"all_7__{clf_name}"
        if model_key not in best_models:
            continue
        pipe = best_models[model_key]
        if not hasattr(pipe, 'predict_proba'):
            ax.set_title(f'{clf_name} (no proba)', fontweight='bold'); continue
        y_proba = pipe.predict_proba(X_test_raw)
        for cls_idx in range(min(5, len(le.classes_))):
            y_bin = (y_test == cls_idx).astype(int)
            prob_true, prob_pred = calibration_curve(y_bin, y_proba[:, cls_idx],
                                                     n_bins=10, strategy='uniform')
            ax.plot(prob_pred, prob_true, 'o-', label=le.classes_[cls_idx], markersize=4)
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect')
        ax.set_xlabel('Mean predicted probability'); ax.set_ylabel('Fraction of positives')
        ax.set_title(f'Calibration — {clf_name}', fontweight='bold')
        ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
    fig.suptitle('Calibration Curves', fontsize=14, fontweight='bold')
    fig.tight_layout(); save_fig(fig, '13_calibration')

    # ── 4.5 Per-class F1 heatmap ──────────────────────────────────────────
    log.info("\nPhase 4.5: Per-Class F1 Heatmap")
    per_class_data = {}
    for clf_name, r in all_results['all_7'].items():
        if 'error' in r: continue
        model_key = f"all_7__{clf_name}"
        if model_key not in best_models: continue
        pipe = best_models[model_key]
        y_pred = pipe.predict(X_test_raw)
        from sklearn.metrics import f1_score
        per_class_f1 = f1_score(y_test, y_pred, average=None, labels=range(len(le.classes_)))
        per_class_data[clf_name] = per_class_f1

    if per_class_data:
        heatmap_df = pd.DataFrame(per_class_data, index=le.classes_)
        fig, ax = plt.subplots(figsize=(14, 10))
        sns.heatmap(heatmap_df, annot=True, fmt='.2f', cmap='RdYlGn',
                    vmin=0.5, vmax=1.0, linewidths=0.5, ax=ax)
        ax.set_title('Per-Class F1 Score', fontsize=14, fontweight='bold')
        fig.tight_layout(); save_fig(fig, '14_per_class_heatmap')
        save_table(heatmap_df, 'per_class_f1')

    # ── 4.6 Cross-dataset validation ──────────────────────────────────────
    log.info("\nPhase 4.6: Cross-Dataset Feature Consistency")
    shared = ['N', 'P', 'K']
    primary_cons = s2['consensus']
    secondary_cons = s2['consensus_secondary']

    xval = pd.DataFrame({'feature': shared})
    for label, cons in [('primary', primary_cons), ('secondary', secondary_cons)]:
        sub = cons[cons['feature'].isin(shared)][['feature', 'mean_score']]
        sub.columns = ['feature', f'{label}_rank_score']
        xval = xval.merge(sub, on='feature', how='left')
    xval['consistency'] = 1 - abs(xval['primary_rank_score'] - xval['secondary_rank_score'])
    save_table(xval, 'cross_dataset_consistency')

    log.info(f"Cross-dataset consistency:\n{xval.to_string(index=False)}")

    mark_done(4)
    log.info("SESSION 4 COMPLETE ✓")


# ═══════════════════════════════════════════════════════════════════════════════
# SESSION 5 — PAPER ARTIFACTS
# ═══════════════════════════════════════════════════════════════════════════════
def session5():
    log.info("="*60 + "\nSESSION 5: PAPER ARTIFACTS\n" + "="*60)

    s1 = load_ckpt('s1'); s2 = load_ckpt('s2'); s3 = load_ckpt('s3')
    if any(x is None for x in [s1, s2, s3]):
        raise RuntimeError("Run sessions 1-4 first")

    all_results = s3['all_results']

    # ── 5.1 Master results table ──────────────────────────────────────────
    rows = []
    for subset, res in all_results.items():
        for clf, r in res.items():
            if 'error' not in r:
                rows.append({
                    'Features': subset, 'Classifier': clf,
                    'Accuracy': f"{r['accuracy_mean']:.4f}±{r['accuracy_std']:.4f}",
                    'Kappa': f"{r['cohens_kappa']:.4f}", 'MCC': f"{r['mcc']:.4f}",
                    'Macro-F1': f"{r['macro_f1']:.4f}", 'Brier': f"{r['brier_mean']:.4f}",
                    'ECE': f"{r['ece']:.4f}", 'Time(s)': f"{r['train_time']:.1f}",
                })
    master = pd.DataFrame(rows)
    save_table(master, 'master_results')

    # ── 5.1b Friedman statistical test ───────────────────────────────
    log.info("\nFriedman test (primary all_7 classifiers):")
    primary_results = all_results.get('all_7', {})
    cv_score_dicts = {}
    for clf_name, r in primary_results.items():
        if 'error' not in r and 'fold_scores' in r:
            cv_score_dicts[clf_name] = r['fold_scores']
    if len(cv_score_dicts) >= 3:
        ft = friedman_test(cv_score_dicts)
        if ft:
            log.info(f"  Statistic={ft['statistic']:.4f}, p={ft['p_value']:.6f}, "
                     f"significant={ft['significant']}")
            save_json(ft, 'friedman_test')
            cd = nemenyi_critical_difference(len(cv_score_dicts))
            log.info(f"  Nemenyi CD (alpha=0.05): {cd:.3f}")

    # ── 5.2 Best overall ──────────────────────────────────────────────────
    best_row = None; best_acc = 0
    for subset, res in all_results.items():
        for clf, r in res.items():
            if 'error' not in r and r['accuracy_mean'] > best_acc:
                best_acc = r['accuracy_mean']
                best_row = {'subset': subset, 'clf': clf, **r}
    if best_row:
        log.info(f"\n★ BEST: {best_row['clf']} on {best_row['subset']} — "
                 f"Acc={best_row['accuracy_mean']:.4f} Kappa={best_row['cohens_kappa']:.4f}")

    # ── 5.3 Final summary ─────────────────────────────────────────────────
    final = {
        'title': ('RobustCrop: A Leak-Free Machine Learning Pipeline for Crop Recommendation '
                  'with Sensor Degradation Analysis and Cross-Dataset Feature Consistency'),
        'dual_dataset_design': {
            'primary': 'Crop Recommendation (Kaggle) — semi-synthetic, 2200 samples, 22 classes',
            'secondary': 'Soil Fertility (Kaggle, Rahul Jaiswal) — real lab measurements, '
                         '880 samples, 3 fertility classes, natural imbalance',
            'shared_features': 'N, P, K (pH close match)',
            'cross_dataset_consistency': True,
        },
        'peer_review_fixes': {
            '2.1_semi_synthetic': 'Acknowledged + real secondary dataset + degradation variants',
            '2.2_data_leakage': 'FS inside CV loop — no leakage',
            '2.3_sensor_degradation': 'Literature-grounded (Rana 2019, Lobnik 2011)',
            '3.1_modular': 'src/*.py architecture',
            '3.2_metrics': 'Kappa, MCC, Brier, ECE (no redundant)',
            '3.3_interpretability': 'SHAP + GaussianNB calibration',
            '4.1_cross_dataset': 'Real secondary for feature consistency analysis',
        },
        'best_classifier': best_row['clf'] if best_row else 'N/A',
        'best_accuracy': best_acc,
    }
    save_json(final, 'final_summary')

    # List artifacts
    log.info("\n--- Generated Artifacts ---")
    for name, d in [('Figures', FIG_DIR), ('Tables', TABLE_DIR), ('Metrics', METRIC_DIR)]:
        files = sorted(d.glob('*'))
        log.info(f"\n{name} ({len(files)} files):")
        for f in files:
            log.info(f"  {f.name} ({f.stat().st_size/1024:.1f} KB)")

    mark_done(5)
    log.info("\n" + "="*60 + "\nALL SESSIONS COMPLETE ✓\n" + "="*60)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    global log
    log = setup_logging()
    parser = argparse.ArgumentParser(description='Crop Recommendation Pipeline v3.1')
    parser.add_argument('--session', type=int, choices=[1, 2, 3, 4, 5])
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--skip', type=int, default=0)
    args = parser.parse_args()

    log.info("="*60)
    log.info("CROP RECOMMENDATION — PIPELINE v3.1 (Dual Dataset)")
    log.info(f"Started: {datetime.now():%Y-%m-%d %H:%M:%S}")
    log.info("="*60)

    sessions = {1: session1, 2: session2, 3: session3, 4: session4, 5: session5}

    if args.all:
        for num in sorted(sessions):
            if num <= args.skip:
                continue
            if session_done(num):
                log.info(f"Session {num} done, skipping. Delete {CKPT_DIR}/s{num}.done to re-run.")
                continue
            sessions[num]()
    elif args.session:
        sessions[args.session]()
    else:
        parser.print_help()

    log.info(f"\nFinished: {datetime.now():%Y-%m-%d %H:%M:%S}")


if __name__ == '__main__':
    main()
