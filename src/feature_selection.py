"""
Feature Selection — LEAK-FREE.
All pipeline-compatible selectors operate INSIDE the CV loop via sklearn Pipeline.
Critique 2.2 fix: No pre-selection on full X_train before CV.

v3.2: run_all_fs_methods() now carries explicit warnings about data leakage.
      It is ONLY for descriptive/interpretation analysis on the full dataset.
      Training uses per-fold MI selection inside the Pipeline.
"""
import warnings
import numpy as np, pandas as pd, logging
from sklearn.feature_selection import (
    SelectKBest, chi2, mutual_info_classif, RFE, f_classif
)
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, TransformerMixin
from .config import FEATURES, RANDOM_STATE

log = logging.getLogger(__name__)

# ── Pipeline-compatible selectors ──────────────────────────────────────────────

class TopKFromScores(BaseEstimator, TransformerMixin):
    """Select top-k features using a scoring function. Pipeline-compatible."""
    def __init__(self, score_func='mutual_info', k=5):
        self.score_func = score_func
        self.k = k
        self.scores_ = None
        self.selected_ = None
    
    def fit(self, X, y):
        if self.score_func == 'mutual_info':
            scores = mutual_info_classif(X, y, random_state=RANDOM_STATE)
        elif self.score_func == 'chi2':
            X_pos = np.clip(X, 0, None)
            scores = chi2(X_pos, y)[0]
        elif self.score_func == 'f_classif':
            scores = f_classif(X, y)[0]
        elif self.score_func == 'rf_importance':
            rf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
            rf.fit(X, y)
            scores = rf.feature_importances_
        elif self.score_func == 'extratrees':
            et = ExtraTreesClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
            et.fit(X, y)
            scores = et.feature_importances_
        elif self.score_func == 'lasso':
            lr = LogisticRegression(penalty='l1', solver='saga', C=1.0,
                                    max_iter=3000, random_state=RANDOM_STATE)
            lr.fit(X, y)
            scores = np.abs(lr.coef_).mean(axis=0)
        else:
            raise ValueError(f"Unknown score_func: {self.score_func}")
        self.scores_ = scores
        top_k = np.argsort(scores)[::-1][:self.k]
        self.selected_ = sorted(top_k.tolist())
        return self
    
    def transform(self, X):
        return X[:, self.selected_] if hasattr(X, 'shape') and len(X.shape) == 2 else X.iloc[:, self.selected_]
    
    def get_feature_names_out(self, input_features=None):
        if input_features is not None:
            return [input_features[i] for i in self.selected_]
        return [f"f{i}" for i in self.selected_]

class RFESelector(BaseEstimator, TransformerMixin):
    """RFE with RF estimator. Pipeline-compatible."""
    def __init__(self, k=5):
        self.k = k
        self.rfe_ = None
        self.selected_ = None
    
    def fit(self, X, y):
        rf = RandomForestClassifier(n_estimators=50, random_state=RANDOM_STATE, n_jobs=-1)
        self.rfe_ = RFE(rf, n_features_to_select=self.k, step=1)
        self.rfe_.fit(X, y)
        self.selected_ = sorted(np.where(self.rfe_.support_)[0].tolist())
        return self
    
    def transform(self, X):
        return self.rfe_.transform(X)

# ── Standalone FS for analysis (NOT inside CV) ────────────────────────────────

def run_all_fs_methods(X, y, feature_names=None):
    """
    Run all FS methods for INTERPRETATION ONLY.
    
    .. warning::
        This function fits feature selection on the FULL dataset.  The resulting
        consensus ranking is DESCRIPTIVE ONLY — it shows which features are
        important across methods, but using it for feature selection before
        training would leak information.
        
        For training, use per-fold MI selection inside the Pipeline (Session 3).
        This function is used in Session 2 for the ablation study and
        interpretability analysis only.
    
    Returns a dict of DataFrames with feature scores per method.
    """
    warnings.warn(
        "run_all_fs_methods() fits on the full dataset. Results are descriptive "
        "ONLY. Do NOT use for feature selection before training. Use the "
        "per-fold MI selection inside the Pipeline for training.",
        UserWarning,
        stacklevel=2,
    )
    if feature_names is None:
        feature_names = FEATURES
    results = {}
    
    # 1. Mutual Information
    mi = mutual_info_classif(X, y, random_state=RANDOM_STATE)
    results['mutual_info'] = pd.DataFrame({'feature': feature_names, 'score': mi}).sort_values('score', ascending=False)
    
    # 2. Chi-Square (needs non-negative)
    X_pos = np.clip(X, 0, None) if isinstance(X, np.ndarray) else np.clip(X.values, 0, None)
    chi2_scores = chi2(X_pos, y)[0]
    results['chi2'] = pd.DataFrame({'feature': feature_names, 'score': chi2_scores}).sort_values('score', ascending=False)
    
    # 3. RF Importance
    rf = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)
    rf.fit(X, y)
    results['rf_importance'] = pd.DataFrame({'feature': feature_names, 'score': rf.feature_importances_}).sort_values('score', ascending=False)
    
    # 4. Extra Trees
    et = ExtraTreesClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)
    et.fit(X, y)
    results['extratrees'] = pd.DataFrame({'feature': feature_names, 'score': et.feature_importances_}).sort_values('score', ascending=False)
    
    # 5. LASSO
    lr = LogisticRegression(penalty='l1', solver='saga', C=1.0, max_iter=5000,
                            random_state=RANDOM_STATE)
    lr.fit(X, y)
    lasso_scores = np.abs(lr.coef_).mean(axis=0)
    results['lasso'] = pd.DataFrame({'feature': feature_names, 'score': lasso_scores}).sort_values('score', ascending=False)
    
    # 6. RFE
    rf_rfe = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
    rfe = RFE(rf_rfe, n_features_to_select=1, step=1)
    rfe.fit(X, y)
    results['rfe'] = pd.DataFrame({'feature': feature_names, 'ranking': rfe.ranking_}).sort_values('ranking')
    
    # Consensus: normalize each to [0,1], average
    consensus = pd.DataFrame({'feature': feature_names})
    for method, df in results.items():
        col = 'score' if 'score' in df.columns else 'ranking'
        vals = df.set_index('feature')[col]
        if col == 'ranking':
            vals = (vals.max() - vals + 1) / vals.max()
        else:
            rng = vals.max() - vals.min()
            vals = (vals - vals.min()) / (rng + 1e-10)
        consensus[method] = consensus['feature'].map(vals)
    consensus['mean_score'] = consensus[[c for c in consensus.columns if c != 'feature']].mean(axis=1)
    consensus = consensus.sort_values('mean_score', ascending=False)
    results['consensus'] = consensus
    return results
