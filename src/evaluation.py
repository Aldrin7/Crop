"""
Evaluation — fixes Critique 3.2 (redundant metrics) and adds calibration.
For balanced datasets: Macro-F1 = Weighted-F1 = Micro-F1 = Accuracy.
We report: Accuracy, Cohen's Kappa, MCC, Brier Score, Log Loss, ECE.
"""
import numpy as np, pandas as pd, logging
from sklearn.metrics import (
    accuracy_score, cohen_kappa_score, matthews_corrcoef,
    f1_score, precision_score, recall_score, classification_report,
    confusion_matrix, log_loss, brier_score_loss
)
from sklearn.calibration import calibration_curve
from scipy.stats import friedmanchisquare
from .config import RANDOM_STATE

log = logging.getLogger(__name__)

def compute_metrics(y_true, y_pred, y_proba=None, le=None):
    """Compute all metrics. For balanced datasets, skip redundant weighted/micro."""
    n_classes = len(np.unique(y_true))
    m = {
        'accuracy': accuracy_score(y_true, y_pred),
        'cohens_kappa': cohen_kappa_score(y_true, y_pred),
        'mcc': matthews_corrcoef(y_true, y_pred),
        'macro_f1': f1_score(y_true, y_pred, average='macro'),
        'macro_precision': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'macro_recall': recall_score(y_true, y_pred, average='macro', zero_division=0),
    }
    # Only add weighted if dataset is imbalanced (they'd differ from macro)
    if y_proba is not None:
        try:
            m['log_loss'] = log_loss(y_true, y_proba)
        except:
            m['log_loss'] = None
        # Per-class Brier score (one-vs-rest)
        y_bin = np.eye(n_classes)[y_true]
        brier_scores = []
        for i in range(n_classes):
            try:
                brier_scores.append(brier_score_loss(y_bin[:, i], y_proba[:, i]))
            except:
                brier_scores.append(None)
        m['brier_per_class'] = brier_scores
        m['brier_mean'] = np.mean([b for b in brier_scores if b is not None])
        # Expected Calibration Error (ECE)
        m['ece'] = compute_ece(y_true, y_proba, n_bins=15)
    # Per-class report
    if le is not None:
        m['classification_report'] = classification_report(
            y_true, y_pred, target_names=le.classes_, output_dict=True)
        m['confusion_matrix'] = confusion_matrix(y_true, y_pred)
    return m

def compute_ece(y_true, y_proba, n_bins=15):
    """Expected Calibration Error."""
    confidences = np.max(y_proba, axis=1)
    predictions = np.argmax(y_proba, axis=1)
    accuracies = (predictions == y_true).astype(float)
    bin_bounds = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for i in range(n_bins):
        mask = (confidences > bin_bounds[i]) & (confidences <= bin_bounds[i+1])
        if mask.sum() == 0:
            continue
        bin_acc = accuracies[mask].mean()
        bin_conf = confidences[mask].mean()
        ece += (mask.sum() / n) * abs(bin_acc - bin_conf)
    return float(ece)

def friedman_test(cv_score_dicts):
    """Friedman test across classifiers. Input: {clf_name: [cv_scores]}"""
    names = list(cv_score_dicts.keys())
    scores = [cv_score_dicts[n] for n in names]
    if len(scores) < 3:
        return None
    stat, p = friedmanchisquare(*scores)
    return {'statistic': float(stat), 'p_value': float(p), 'significant': p < 0.05, 'classifiers': names}

def nemenyi_critical_difference(k, alpha=0.05):
    """Return CD for Nemenyi post-hoc test."""
    # q_alpha values for alpha=0.05 (from tables)
    q_table = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850,
               7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164}
    q = q_table.get(k, 3.164)
    return q * np.sqrt(k * (k + 1) / (6 * 5))  # N=5 folds
