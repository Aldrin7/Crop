"""Classifier definitions — returns a dict of named estimators.
v3.2: class_weight='balanced' applied where natively supported.
      BalWeightWrapper adds sample_weight-based balancing for classifiers
      that don't support class_weight natively (XGBoost, GB, MLP, KNN, GaussianNB).
      This ensures ALL classifiers receive fair imbalance handling on the
      secondary (imbalanced) dataset.
"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.class_weight import compute_sample_weight
from .config import RANDOM_STATE


class BalWeightWrapper(BaseEstimator, ClassifierMixin):
    """Wraps any classifier to pass sample_weight during fit().
    
    Computes sample_weight='balanced' from y at fit time, then delegates
    to the inner estimator's fit(sample_weight=...) if supported, or
    fit() otherwise.  This gives XGBoost, GB, MLP, KNN, GaussianNB the
    same imbalance correction that class_weight='balanced' gives RF/SVM/DT/LR.
    """

    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y, **fit_params):
        sw = compute_sample_weight('balanced', y)
        import inspect
        sig = inspect.signature(self.estimator.fit)
        if 'sample_weight' in sig.parameters:
            self.estimator.fit(X, y, sample_weight=sw, **fit_params)
        else:
            self.estimator.fit(X, y, **fit_params)
        # Expose common attributes expected by sklearn utilities
        self.classes_ = getattr(self.estimator, 'classes_', np.unique(y))
        return self

    def predict(self, X):
        return self.estimator.predict(X)

    def predict_proba(self, X):
        return self.estimator.predict_proba(X)

    def decision_function(self, X):
        return self.estimator.decision_function(X)

    @property
    def feature_importances_(self):
        return getattr(self.estimator, 'feature_importances_', None)

    def __getattr__(self, name):
        # Delegate attribute access to the inner estimator for introspection
        # (e.g. feature_importances_, estimators_, coef_)
        try:
            return super().__getattribute__(name)
        except AttributeError:
            return getattr(self.estimator, name)


def get_classifiers():
    """Core classifiers. class_weight='balanced' where natively supported."""
    return {
        'RandomForest': RandomForestClassifier(
            n_estimators=200, max_depth=20, min_samples_split=5,
            class_weight='balanced', random_state=RANDOM_STATE, n_jobs=-1),
        'SVM_RBF': SVC(
            kernel='rbf', C=10, gamma='scale',
            class_weight='balanced', random_state=RANDOM_STATE, probability=True),
        'KNN': BalWeightWrapper(KNeighborsClassifier(
            n_neighbors=7, weights='distance', n_jobs=-1)),
        'DecisionTree': DecisionTreeClassifier(
            max_depth=15, min_samples_split=5,
            class_weight='balanced', random_state=RANDOM_STATE),
        'GradientBoosting': BalWeightWrapper(GradientBoostingClassifier(
            n_estimators=150, max_depth=5, learning_rate=0.1,
            random_state=RANDOM_STATE)),
        'LogisticRegression': LogisticRegression(
            max_iter=5000, solver='lbfgs',
            C=1.0, class_weight='balanced', random_state=RANDOM_STATE),
        'MLP': BalWeightWrapper(MLPClassifier(
            hidden_layer_sizes=(128, 64, 32), max_iter=500,
            early_stopping=True, validation_fraction=0.15,
            random_state=RANDOM_STATE, batch_size=64)),
        'GaussianNB': BalWeightWrapper(GaussianNB()),
    }


def get_optional_classifiers():
    """Returns XGBoost/LightGBM if installed, else empty dict.
    Both wrapped with BalWeightWrapper for consistent sample_weight balancing."""
    extras = {}
    try:
        import xgboost as xgb
        extras['XGBoost'] = BalWeightWrapper(xgb.XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            random_state=RANDOM_STATE, n_jobs=-1, eval_metric='mlogloss'))
    except ImportError:
        pass
    try:
        import lightgbm as lgb
        extras['LightGBM'] = lgb.LGBMClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            class_weight='balanced',
            random_state=RANDOM_STATE, n_jobs=-1, verbose=-1)
    except ImportError:
        pass
    return extras


def all_classifiers():
    d = get_classifiers()
    d.update(get_optional_classifiers())
    return d
