"""Basic tests for the RobustCrop pipeline modules.

Run with: python -m pytest tests/ -v
"""
import sys
import numpy as np
import pandas as pd
import pytest
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import FEATURES, TARGET, RANDOM_STATE, SECONDARY_FEATURES


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def synthetic_data():
    """Create a small synthetic dataset for testing."""
    rng = np.random.RandomState(RANDOM_STATE)
    n = 200
    X = pd.DataFrame(rng.randn(n, len(FEATURES)), columns=FEATURES)
    y = rng.choice(['rice', 'wheat', 'maize'], size=n)
    return X, y


@pytest.fixture
def synthetic_df(synthetic_data):
    """Create a DataFrame with features + target."""
    X, y = synthetic_data
    df = X.copy()
    df[TARGET] = y
    return df


# ── Config Tests ─────────────────────────────────────────────────────────────

def test_config_features_exist():
    assert len(FEATURES) == 7
    assert 'N' in FEATURES
    assert 'rainfall' in FEATURES


def test_config_secondary_features():
    assert len(SECONDARY_FEATURES) == 12
    assert 'N' in SECONDARY_FEATURES


# ── Preprocessing Tests ─────────────────────────────────────────────────────

def test_handle_missing_no_nans(synthetic_df):
    from src.preprocessing import handle_missing
    result = handle_missing(synthetic_df.copy())
    assert result[FEATURES].isnull().sum().sum() == 0


def test_handle_missing_with_nans(synthetic_df):
    from src.preprocessing import handle_missing
    df = synthetic_df.copy()
    df.loc[0, 'N'] = np.nan
    df.loc[1, 'P'] = np.nan
    result = handle_missing(df)
    assert result[FEATURES].isnull().sum().sum() == 0


def test_detect_outliers(synthetic_df):
    from src.preprocessing import detect_outliers
    report = detect_outliers(synthetic_df)
    assert isinstance(report, dict)


def test_encode_target(synthetic_data):
    from src.preprocessing import encode_target
    _, y = synthetic_data
    y_enc, le = encode_target(y)
    assert y_enc.min() >= 0
    assert len(le.classes_) == 3


def test_scale_features_deprecation_warning(synthetic_data):
    from src.preprocessing import scale_features
    X, _ = synthetic_data
    X_train, X_test = X.iloc[:150], X.iloc[150:]
    with pytest.warns(DeprecationWarning, match="leak-free"):
        scale_features(X_train, X_test)


# ── Model Tests ──────────────────────────────────────────────────────────────

def test_all_classifiers_returns_dict():
    from src.models import all_classifiers
    clfs = all_classifiers()
    assert isinstance(clfs, dict)
    assert len(clfs) >= 8  # at least 8 core classifiers


def test_bal_weight_wrapper_fit_predict(synthetic_data):
    from src.models import BalWeightWrapper
    from sklearn.naive_bayes import GaussianNB
    X, y = synthetic_data
    from sklearn.preprocessing import LabelEncoder
    y_enc = LabelEncoder().fit_transform(y)
    clf = BalWeightWrapper(GaussianNB())
    clf.fit(X.iloc[:150].values, y_enc[:150])
    preds = clf.predict(X.iloc[150:].values)
    assert len(preds) == 50
    assert hasattr(clf, 'classes_')


def test_bal_weight_wrapper_has_predict_proba(synthetic_data):
    from src.models import BalWeightWrapper
    from sklearn.neighbors import KNeighborsClassifier
    X, y = synthetic_data
    from sklearn.preprocessing import LabelEncoder
    y_enc = LabelEncoder().fit_transform(y)
    clf = BalWeightWrapper(KNeighborsClassifier(n_neighbors=3))
    clf.fit(X.iloc[:150].values, y_enc[:150])
    proba = clf.predict_proba(X.iloc[150:].values)
    assert proba.shape == (50, 3)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)


# ── Feature Selection Tests ─────────────────────────────────────────────────

def test_topk_from_scores_fit_transform(synthetic_data):
    from src.feature_selection import TopKFromScores
    X, y = synthetic_data
    from sklearn.preprocessing import LabelEncoder
    y_enc = LabelEncoder().fit_transform(y)
    selector = TopKFromScores(score_func='mutual_info', k=3)
    selector.fit(X.values, y_enc)
    X_sel = selector.transform(X.values)
    assert X_sel.shape == (200, 3)
    assert len(selector.selected_) == 3


def test_topk_chi2(synthetic_data):
    from src.feature_selection import TopKFromScores
    X, y = synthetic_data
    from sklearn.preprocessing import LabelEncoder
    y_enc = LabelEncoder().fit_transform(y)
    X_pos = np.clip(X.values, 0, None)
    selector = TopKFromScores(score_func='chi2', k=4)
    selector.fit(X_pos, y_enc)
    X_sel = selector.transform(X_pos)
    assert X_sel.shape == (200, 4)


def test_rfe_selector(synthetic_data):
    from src.feature_selection import RFESelector
    X, y = synthetic_data
    from sklearn.preprocessing import LabelEncoder
    y_enc = LabelEncoder().fit_transform(y)
    selector = RFESelector(k=3)
    selector.fit(X.values, y_enc)
    X_sel = selector.transform(X.values)
    assert X_sel.shape == (200, 3)


def test_run_all_fs_methods_warning(synthetic_data):
    from src.feature_selection import run_all_fs_methods
    X, y = synthetic_data
    from sklearn.preprocessing import LabelEncoder
    y_enc = LabelEncoder().fit_transform(y)
    with pytest.warns(UserWarning, match="descriptive ONLY"):
        results = run_all_fs_methods(X.values, y_enc, feature_names=FEATURES)
    assert 'consensus' in results
    assert len(results) >= 7  # 6 methods + consensus


# ── Evaluation Tests ─────────────────────────────────────────────────────────

def test_compute_metrics(synthetic_data):
    from src.evaluation import compute_metrics
    from sklearn.preprocessing import LabelEncoder
    X, y = synthetic_data
    y_enc = LabelEncoder().fit_transform(y)
    y_pred = y_enc.copy()
    rng = np.random.RandomState(42)
    flip_idx = rng.choice(len(y_pred), size=20, replace=False)
    y_pred[flip_idx] = rng.choice(3, size=20)
    n_classes = len(np.unique(y_enc))
    y_proba = np.eye(n_classes)[y_pred]  # one-hot
    m = compute_metrics(y_enc, y_pred, y_proba)
    assert 'accuracy' in m
    assert 'cohens_kappa' in m
    assert 'mcc' in m
    assert 'macro_f1' in m
    assert 0 <= m['accuracy'] <= 1


def test_compute_metrics_without_proba():
    from src.evaluation import compute_metrics
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 2, 0, 1, 1])
    m = compute_metrics(y_true, y_pred)
    assert 'accuracy' in m
    assert 'brier_mean' not in m  # no proba → no brier


def test_friedman_test():
    from src.evaluation import friedman_test
    scores = {
        'RF': [0.9, 0.91, 0.92, 0.9, 0.91],
        'SVM': [0.85, 0.86, 0.87, 0.85, 0.86],
        'KNN': [0.80, 0.81, 0.82, 0.80, 0.81],
    }
    result = friedman_test(scores)
    assert result is not None
    assert 'statistic' in result
    assert 'p_value' in result


def test_friedman_too_few():
    from src.evaluation import friedman_test
    scores = {'A': [1, 2], 'B': [3, 4]}
    result = friedman_test(scores)
    assert result is None


def test_nemenyi_cd():
    from src.evaluation import nemenyi_critical_difference
    cd = nemenyi_critical_difference(5)
    assert cd > 0
    cd10 = nemenyi_critical_difference(10)
    assert cd10 > cd  # more classifiers → larger CD


# ── Noise Injection Tests ────────────────────────────────────────────────────

def test_degrade_dataset_shapes(synthetic_df):
    from src.noise_injection import degrade_dataset
    for scenario in ['mild', 'moderate', 'severe']:
        degraded = degrade_dataset(synthetic_df, scenario=scenario)
        assert degraded.shape == synthetic_df.shape
        assert set(degraded.columns) == set(synthetic_df.columns)


def test_degrade_dataset_introduces_nans(synthetic_df):
    from src.noise_injection import degrade_dataset
    degraded = degrade_dataset(synthetic_df, scenario='severe')
    nans = degraded[FEATURES].isnull().sum().sum()
    assert nans > 0  # severe degradation should create dropout


def test_degrade_dataset_clipping(synthetic_df):
    from src.noise_injection import degrade_dataset
    from src.config import SENSOR
    degraded = degrade_dataset(synthetic_df, scenario='severe')
    for feat in FEATURES:
        lo, hi = SENSOR[feat]['range']
        vals = degraded[feat].dropna()
        assert vals.min() >= lo
        assert vals.max() <= hi


# ── Explainability Tests ────────────────────────────────────────────────────

def test_correlation_violation_report(synthetic_data):
    from src.explainability import correlation_violation_report
    X, _ = synthetic_data
    violations = correlation_violation_report(X.values, FEATURES)
    assert isinstance(violations, list)
    for v in violations:
        assert 'feature_1' in v
        assert 'abs_correlation' in v


# ── Integration: Pipeline smoke test ────────────────────────────────────────

def test_pipeline_fit_predict_smoke(synthetic_data):
    """Smoke test: build a Pipeline, fit, predict — no crashes."""
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.feature_selection import SelectKBest, mutual_info_classif
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import LabelEncoder

    X, y = synthetic_data
    y_enc = LabelEncoder().fit_transform(y)

    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('selector', SelectKBest(mutual_info_classif, k=5)),
        ('clf', RandomForestClassifier(n_estimators=10, random_state=42)),
    ])

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    for train_idx, test_idx in cv.split(X.values, y_enc):
        pipe.fit(X.values[train_idx], y_enc[train_idx])
        preds = pipe.predict(X.values[test_idx])
        assert len(preds) == len(test_idx)
        break  # one fold is enough for smoke test
