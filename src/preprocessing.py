"""Preprocessing — scaling, encoding, missing-value handling, outlier detection."""
import numpy as np, pandas as pd, logging
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from .config import FEATURES, TARGET, RANDOM_STATE, TEST_SIZE

log = logging.getLogger(__name__)

SCALERS = {
    'StandardScaler': StandardScaler,
    'MinMaxScaler': MinMaxScaler,
    'RobustScaler': RobustScaler,
}

def handle_missing(df, feature_cols=None):
    """Median imputation for numerical features (sensor dropout)."""
    if feature_cols is None:
        feature_cols = FEATURES
    feature_cols = [c for c in feature_cols if c in df.columns]
    missing = df[feature_cols].isnull().sum()
    if missing.sum() > 0:
        log.info(f"Missing values found: {dict(missing[missing > 0])}")
        imp = SimpleImputer(strategy='median')
        df[feature_cols] = imp.fit_transform(df[feature_cols])
    return df

def detect_outliers(df, feature_cols=None):
    """IQR-based outlier report (no removal — preserve real-world variance)."""
    if feature_cols is None:
        feature_cols = FEATURES
    feature_cols = [c for c in feature_cols if c in df.columns]
    report = {}
    for f in feature_cols:
        Q1, Q3 = df[f].quantile(0.25), df[f].quantile(0.75)
        IQR = Q3 - Q1
        n = ((df[f] < Q1 - 1.5*IQR) | (df[f] > Q3 + 1.5*IQR)).sum()
        if n > 0:
            report[f] = {'count': int(n), 'pct': round(n/len(df)*100, 2)}
    return report

def encode_target(y):
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    return y_enc, le

def scale_features(X_train, X_test, scaler_name='StandardScaler', feature_cols=None):
    """Fit scaler on train, transform both train and test."""
    if feature_cols is None:
        feature_cols = FEATURES
    scaler = SCALERS[scaler_name]()
    X_train_s = pd.DataFrame(scaler.fit_transform(X_train), columns=feature_cols, index=X_train.index)
    X_test_s = pd.DataFrame(scaler.transform(X_test), columns=feature_cols, index=X_test.index)
    return X_train_s, X_test_s, scaler

def prepare_data(df, scaler_name='StandardScaler', test_size=TEST_SIZE,
                 target_col=None, feature_cols=None):
    """Full preprocessing: impute → encode → scale → split."""
    from sklearn.model_selection import train_test_split
    if target_col is None:
        target_col = TARGET
    if feature_cols is None:
        feature_cols = FEATURES
    df = handle_missing(df.copy(), feature_cols=feature_cols)
    outliers = detect_outliers(df, feature_cols=feature_cols)
    if outliers:
        log.info(f"Outliers (kept): {outliers}")
    X = df[feature_cols].copy()
    y_enc, le = encode_target(df[target_col])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=test_size, random_state=RANDOM_STATE, stratify=y_enc
    )
    X_train_s, X_test_s, scaler = scale_features(X_train, X_test, scaler_name,
                                                   feature_cols=feature_cols)
    return {
        'X_train': X_train_s, 'X_test': X_test_s,
        'X_train_raw': X_train, 'X_test_raw': X_test,  # unscaled — for leak-free pipeline
        'y_train': y_train, 'y_test': y_test,
        'label_encoder': le, 'scaler': scaler,
        'outlier_report': outliers,
        'feature_cols': feature_cols,
        'target_col': target_col,
    }
