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

def handle_missing(df):
    """Median imputation for numerical features (sensor dropout)."""
    missing = df[FEATURES].isnull().sum()
    if missing.sum() > 0:
        log.info(f"Missing values found: {dict(missing[missing > 0])}")
        imp = SimpleImputer(strategy='median')
        df[FEATURES] = imp.fit_transform(df[FEATURES])
    return df

def detect_outliers(df):
    """IQR-based outlier report (no removal — preserve real-world variance)."""
    report = {}
    for f in FEATURES:
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

def scale_features(X_train, X_test, scaler_name='StandardScaler'):
    """Fit scaler on train, transform both train and test."""
    scaler = SCALERS[scaler_name]()
    X_train_s = pd.DataFrame(scaler.fit_transform(X_train), columns=FEATURES, index=X_train.index)
    X_test_s = pd.DataFrame(scaler.transform(X_test), columns=FEATURES, index=X_test.index)
    return X_train_s, X_test_s, scaler

def prepare_data(df, scaler_name='StandardScaler', test_size=TEST_SIZE):
    """Full preprocessing: impute → encode → scale → split."""
    from sklearn.model_selection import train_test_split
    df = handle_missing(df.copy())
    outliers = detect_outliers(df)
    if outliers:
        log.info(f"Outliers (kept): {outliers}")
    X = df[FEATURES].copy()
    y_enc, le = encode_target(df[TARGET])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=test_size, random_state=RANDOM_STATE, stratify=y_enc
    )
    X_train_s, X_test_s, scaler = scale_features(X_train, X_test, scaler_name)
    return {
        'X_train': X_train_s, 'X_test': X_test_s,
        'y_train': y_train, 'y_test': y_test,
        'label_encoder': le, 'scaler': scaler,
        'outlier_report': outliers,
    }
