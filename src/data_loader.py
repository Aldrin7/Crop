"""Data acquisition — loads primary + secondary real-world datasets."""
import os, sys, urllib.request, logging
import pandas as pd
import numpy as np
from pathlib import Path
from .config import (
    RAW_DIR, FEATURES, TARGET, RANDOM_STATE,
    SECONDARY_FILE, SECONDARY_FEATURES, SECONDARY_TARGET, SHARED_FEATURES,
)

log = logging.getLogger(__name__)

PRIMARY_URLS = [
    "https://raw.githubusercontent.com/ankitaS11/Crop-Yield-Prediction-in-India-using-ML/main/Crop_recommendation.csv",
]

# ─────────────────────────────────────────────────────────────────────────────
# PRIMARY — Crop Recommendation (semi-synthetic, 2200 samples, 22 classes)
# ─────────────────────────────────────────────────────────────────────────────

def load_primary():
    """Load primary Crop Recommendation dataset (2200 samples, 7 features, 22 classes)."""
    csv = RAW_DIR / "Crop_recommendation.csv"
    if not csv.exists():
        raise FileNotFoundError(f"Primary dataset not found: {csv}")
    df = pd.read_csv(csv)
    log.info(f"Primary: {df.shape[0]} rows, {df[TARGET].nunique()} classes")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# SECONDARY — Soil Fertility (REAL lab measurements, 880 samples, 3 classes)
# ─────────────────────────────────────────────────────────────────────────────

def load_secondary():
    """
    Load secondary real-world soil fertility dataset.
    
    Source: Rahul Jaiswal (Kaggle) — actual soil laboratory test results
    from Indian agricultural testing centres.
    
    880 samples, 12 physicochemical features, 3 fertility classes:
      0 = High fertility  (401 samples)
      1 = Medium fertility (440 samples)  
      2 = Low fertility    (39 samples)
    
    Natural class imbalance reflects real-world soil conditions.
    Missing values present (~3%) — real sensor/lab dropout.
    """
    csv = RAW_DIR / SECONDARY_FILE
    if not csv.exists():
        raise FileNotFoundError(f"Secondary dataset not found: {csv}")
    
    df = pd.read_csv(csv)
    
    # Standardise column names
    df.columns = [c.strip() for c in df.columns]
    if SECONDARY_TARGET not in df.columns and 'Output' in df.columns:
        df = df.rename(columns={'Output': SECONDARY_TARGET})
    
    log.info(f"Secondary: {df.shape[0]} rows, {df[SECONDARY_TARGET].nunique()} classes")
    log.info(f"  Features: {SECONDARY_FEATURES}")
    log.info(f"  Class distribution: {dict(df[SECONDARY_TARGET].value_counts())}")
    log.info(f"  Missing values: {df[SECONDARY_FEATURES].isnull().sum().sum()}")
    
    return df


def get_shared_features(primary_df, secondary_df):
    """
    Extract the overlapping feature space between datasets.
    Returns aligned DataFrames with shared features only.
    Both have N, P, K; primary has 'ph' ≈ secondary 'pH'.
    """
    sec_rename = {}
    if 'pH' in secondary_df.columns:
        sec_rename['pH'] = 'ph'
    
    secondary_renamed = secondary_df.rename(columns=sec_rename)
    
    # Use N, P, K, pH (ph) as the shared feature space
    shared = ['N', 'P', 'K', 'ph']
    
    primary_shared = primary_df[shared + [TARGET]].copy()
    secondary_shared = secondary_renamed[shared + [SECONDARY_TARGET]].copy()
    
    log.info(f"Shared feature space: {shared}")
    log.info(f"  Primary: {primary_shared.shape[0]} samples")
    log.info(f"  Secondary: {secondary_shared.shape[0]} samples")
    
    return primary_shared, secondary_shared, shared


# ─────────────────────────────────────────────────────────────────────────────
# DEGRADATION VARIANTS (sensor drift simulation on primary)
# ─────────────────────────────────────────────────────────────────────────────

def load_secondary_variants(n_variants=3):
    """
    Generate sensor degradation variants of the primary dataset.
    Simulates literature-grounded sensor drift for robustness testing.
    """
    from .noise_injection import degrade_dataset
    df = load_primary()
    scenarios = ['mild', 'moderate', 'severe'][:n_variants]
    variants = {}
    for sc in scenarios:
        degraded = degrade_dataset(df, scenario=sc, seed=RANDOM_STATE)
        variants[sc] = degraded
        log.info(f"Degradation variant '{sc}': {degraded.shape}")
    return variants


# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────────────────────

def get_dataset_summary(df, target_col=TARGET):
    """Return a dict of dataset statistics."""
    feats = [c for c in df.columns if c != target_col]
    vc = df[target_col].value_counts()
    return {
        'n_samples': len(df),
        'n_features': len(feats),
        'n_classes': df[target_col].nunique(),
        'features': feats,
        'classes': sorted(df[target_col].unique().tolist()),
        'class_counts': vc.to_dict(),
        'missing': int(df[feats].isnull().sum().sum()),
        'duplicates': int(df.duplicated().sum()),
        'is_balanced': vc.nunique() == 1,
        'min_class': int(vc.min()),
        'max_class': int(vc.max()),
        'imbalance_ratio': round(vc.max() / vc.min(), 2) if vc.min() > 0 else float('inf'),
    }
