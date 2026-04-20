"""
Literature-grounded sensor degradation simulation.
Sources:
  - Rana et al. (2019) "IoT-based smart agriculture" — NPK electrochemical drift
  - Lobnik et al. (2011) pH electrode long-term stability
  - Sensirion SHT4x datasheet — humidity/temp sensor specs
  - Martínez et al. (2007) tipping-bucket rain gauge accuracy
"""
import numpy as np
import pandas as pd
from .config import FEATURES, TARGET, SENSOR, DEGRADATION

def degrade_dataset(df, scenario='moderate', seed=42):
    """Apply sensor-specific drift + noise to each feature column."""
    rng = np.random.RandomState(seed)
    days = DEGRADATION[scenario]
    out = df.copy()
    for feat in FEATURES:
        s = SENSOR[feat]
        drift = s['drift_pct_day'] * days
        noise_sigma = s['noise']
        lo, hi = s['range']
        # multiplicative drift + additive Gaussian noise
        values = out[feat].values.astype(float)
        drift_factor = 1.0 + rng.uniform(-drift, drift, size=len(values))
        noise = rng.normal(0, noise_sigma, size=len(values))
        degraded = values * drift_factor + noise
        # inject 1-3% missing at random (simulates sensor dropout)
        dropout_mask = rng.random(len(degraded)) < (0.01 + 0.005 * (days / 30))
        degraded[dropout_mask] = np.nan
        # clip to plausible range
        degraded = np.clip(degraded, lo, hi)
        out[feat] = degraded
    return out

def add_class_imbalance(df, imbalance_ratio=0.3, seed=42):
    """Subsample minority classes to create realistic imbalance."""
    rng = np.random.RandomState(seed)
    classes = df[TARGET].unique()
    n_minority = int(100 * imbalance_ratio)
    frames = []
    for cls in classes:
        sub = df[df[TARGET] == cls]
        if rng.random() < 0.5:
            frames.append(sub.sample(n=n_minority, random_state=seed))
        else:
            frames.append(sub)
    return pd.concat(frames, ignore_index=True).sample(frac=1, random_state=seed)
