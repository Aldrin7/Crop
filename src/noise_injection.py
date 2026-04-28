"""
Literature-grounded sensor degradation simulation.
v3.2: Monotonic directional drift (realistic electrochemical sensor behavior).
      Dead code (add_class_imbalance) removed — was unused and misleading.

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
    """Apply sensor-specific monotonic drift + noise to each feature column.
    
    v3.2: Drift is directional (sensors lose sensitivity over time, not random).
    Each sensor gets a fixed drift direction per seed, consistent across samples.
    Dropout rate scaled to deployment duration.
    """
    rng = np.random.RandomState(seed)
    days = DEGRADATION[scenario]
    out = df.copy()
    for feat in FEATURES:
        s = SENSOR[feat]
        drift = s['drift_pct_day'] * days
        noise_sigma = s['noise']
        lo, hi = s['range']
        values = out[feat].values.astype(float)
        # Monotonic directional drift: sensor consistently loses sensitivity
        # Direction fixed per feature (some drift high, some drift low)
        drift_direction = rng.choice([-1, 1])
        drift_magnitude = drift * (0.6 + 0.4 * rng.random())  # 60-100% of max drift
        drift_factor = 1.0 + drift_direction * drift_magnitude
        # Additive Gaussian noise (sensor read noise)
        noise = rng.normal(0, noise_sigma, size=len(values))
        degraded = values * drift_factor + noise
        # Dropout: realistic rates scaled to deployment duration
        # Mild: ~2%, Moderate: ~5%, Severe: ~10%
        dropout_rate = 0.02 + 0.08 * (days / 90)
        dropout_mask = rng.random(len(degraded)) < dropout_rate
        degraded[dropout_mask] = np.nan
        # Clip to plausible range
        degraded = np.clip(degraded, lo, hi)
        out[feat] = degraded
    return out
