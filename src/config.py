"""Central configuration — all paths, hyperparams, literature-grounded sensor specs."""
from pathlib import Path
import numpy as np

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / 'data'
RAW_DIR = DATA_DIR / 'raw'; PROC_DIR = DATA_DIR / 'processed'
CKPT_DIR = DATA_DIR / 'checkpoints'; MODEL_DIR = BASE_DIR / 'models'
RESULTS_DIR = BASE_DIR / 'results'
FIG_DIR = RESULTS_DIR / 'figures'; TABLE_DIR = RESULTS_DIR / 'tables'
METRIC_DIR = RESULTS_DIR / 'metrics'; LOG_DIR = BASE_DIR / 'logs'
for _d in [RAW_DIR, PROC_DIR, CKPT_DIR, MODEL_DIR, FIG_DIR, TABLE_DIR, METRIC_DIR, LOG_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# ── Core params ────────────────────────────────────────────────────────────────
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_OUTER = 5          # outer fold — unbiased generalisation estimate
CV_INNER = 3          # inner fold — FS + hyper-parameter tuning
FEATURES = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
TARGET = 'label'

# ── Sensor degradation — literature-grounded ───────────────────────────────────
# Sources:
#   Rana et al. (2019) "IoT sensor reliability in agriculture" — NPK electrochem.
#   Lobnik et al. (2011) pH glass electrode long-term drift
#   Sensirion SHT4x datasheet — humidity/temp accuracy spec
SENSOR = {
    'N':           {'drift_pct_day': 0.010, 'noise': 2.0, 'range': (0, 150)},
    'P':           {'drift_pct_day': 0.015, 'noise': 1.5, 'range': (0, 150)},
    'K':           {'drift_pct_day': 0.012, 'noise': 1.5, 'range': (0, 210)},
    'temperature': {'drift_pct_day': 0.002, 'noise': 0.5, 'range': (-5, 50)},
    'humidity':    {'drift_pct_day': 0.005, 'noise': 1.0, 'range': (10, 100)},
    'ph':          {'drift_pct_day': 0.001, 'noise': 0.1, 'range': (3, 10)},
    'rainfall':    {'drift_pct_day': 0.003, 'noise': 5.0, 'range': (0, 350)},
}

DEGRADATION = {
    'fresh':    0,   # freshly calibrated
    'mild':     7,   # 7-day deployment
    'moderate': 30,  # 30-day deployment  
    'severe':   90,  # 90-day deployment
}
