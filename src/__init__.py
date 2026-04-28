"""RobustCrop — Leak-free ML pipeline for crop recommendation."""
from .config import *
from .data_loader import load_primary, load_secondary, load_secondary_variants
from .preprocessing import prepare_data, handle_missing, encode_target
from .feature_selection import run_all_fs_methods, TopKFromScores, RFESelector
from .models import all_classifiers, get_classifiers, get_optional_classifiers
from .evaluation import compute_metrics, friedman_test, nemenyi_critical_difference
from .explainability import compute_shap_values, analyze_gaussian_nb_calibration
from .noise_injection import degrade_dataset
from .utils import setup_logging, save_fig, save_table, save_json
