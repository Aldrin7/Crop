"""Utilities — logging, checkpointing, figure/table saving."""
import os, json, pickle, logging
from pathlib import Path
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from .config import CKPT_DIR, FIG_DIR, TABLE_DIR, METRIC_DIR, LOG_DIR

def setup_logging():
    log_file = LOG_DIR / f"run_{datetime.now():%Y%m%d_%H%M%S}.log"
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] [%(levelname)s] %(message)s',
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
        datefmt='%H:%M:%S',
    )
    return logging.getLogger(__name__)

def save_ckpt(name, data):
    p = CKPT_DIR / f"{name}.pkl"
    with open(p, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    logging.info(f"Checkpoint: {name} ({p.stat().st_size/1024:.1f} KB)")

def load_ckpt(name):
    p = CKPT_DIR / f"{name}.pkl"
    if p.exists():
        with open(p, 'rb') as f:
            logging.info(f"Loaded: {name}")
            return pickle.load(f)
    return None

def ckpt_exists(name):
    return (CKPT_DIR / f"{name}.pkl").exists()

def session_done(n):
    return (CKPT_DIR / f"s{n}.done").exists()

def mark_done(n):
    (CKPT_DIR / f"s{n}.done").write_text(datetime.now().isoformat())

def save_fig(fig, name, dpi=300):
    fig.savefig(FIG_DIR / f"{name}.png", dpi=dpi, bbox_inches='tight', facecolor='white')
    fig.savefig(FIG_DIR / f"{name}.pdf", bbox_inches='tight', facecolor='white')
    plt.close(fig)
    logging.info(f"Figure: {name}.png/pdf")

def save_table(df, name):
    df.to_csv(TABLE_DIR / f"{name}.csv", index=True)
    try:
        df.to_latex(TABLE_DIR / f"{name}.tex", index=True, float_format="%.4f")
    except:
        pass
    logging.info(f"Table: {name}.csv")

def save_json(data, name):
    p = METRIC_DIR / f"{name}.json"
    with open(p, 'w') as f:
        json.dump(data, f, indent=2, default=str)
    logging.info(f"JSON: {name}.json")
