#!/usr/bin/env python3
"""
Download the Crop Recommendation Dataset from multiple fallback sources.
Run this before pipeline.py if automatic download fails.
"""

import os
import sys
import zipfile
import urllib.request
from pathlib import Path

RAW_DIR = Path(__file__).parent / 'data' / 'raw'
RAW_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT = RAW_DIR / 'Crop_recommendation.csv'

if OUTPUT.exists():
    print(f"Dataset already exists: {OUTPUT} ({OUTPUT.stat().st_size / 1024:.1f} KB)")
    sys.exit(0)

URLS = [
    # Direct CSV mirrors
    "https://raw.githubusercontent.com/dsrscientist/dataset1/master/crop_recommendation.csv",
    # GitHub release mirrors
    "https://raw.githubusercontent.com/Gladiator07/Crop-Recommendation-Dataset/main/Crop_recommendation.csv",
    "https://raw.githubusercontent.com/akagami999/Crop-Recommendation/main/Crop_recommendation.csv",
]

print("Downloading Crop Recommendation Dataset...")
for url in URLS:
    try:
        print(f"  Trying: {url[:80]}...")
        urllib.request.urlretrieve(url, str(OUTPUT))
        if OUTPUT.stat().st_size > 1000:  # Sanity check
            print(f"  ✓ Success! ({OUTPUT.stat().st_size / 1024:.1f} KB)")
            sys.exit(0)
        else:
            OUTPUT.unlink()
            print("  ✗ File too small, trying next...")
    except Exception as e:
        print(f"  ✗ Failed: {e}")

# Kaggle API fallback
try:
    print("\n  Trying Kaggle API...")
    os.system(f"cd {RAW_DIR} && kaggle datasets download -d atharvaingle/crop-recommendation-dataset --unzip -q 2>/dev/null")
    if OUTPUT.exists():
        print(f"  ✓ Kaggle download success!")
        sys.exit(0)
except Exception:
    pass

print("\n⚠ All download methods failed.")
print("Please download manually from:")
print("  https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset")
print(f"  And place Crop_recommendation.csv in: {RAW_DIR}")
sys.exit(1)
