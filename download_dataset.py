"""
Download the UCI South German Credit dataset.
Run this script on your local machine (not in a sandboxed environment).

    python download_dataset.py

It will download and extract the dataset into data/raw/.
"""

import urllib.request
import zipfile
import os

URL = "https://archive.ics.uci.edu/static/public/522/south+german+credit.zip"
RAW_DIR = os.path.join("data", "raw")
ZIP_PATH = os.path.join(RAW_DIR, "south_german_credit.zip")

os.makedirs(RAW_DIR, exist_ok=True)

print(f"Downloading from {URL} ...")
urllib.request.urlretrieve(URL, ZIP_PATH)
print(f"Saved to {ZIP_PATH}")

print("Extracting...")
with zipfile.ZipFile(ZIP_PATH, "r") as zf:
    zf.extractall(RAW_DIR)

print(f"Done. Files in {RAW_DIR}:")
for f in os.listdir(RAW_DIR):
    print(f"  {f}")
