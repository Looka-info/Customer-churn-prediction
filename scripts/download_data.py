"""Script to download and verify the IBM Telco Customer Churn dataset."""
import os
import urllib.request
import hashlib
import pandas as pd

DATA_DIR = os.path.join("data", "raw")
DEST_FILE = os.path.join(DATA_DIR, "WA_Fn-UseC_-Telco-Customer-Churn.csv")
URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"

def download_dataset():
    os.makedirs(DATA_DIR, exist_ok=True)
    if os.path.exists(DEST_FILE):
        print(f"Dataset already exists at {DEST_FILE}")
    else:
        print(f"Downloading from {URL} to {DEST_FILE}...")
        req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as resp, open(DEST_FILE, "wb") as f:
            f.write(resp.read())
        print("Download complete.")

    file_size = os.path.getsize(DEST_FILE)
    print(f"File size: {file_size:,} bytes")
    with open(DEST_FILE, "rb") as f:
        sha256 = hashlib.sha256(f.read()).hexdigest()
    print(f"SHA256: {sha256}")

    df = pd.read_csv(DEST_FILE)
    print(f"Dataset shape: {df.shape}")
    print(f"Columns ({len(df.columns)}): {list(df.columns)}")
    print("Target distribution:")
    print(df["Churn"].value_counts(dropna=False))
    print("\nEmpty TotalCharges strings count:")
    print((df["TotalCharges"] == " ").sum())

if __name__ == "__main__":
    download_dataset()
