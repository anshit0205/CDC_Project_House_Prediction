"""
Satellite Image Downloader for House Price Prediction
Downloads satellite imagery from Mapbox API for each property location.
"""

import os
import time
import requests
import pandas as pd
from tqdm import tqdm

# Configuration
# TODO: Replace with your Mapbox API token from https://www.mapbox.com/
MAPBOX_TOKEN = "YOUR_MAPBOX_TOKEN_HERE"
ZOOM = 18
IMG_SIZE = "512x512"
SCALE = 2
STYLE = "satellite-v9"

def download_satellite_images(csv_path, output_dir="mapbox_images"):
    """
    Download satellite images for each property in the dataset.
    
    Parameters:
    -----------
    csv_path : str
        Path to CSV file containing property data with 'id', 'lat', 'long' columns
    output_dir : str
        Directory to save downloaded images
    
    Returns:
    --------
    None
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    df = pd.read_csv(csv_path)
    print(f"📊 Loaded {len(df)} properties from {csv_path}")
    
    downloaded = 0
    skipped = 0
    failed = 0
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Downloading images"):
        pid = row["id"]
        lat = row["lat"]
        lon = row["long"]
        
        # Generate filename
        fname = f"{pid}_z{ZOOM}_s{SCALE}.png"
        out_path = os.path.join(output_dir, fname)
        
        # Skip if already exists
        if os.path.exists(out_path):
            skipped += 1
            continue
        
        # Build Mapbox URL
        url = (
            f"https://api.mapbox.com/styles/v1/mapbox/{STYLE}/static/"
            f"{lon},{lat},{ZOOM}/{IMG_SIZE}@{SCALE}x"
            f"?access_token={MAPBOX_TOKEN}"
        )
        
        # Download image
        try:
            r = requests.get(url, timeout=30)
            if r.status_code == 200:
                with open(out_path, "wb") as f:
                    f.write(r.content)
                downloaded += 1
            else:
                print(f"❌ Failed for {pid}: Status {r.status_code}")
                failed += 1
        except Exception as e:
            print(f"❌ Error for {pid}: {e}")
            failed += 1
        
        # Rate limiting
        time.sleep(0.2)
    
    print(f"\n✅ Download complete!")
    print(f"   Downloaded: {downloaded}")
    print(f"   Skipped (already exists): {skipped}")
    print(f"   Failed: {failed}")

if __name__ == "__main__":
    # Example usage
    CSV_FILE = "data/train.csv"  # Update with your path
    OUTPUT_DIR = "data/satellite_images" #change directory name accordingly
    
    download_satellite_images(CSV_FILE, OUTPUT_DIR)
