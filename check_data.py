import os
import sys
import yaml
import pandas as pd
from tqdm import tqdm
from PIL import Image
import numpy as np

# Add src to path
sys.path.append('.')

# Load config
with open('configs/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Check data directory structure
print("\n=== Checking Data Directory Structure ===")
raw_path = os.path.join('data/raw/Breast-Thermography-Raw')
print(f"Raw data path: {os.path.abspath(raw_path)}")
print("\nDirectory contents:")
for item in os.listdir(raw_path):
    item_path = os.path.join(raw_path, item)
    if os.path.isdir(item_path):
        print(f"- {item}/: {len(os.listdir(item_path))} items")
    else:
        print(f"- {item}")

# Check Excel file
print("\n=== Checking Diagnostics File ===")
excel_path = os.path.join(raw_path, 'Diagnostics.xlsx')
if os.path.exists(excel_path):
    try:
        df = pd.read_excel(excel_path)
        print(f"Successfully loaded {excel_path}")
        print("\nColumns in Excel file:")
        print(df.columns.tolist())
        print("\nFirst few rows:")
        print(df.head())
    except Exception as e:
        print(f"Error loading Excel file: {str(e)}")
else:
    print(f"Error: Excel file not found at {excel_path}")

# Check sample images
print("\n=== Checking Sample Images ===")
for label in ['Benign', 'Malignant']:
    label_path = os.path.join(raw_path, label)
    if os.path.exists(label_path):
        print(f"\nChecking {label} images:")
        image_files = [f for f in os.listdir(label_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if image_files:
            sample = image_files[0]
            try:
                img = Image.open(os.path.join(label_path, sample))
                print(f"- Sample image: {sample} - Size: {img.size}, Mode: {img.mode}")
                img_array = np.array(img)
                print(f"  Array shape: {img_array.shape}, Min: {img_array.min()}, Max: {img_array.max()}")
            except Exception as e:
                print(f"- Error loading {sample}: {str(e)}")
        else:
            print(f"- No image files found in {label_path}")
    else:
        print(f"- Directory not found: {label_path}")
