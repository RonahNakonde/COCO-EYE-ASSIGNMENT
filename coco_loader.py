from pycocotools.coco import COCO
import numpy as np
import matplotlib.pyplot as plt
import requests
import json
import os
import zipfile
from pprint import pprint

# Define the URL and paths
annotations_url = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
base_dir = os.path.join(os.getcwd(), "coco_data")
annotations_zip_path = os.path.join(base_dir, "annotations_trainval2017.zip")
annotations_extract_path = os.path.join(base_dir, "annotations")

# Ensure directories exist
os.makedirs(base_dir, exist_ok=True) 

def download_file(url, output_path):
    """ Downloads the file from the COCO URL and saves it to the specified output path."""
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Downloaded: {output_path}")
    else:
        raise Exception(f"Failed to download {url}. Status code: {response.status_code}")

def extract_zip(zip_path, extract_to):
    """ Extracts the ZIP file to the specified directory."""
    if zipfile.is_zipfile(zip_path):
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print(f"Extracted: {zip_path} to {extract_to}")
    else:
        raise Exception(f"The file {zip_path} is not a valid ZIP file.")

# Download the annotations ZIP file
if not os.path.exists(annotations_zip_path):
    print("Downloading annotations ZIP file...")
    download_file(annotations_url, annotations_zip_path)

# Extract the ZIP file.
if not os.path.exists(annotations_extract_path):
    print("Extracting annotations ZIP file...")
    extract_zip(annotations_zip_path, annotations_extract_path)

# Locate the JSON file
annotations_path = os.path.join(annotations_extract_path, "annotations", "instances_val2017.json")
if not os.path.exists(annotations_path):
    raise FileNotFoundError(f"Annotations JSON file not found at {annotations_path}")

print(f"Using annotations file: {annotations_path}")

# Load the JSON data into a dictionary.
with open(annotations_path, "r") as f:
    data = json.load(f)

# Explore the data
for key in data.keys():
    print(f"{key}:")
    pprint(data[key])
    print("-----------------")
