import os
import requests
import zipfile
from tqdm import tqdm

def download_file(url, dest_path):
    """Download a file with progress bar"""
    print(f"Downloading {url}")
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 8192
    
    with open(dest_path, 'wb') as f, tqdm(
        desc=os.path.basename(dest_path),
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as progress_bar:
        for chunk in response.iter_content(chunk_size=block_size):
            size = f.write(chunk)
            progress_bar.update(size)

def extract_zip(zip_path, extract_to):
    """Extract zip file"""
    print(f"Extracting {zip_path}")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

def download_coco_subset():
    """Download a minimal COCO dataset subset for training"""
    
    # Create directories
    os.makedirs('coco', exist_ok=True)
    os.makedirs('coco/images', exist_ok=True)
    os.makedirs('coco/annotations', exist_ok=True)
    
    # Download annotations (small file)
    print("Downloading COCO annotations...")
    annotations_url = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
    annotations_zip = "coco/annotations_trainval2017.zip"
    
    if not os.path.exists(annotations_zip):
        download_file(annotations_url, annotations_zip)
    
    # Extract annotations
    extract_zip(annotations_zip, "coco/")
    
    # Download a small subset of training images (about 1GB instead of 18GB)
    print("Downloading COCO training images subset...")
    
    # We'll download the validation set which is smaller (1GB vs 18GB for full training set)
    val_images_url = "http://images.cocodataset.org/zips/val2017.zip"
    val_images_zip = "coco/val2017.zip"
    
    if not os.path.exists(val_images_zip):
        download_file(val_images_url, val_images_zip)
    
    # Extract images
    extract_zip(val_images_zip, "coco/images/")
    
    # For training, we'll use the validation set
    # Rename val2017 to train2017 for our training script
    if os.path.exists("coco/images/val2017") and not os.path.exists("coco/images/train2017"):
        os.rename("coco/images/val2017", "coco/images/train2017")
    
    # Also update annotation file reference
    val_ann = "coco/annotations/instances_val2017.json"
    train_ann = "coco/annotations/instances_train2017.json"
    
    if os.path.exists(val_ann) and not os.path.exists(train_ann):
        import shutil
        shutil.copy2(val_ann, train_ann)
    
    print("COCO dataset download completed!")
    print("Dataset structure:")
    print("coco/")
    print("├── images/")
    print("│   └── train2017/")
    print("└── annotations/")
    print("    └── instances_train2017.json")

if __name__ == "__main__":
    download_coco_subset()