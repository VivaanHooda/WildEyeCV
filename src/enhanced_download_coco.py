import os
import requests
import zipfile
from tqdm import tqdm
import json

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

def download_full_coco_dataset():
    """Download the FULL COCO dataset for better training"""
    
    # Create directories
    os.makedirs('coco', exist_ok=True)
    os.makedirs('coco/images', exist_ok=True)
    os.makedirs('coco/annotations', exist_ok=True)
    
    print("=" * 60)
    print("DOWNLOADING FULL COCO DATASET (20GB+)")
    print("This will take a while but provides much better training data")
    print("=" * 60)
    
    # Download annotations first (small file)
    print("\n1. Downloading COCO annotations...")
    annotations_url = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
    annotations_zip = "coco/annotations_trainval2017.zip"
    
    if not os.path.exists(annotations_zip):
        download_file(annotations_url, annotations_zip)
    
    # Extract annotations
    extract_zip(annotations_zip, "coco/")
    
    # Download FULL training set (18GB)
    print("\n2. Downloading FULL COCO training images (18GB)...")
    print("This is the big one - grab a coffee! ☕")
    train_images_url = "http://images.cocodataset.org/zips/train2017.zip"
    train_images_zip = "coco/train2017.zip"
    
    if not os.path.exists(train_images_zip):
        download_file(train_images_url, train_images_zip)
    
    # Extract training images
    print("\n3. Extracting training images...")
    extract_zip(train_images_zip, "coco/images/")
    
    # Download validation set (1GB) for testing
    print("\n4. Downloading validation images (1GB)...")
    val_images_url = "http://images.cocodataset.org/zips/val2017.zip"
    val_images_zip = "coco/val2017.zip"
    
    if not os.path.exists(val_images_zip):
        download_file(val_images_url, val_images_zip)
    
    # Extract validation images
    print("\n5. Extracting validation images...")
    extract_zip(val_images_zip, "coco/images/")
    
    # Analyze dataset
    print("\n6. Analyzing dataset...")
    analyze_coco_dataset()
    
    print("\n" + "=" * 60)
    print("FULL COCO DATASET DOWNLOAD COMPLETED!")
    print("=" * 60)
    print("Dataset structure:")
    print("coco/")
    print("├── images/")
    print("│   ├── train2017/ (118,287 images)")  
    print("│   └── val2017/ (5,000 images)")
    print("└── annotations/")
    print("    ├── instances_train2017.json")
    print("    └── instances_val2017.json")
    print("\nReady for training!")

def analyze_coco_dataset():
    """Analyze the COCO dataset to show animal statistics"""
    try:
        # Load annotations
        with open('coco/annotations/instances_train2017.json', 'r') as f:
            data = json.load(f)
        
        # Animal category mapping
        animal_categories = {
            16: 'bird', 17: 'cat', 18: 'dog', 19: 'horse', 20: 'sheep', 
            21: 'cow', 22: 'elephant', 23: 'bear', 24: 'zebra', 25: 'giraffe'
        }
        
        # Count annotations per animal
        animal_counts = {}
        for ann in data['annotations']:
            cat_id = ann['category_id']
            if cat_id in animal_categories:
                animal_name = animal_categories[cat_id]
                animal_counts[animal_name] = animal_counts.get(animal_name, 0) + 1
        
        print("\nAnimal annotations in training set:")
        for animal, count in sorted(animal_counts.items()):
            print(f"  {animal}: {count:,} annotations")
        
        print(f"\nTotal animal annotations: {sum(animal_counts.values()):,}")
        print(f"Total images: {len(data['images']):,}")
        
    except Exception as e:
        print(f"Could not analyze dataset: {e}")

def download_quick_subset():
    """Download a manageable subset for quick testing"""
    print("=" * 60)
    print("DOWNLOADING QUICK SUBSET FOR TESTING")
    print("This downloads validation set only (1GB)")
    print("=" * 60)
    
    # Create directories
    os.makedirs('coco', exist_ok=True)
    os.makedirs('coco/images', exist_ok=True)
    os.makedirs('coco/annotations', exist_ok=True)
    
    # Download annotations
    print("1. Downloading annotations...")
    annotations_url = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
    annotations_zip = "coco/annotations_trainval2017.zip"
    
    if not os.path.exists(annotations_zip):
        download_file(annotations_url, annotations_zip)
    
    extract_zip(annotations_zip, "coco/")
    
    # Download validation images only
    print("2. Downloading validation images (1GB)...")
    val_images_url = "http://images.cocodataset.org/zips/val2017.zip"
    val_images_zip = "coco/val2017.zip"
    
    if not os.path.exists(val_images_zip):
        download_file(val_images_url, val_images_zip)
    
    extract_zip(val_images_zip, "coco/images/")
    
    # Copy val to train for training script compatibility
    if os.path.exists("coco/images/val2017") and not os.path.exists("coco/images/train2017"):
        import shutil
        shutil.copytree("coco/images/val2017", "coco/images/train2017")
        shutil.copy2("coco/annotations/instances_val2017.json", 
                     "coco/annotations/instances_train2017.json")
    
    print("Quick subset ready for testing!")

if __name__ == "__main__":
    import sys
    
    print("COCO Dataset Downloader")
    print("======================")
    print("Choose download option:")
    print("1. Full dataset (20GB) - Best training results")
    print("2. Quick subset (1GB) - For testing")
    print("3. Exit")
    
    while True:
        choice = input("\nEnter choice (1-3): ").strip()
        
        if choice == '1':
            print("\n⚠️  WARNING: This will download 20GB+ of data!")
            confirm = input("Continue? (y/n): ").strip().lower()
            if confirm == 'y':
                download_full_coco_dataset()
                break
            else:
                print("Download cancelled.")
                
        elif choice == '2':
            download_quick_subset()
            break
            
        elif choice == '3':
            print("Exiting...")
            break
            
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")