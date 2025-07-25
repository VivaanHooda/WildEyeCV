# main.py

from animal_detector_2 import AnimalDetector, get_data_loaders
import torch
from PIL import Image
import gc
import os

def main():
    torch.cuda.empty_cache()  # Clear any existing allocations
    gc.collect()  # Run garbage collection

    # 1. Initialize the detector
    detector = AnimalDetector(num_classes=8)  # 8 animal categories from COCO
    
    # 2. Set up data loaders using COCO dataset
    coco_config = {
        'root_dir': 'coco/images/train2017',
        'annotation_file': 'coco/annotations/instances_train2017.json',
        'max_images_per_category': 100  # Reduced for faster training
    }
    
    train_loader = get_data_loaders(
        coco_config=coco_config,
        batch_size=2  # Small batch size for memory efficiency
    )
    
    # 3. Train the model
    optimizer = torch.optim.SGD(detector.model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0001)
    detector.train_model(train_loader, optimizer, num_epochs=2)  # Reduced epochs for quick test
    
    # 4. Save the trained model
    detector.save_model('animals_trained.pth')
    print("Model training completed and saved as 'animals_trained.pth'")

if __name__ == "__main__":
    main()