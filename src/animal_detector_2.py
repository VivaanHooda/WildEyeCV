import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_320_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as F
import torchvision.transforms as T
from pycocotools.coco import COCO
from PIL import Image, ImageDraw
import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
import requests
import gc


def download_file(url, dest_path):
    """Helper function to download files"""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024
    progress_bar = tqdm(total=total_size, unit='iB', unit_scale=True)
    
    with open(dest_path, 'wb') as f:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            f.write(data)
    progress_bar.close()


class COCOAnimalDataset(Dataset):
    def __init__(self, root_dir, annotation_file, transform=None, max_images_per_category=200):
        self.root_dir = root_dir
        self.coco = COCO(annotation_file)
        
        # COCO animal category IDs and their mapping to our unified indices
        self.category_mapping = {
            18: 0,   # dog -> 0
            19: 1,   # horse -> 1
            20: 2,   # sheep -> 2
            21: 3,   # cow -> 3
            22: 4,   # elephant -> 4
            23: 5,   # bear -> 5
            24: 6,   # zebra -> 6
            25: 7    # giraffe -> 7
        }
        
        # Get images containing selected animals
        self.image_ids = []
        for cat_id in self.category_mapping.keys():
            img_ids = self.coco.getImgIds(catIds=[cat_id])[:max_images_per_category]
            self.image_ids.extend(img_ids)
        self.image_ids = list(set(self.image_ids))
        
        self.transform = transform or T.Compose([
            T.ToTensor(),
            T.RandomHorizontalFlip(0.5)
        ])
    
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        
        # Load image
        img_path = os.path.join(self.root_dir, img_info['file_name'])
        if not os.path.exists(img_path):
            # Return a dummy sample if image doesn't exist
            dummy_image = torch.zeros(3, 224, 224)
            dummy_target = {
                'boxes': torch.zeros(0, 4),
                'labels': torch.zeros(0, dtype=torch.int64),
                'image_id': torch.tensor([idx])
            }
            return dummy_image, dummy_target
            
        image = Image.open(img_path).convert('RGB')
        
        # Get annotations
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        annotations = self.coco.loadAnns(ann_ids)
        
        boxes = []
        labels = []
        
        for ann in annotations:
            if ann['category_id'] in self.category_mapping:
                x, y, w, h = ann['bbox']
                boxes.append([x, y, x+w, y+h])
                labels.append(self.category_mapping[ann['category_id']])
        
        # Handle case where no valid annotations exist
        if len(boxes) == 0:
            boxes = torch.zeros(0, 4)
            labels = torch.zeros(0, dtype=torch.int64)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
        
        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([idx])
        }
        
        if self.transform:
            image = self.transform(image)
            
        return image, target


class AnimalDetector:
    def __init__(self, num_classes, weights_path=None):
        # Initialize model
        self.model = fasterrcnn_mobilenet_v3_large_320_fpn(pretrained=True)
        
        # Modify the box predictor for our number of classes (+1 for background)
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes + 1)
        
        # Load trained weights if provided
        if weights_path and os.path.exists(weights_path):
            self.model.load_state_dict(torch.load(weights_path, map_location='cpu'))
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Unified category names (COCO)
        self.category_names = [
            'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe'
        ]

    def predict(self, frame):
        self.model.eval()  # Set to evaluation mode
        # Convert frame to tensor
        if isinstance(frame, np.ndarray):
            frame = Image.fromarray(frame)
        if isinstance(frame, Image.Image):
            frame = F.to_tensor(frame)
        frame = frame.unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            prediction = self.model(frame)[0]
        
        return prediction
    
    def draw_boxes(self, frame, prediction, confidence_threshold=0.5):
        if isinstance(frame, torch.Tensor):
            frame = F.to_pil_image(frame)
        elif isinstance(frame, np.ndarray):
            frame = Image.fromarray(frame)
            
        draw = ImageDraw.Draw(frame)
        
        boxes = prediction['boxes']
        scores = prediction['scores']
        labels = prediction['labels']
        
        for box, score, label in zip(boxes, scores, labels):
            if score > confidence_threshold:
                box = box.cpu().numpy()
                label_idx = label.cpu().item()
                if 0 <= label_idx < len(self.category_names):
                    label_name = self.category_names[label_idx]
                else:
                    label_name = f"class_{label_idx}"
                    
                # Draw rectangle
                draw.rectangle(box.tolist(), outline='red', width=3)
                # Add label and score
                text = f"{label_name}: {score:.2f}"
                draw.text((box[0], box[1]-10), text, fill='red')
        
        return frame
    
    def train_model(self, train_loader, optimizer, num_epochs=10, accumulation_steps=4):
        self.model.train()
        print(f"Starting training with {len(train_loader)} batches per epoch")
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            total_loss = 0
            optimizer.zero_grad()  # Zero gradients at start of epoch
            valid_batches = 0
            
            for batch_idx, (images, targets) in enumerate(train_loader):
                try:
                    # Clear cache periodically
                    if batch_idx % 10 == 0:
                        torch.cuda.empty_cache()
                    
                    # Filter out empty samples
                    valid_images = []
                    valid_targets = []
                    
                    for img, target in zip(images, targets):
                        if len(target['boxes']) > 0:  # Only include samples with annotations
                            valid_images.append(img.to(self.device))
                            valid_targets.append({k: v.to(self.device) for k, v in target.items()})
                    
                    if len(valid_images) == 0:
                        continue
                    
                    loss_dict = self.model(valid_images, valid_targets)
                    losses = sum(loss for loss in loss_dict.values())
                    
                    # Normalize loss for gradient accumulation
                    losses = losses / accumulation_steps
                    losses.backward()
                    
                    # Only optimize after accumulating several batches
                    if (batch_idx + 1) % accumulation_steps == 0:
                        optimizer.step()
                        optimizer.zero_grad()
                    
                    total_loss += losses.item() * accumulation_steps
                    valid_batches += 1
                    
                    # Free up memory
                    del valid_images, valid_targets, loss_dict, losses
                    torch.cuda.empty_cache()
                    
                    # Print progress every 5% of the epoch
                    if batch_idx % max(1, len(train_loader)//20) == 0:
                        progress = (batch_idx + 1) / len(train_loader) * 100
                        avg_loss = total_loss / max(1, valid_batches)
                        print(f"Progress: {progress:.1f}%, Loss: {avg_loss:.4f}")
                        
                except torch.cuda.OutOfMemoryError:
                    print(f"\nOOM error in batch {batch_idx}. Clearing memory and skipping batch...")
                    torch.cuda.empty_cache()
                    gc.collect()
                    continue
                except Exception as e:
                    print(f"Error in batch {batch_idx}: {e}")
                    continue
            
            avg_loss = total_loss / max(1, valid_batches)
            print(f"Epoch {epoch+1} completed, Average Loss: {avg_loss:.4f}")

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
    
    def load_model(self, path):
        if os.path.exists(path):
            self.model.load_state_dict(torch.load(path, map_location=self.device))
        else:
            print(f"Model file {path} not found!")


def get_data_loaders(coco_config=None, custom_config=None, openimages_config=None, batch_size=2):
    datasets = []
    
    if coco_config:
        coco_dataset = COCOAnimalDataset(
            root_dir=coco_config['root_dir'],
            annotation_file=coco_config['annotation_file'],
            max_images_per_category=coco_config.get('max_images_per_category', 200)
        )
        datasets.append(coco_dataset)
        print(f"COCO dataset loaded with {len(coco_dataset)} samples")
    
    if len(datasets) == 0:
        raise ValueError("No valid datasets provided!")
    
    # Combine all datasets
    combined_dataset = ConcatDataset(datasets)
    
    def collate_fn(batch):
        return tuple(zip(*batch))
    
    # Create data loader
    data_loader = DataLoader(
        combined_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0  # Set to 0 to avoid multiprocessing issues
    )
    
    return data_loader