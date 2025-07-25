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
import ssl
import urllib.request
import random
from collections import defaultdict

class EnhancedCOCOAnimalDataset(Dataset):
    def __init__(self, root_dir, annotation_file, transform=None, 
                 max_images_per_category=None, min_area=100, augment=True):
        self.root_dir = root_dir
        self.coco = COCO(annotation_file)
        self.min_area = min_area
        
        # Enhanced COCO animal category mapping (10 animals)
        self.category_mapping = {
            16: 0,   # bird -> 0
            17: 1,   # cat -> 1  
            18: 2,   # dog -> 2
            19: 3,   # horse -> 3
            20: 4,   # sheep -> 4
            21: 5,   # cow -> 5
            22: 6,   # elephant -> 6
            23: 7,   # bear -> 7
            24: 8,   # zebra -> 8
            25: 9    # giraffe -> 9
        }
        
        # Get images containing selected animals with better distribution
        self.image_ids = self._get_balanced_image_ids(max_images_per_category)
        
        # Enhanced transforms with augmentation
        if augment:
            self.transform = transform or T.Compose([
                T.ToTensor(),
                T.RandomHorizontalFlip(0.5),
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                T.RandomAdjustSharpness(2, p=0.5),
            ])
        else:
            self.transform = transform or T.Compose([T.ToTensor()])
    
    def _get_balanced_image_ids(self, max_images_per_category):
        """Get balanced image IDs across all animal categories"""
        category_images = defaultdict(set)
        
        # Collect images for each category
        for cat_id in self.category_mapping.keys():
            img_ids = self.coco.getImgIds(catIds=[cat_id])
            category_images[cat_id] = set(img_ids)
        
        # Balance the dataset
        if max_images_per_category:
            for cat_id in category_images:
                if len(category_images[cat_id]) > max_images_per_category:
                    category_images[cat_id] = set(random.sample(
                        list(category_images[cat_id]), max_images_per_category))
        
        # Combine all image IDs
        all_image_ids = set()
        for img_ids in category_images.values():
            all_image_ids.update(img_ids)
        
        print(f"Selected {len(all_image_ids)} images for training")
        for cat_id, img_ids in category_images.items():
            cat_name = [name for name, idx in zip(self.get_category_names(), 
                       range(len(self.category_mapping))) 
                       if self.category_mapping[cat_id] == idx][0]
            print(f"  {cat_name}: {len(img_ids)} images")
        
        return list(all_image_ids)
    
    def get_category_names(self):
        return ['bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 
                'elephant', 'bear', 'zebra', 'giraffe']
    
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
                'image_id': torch.tensor([idx]),
                'area': torch.zeros(0),
                'iscrowd': torch.zeros(0, dtype=torch.int64)
            }
            return dummy_image, dummy_target
            
        image = Image.open(img_path).convert('RGB')
        
        # Get annotations
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        annotations = self.coco.loadAnns(ann_ids)
        
        boxes = []
        labels = []
        areas = []
        iscrowd = []
        
        for ann in annotations:
            if ann['category_id'] in self.category_mapping:
                # Filter small objects
                if ann['area'] < self.min_area:
                    continue
                    
                x, y, w, h = ann['bbox']
                # Ensure valid bounding box
                if w > 1 and h > 1:
                    boxes.append([x, y, x+w, y+h])
                    labels.append(self.category_mapping[ann['category_id']])
                    areas.append(ann['area'])
                    iscrowd.append(ann.get('iscrowd', 0))
        
        # Handle case where no valid annotations exist
        if len(boxes) == 0:
            boxes = torch.zeros(0, 4)
            labels = torch.zeros(0, dtype=torch.int64)
            areas = torch.zeros(0)
            iscrowd = torch.zeros(0, dtype=torch.int64)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            areas = torch.as_tensor(areas, dtype=torch.float32)
            iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)
        
        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([idx]),
            'area': areas,
            'iscrowd': iscrowd
        }
        
        if self.transform:
            image = self.transform(image)
            
        return image, target

class EnhancedAnimalDetector:
    def __init__(self, num_classes, weights_path=None, device=None):
        # Auto-detect device
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
                print(f"Using GPU: {torch.cuda.get_device_name()}")
            elif torch.backends.mps.is_available():
                self.device = torch.device('mps')
                print("Using Apple Silicon GPU (MPS)")
            else:
                self.device = torch.device('cpu')
                print("Using CPU")
        else:
            self.device = device
        
        # Fix SSL certificate issues on macOS
        try:
            ssl_context = ssl._create_unverified_context()
            ssl._create_default_https_context = ssl._create_unverified_context
        except AttributeError:
            pass
        
        # Initialize model with proper weights parameter
        try:
            from torchvision.models.detection import FasterRCNN_MobileNet_V3_Large_320_FPN_Weights
            weights = FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.COCO_V1
            self.model = fasterrcnn_mobilenet_v3_large_320_fpn(weights=weights)
        except ImportError:
            self.model = fasterrcnn_mobilenet_v3_large_320_fpn(pretrained=True)
        
        # Modify the box predictor for our number of classes (+1 for background)
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes + 1)
        
        # Load trained weights if provided
        if weights_path and os.path.exists(weights_path):
            try:
                checkpoint = torch.load(weights_path, map_location=self.device)
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                    print(f"✅ Loaded model and training state from {weights_path}")
                else:
                    self.model.load_state_dict(checkpoint)
                    print(f"✅ Loaded model weights from {weights_path}")
            except Exception as e:
                print(f"⚠️ Failed to load custom weights: {e}")
        
        self.model.to(self.device)
        
        # Enhanced category names (10 animals)
        self.category_names = [
            'bird', 'cat', 'dog', 'horse', 'sheep', 
            'cow', 'elephant', 'bear', 'zebra', 'giraffe'
        ]

    def predict(self, frame):
        self.model.eval()
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
    
    def train_model(self, train_loader, val_loader=None, optimizer=None, scheduler=None, 
                   num_epochs=10, accumulation_steps=4, save_every=2):
        """Enhanced training with validation and checkpointing"""
        self.model.train()
        
        if optimizer is None:
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.001, weight_decay=0.0001)
        
        if scheduler is None:
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.7)
        
        print(f"Starting training with {len(train_loader)} batches per epoch")
        print(f"Using device: {self.device}")
        
        best_loss = float('inf')
        training_history = []
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print("-" * 50)
            
            # Training phase
            self.model.train()
            total_loss = 0
            valid_batches = 0
            
            progress_bar = tqdm(train_loader, desc=f"Training Epoch {epoch+1}")
            
            for batch_idx, (images, targets) in enumerate(progress_bar):
                try:
                    # Clear cache periodically
                    if batch_idx % 20 == 0:
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        gc.collect()
                    
                    # Filter out empty samples
                    valid_images = []
                    valid_targets = []
                    
                    for img, target in zip(images, targets):
                        if len(target['boxes']) > 0:
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
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                        optimizer.step()
                        optimizer.zero_grad()
                    
                    total_loss += losses.item() * accumulation_steps
                    valid_batches += 1
                    
                    # Update progress bar
                    avg_loss = total_loss / max(1, valid_batches)
                    progress_bar.set_postfix({'Loss': f'{avg_loss:.4f}'})
                    
                    # Free up memory
                    del valid_images, valid_targets, loss_dict, losses
                    
                except torch.cuda.OutOfMemoryError:
                    print(f"\nOOM error in batch {batch_idx}. Clearing memory...")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()
                    continue
                except Exception as e:
                    print(f"Error in batch {batch_idx}: {e}")
                    continue
            
            # Calculate epoch metrics
            avg_train_loss = total_loss / max(1, valid_batches)
            
            # Validation phase
            val_loss = 0
            if val_loader:
                val_loss = self._validate(val_loader)
            
            # Update learning rate
            scheduler.step()
            
            # Save checkpoint
            if avg_train_loss < best_loss:
                best_loss = avg_train_loss
                self.save_checkpoint(f'best_model_epoch_{epoch+1}.pth', epoch, optimizer, scheduler, avg_train_loss)
            
            if (epoch + 1) % save_every == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch+1}.pth', epoch, optimizer, scheduler, avg_train_loss)
            
            # Log training progress
            epoch_info = {
                'epoch': epoch + 1,
                'train_loss': avg_train_loss,
                'val_loss': val_loss,
                'lr': optimizer.param_groups[0]['lr']
            }
            training_history.append(epoch_info)
            
            print(f"Epoch {epoch+1} completed:")
            print(f"  Train Loss: {avg_train_loss:.4f}")
            if val_loader:
                print(f"  Val Loss: {val_loss:.4f}")
            print(f"  Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        return training_history
    
    def _validate(self, val_loader):
        """Run validation"""
        self.model.train()  # Keep in training mode for loss calculation
        total_loss = 0
        valid_batches = 0
        
        with torch.no_grad():
            for images, targets in tqdm(val_loader, desc="Validating"):
                try:
                    valid_images = []
                    valid_targets = []
                    
                    for img, target in zip(images, targets):
                        if len(target['boxes']) > 0:
                            valid_images.append(img.to(self.device))
                            valid_targets.append({k: v.to(self.device) for k, v in target.items()})
                    
                    if len(valid_images) == 0:
                        continue
                    
                    loss_dict = self.model(valid_images, valid_targets)
                    losses = sum(loss for loss in loss_dict.values())
                    
                    total_loss += losses.item()
                    valid_batches += 1
                    
                except Exception as e:
                    continue
        
        return total_loss / max(1, valid_batches)
    
    def save_checkpoint(self, path, epoch, optimizer, scheduler, loss):
        """Save training checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': loss,
            'device': str(self.device)
        }
        torch.save(checkpoint, path)
        print(f"✅ Checkpoint saved: {path}")
    
    def save_model(self, path):
        """Save model weights only"""
        torch.save(self.model.state_dict(), path)
    
    def load_model(self, path):
        if os.path.exists(path):
            self.model.load_state_dict(torch.load(path, map_location=self.device))
        else:
            print(f"Model file {path} not found!")

def get_enhanced_data_loaders(train_config, val_config=None, batch_size=4, num_workers=2):
    """Get enhanced data loaders with better memory management"""
    
    # Training dataset
    train_dataset = EnhancedCOCOAnimalDataset(
        root_dir=train_config['root_dir'],
        annotation_file=train_config['annotation_file'],
        max_images_per_category=train_config.get('max_images_per_category'),
        min_area=train_config.get('min_area', 100),
        augment=True
    )
    
    datasets = [train_dataset]
    print(f"Training dataset loaded with {len(train_dataset)} samples")
    
    # Validation dataset (optional)
    val_loader = None
    if val_config:
        val_dataset = EnhancedCOCOAnimalDataset(
            root_dir=val_config['root_dir'],
            annotation_file=val_config['annotation_file'],
            max_images_per_category=val_config.get('max_images_per_category'),
            min_area=val_config.get('min_area', 100),
            augment=False  # No augmentation for validation
        )
        print(f"Validation dataset loaded with {len(val_dataset)} samples")
        
        def collate_fn(batch):
            return tuple(zip(*batch))
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available()
        )
    
    # Training data loader
    def collate_fn(batch):
        return tuple(zip(*batch))
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0
    )
    
    return train_loader, val_loader