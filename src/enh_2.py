import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_320_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision import transforms
import torchvision.transforms.functional as F
from PIL import Image, ImageDraw, ImageFont, ImageEnhance
import json
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import math
import time
from collections import defaultdict
import albumentations as A
from albumentations.pytorch import ToTensorV2

class AdvancedTransforms:
    """Advanced data augmentation transforms"""
    
    def __init__(self, is_training=True):
        self.is_training = is_training
        
        if is_training:
            self.transform = A.Compose([
                A.OneOf([
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.1),
                    A.RandomRotate90(p=0.2),
                ], p=0.7),
                
                A.OneOf([
                    A.MotionBlur(blur_limit=5, p=0.3),
                    A.GaussianBlur(blur_limit=3, p=0.3),
                    A.MedianBlur(blur_limit=3, p=0.2),
                ], p=0.3),
                
                A.OneOf([
                    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
                    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.3),
                    A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3),
                ], p=0.5),
                
                A.OneOf([
                    A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, p=0.1),
                    A.RandomRain(slant_lower=-10, slant_upper=10, drop_length=10, p=0.1),
                    A.RandomSunFlare(flare_roi=(0, 0, 1, 0.5), angle_lower=0, p=0.1),
                ], p=0.1),
                
                A.Resize(height=800, width=800, p=1.0),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))
        else:
            self.transform = A.Compose([
                A.Resize(height=800, width=800, p=1.0),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))
    
    def __call__(self, image, bboxes, labels):
        transformed = self.transform(image=np.array(image), bboxes=bboxes, labels=labels)
        return transformed['image'], transformed['bboxes'], transformed['labels']

class MixUpCollate:
    """MixUp data augmentation for object detection"""
    
    def __init__(self, alpha=0.2):
        self.alpha = alpha
    
    def __call__(self, batch):
        if random.random() > 0.5:  # Apply MixUp 50% of the time
            return self.mixup_batch(batch)
        else:
            return self.standard_collate(batch)
    
    def mixup_batch(self, batch):
        # Implement MixUp for object detection
        # This is a simplified version - full implementation would be more complex
        return self.standard_collate(batch)
    
    def standard_collate(self, batch):
        images = []
        targets = []
        
        for image, target in batch:
            images.append(image)
            targets.append(target)
        
        return images, targets

class EnhancedCOCODataset(Dataset):
    """Enhanced COCO dataset with advanced augmentations"""
    
    def __init__(self, root_dir, annotation_file, transforms=None, is_training=True):
        self.root_dir = root_dir
        self.transforms = transforms or AdvancedTransforms(is_training)
        self.is_training = is_training
        
        # Load annotations
        with open(annotation_file, 'r') as f:
            self.coco_data = json.load(f)
        
        # COCO category mapping to our classes
        self.coco_to_custom = {
            1: 1,   # person -> human
            16: 2,  # bird -> bird
            17: 3,  # cat -> cat
            18: 4,  # dog -> dog
            19: 5,  # horse -> horse
            20: 6,  # sheep -> sheep
            21: 7,  # cow -> cow
            22: 8,  # elephant -> elephant
            23: 9,  # bear -> bear
            24: 10, # zebra -> zebra
            25: 11, # giraffe -> giraffe
        }
        
        # Filter images that contain our target classes
        self.valid_images = []
        for image_info in self.coco_data['images']:
            image_id = image_info['id']
            annotations = [ann for ann in self.coco_data['annotations'] 
                         if ann['image_id'] == image_id and ann['category_id'] in self.coco_to_custom]
            if annotations:
                self.valid_images.append((image_info, annotations))
        
        print(f"Loaded {len(self.valid_images)} images with target classes")
    
    def __len__(self):
        return len(self.valid_images)
    
    def __getitem__(self, idx):
        image_info, annotations = self.valid_images[idx]
        
        # Load image
        image_path = os.path.join(self.root_dir, image_info['file_name'])
        image = Image.open(image_path).convert('RGB')
        
        # Process annotations
        boxes = []
        labels = []
        
        for ann in annotations:
            if ann['category_id'] in self.coco_to_custom:
                # Convert COCO bbox format (x, y, width, height) to (x1, y1, x2, y2)
                x, y, w, h = ann['bbox']
                boxes.append([x, y, x + w, y + h])
                labels.append(self.coco_to_custom[ann['category_id']])
        
        if not boxes:
            # Return empty detection if no valid boxes
            return torch.zeros((3, 800, 800)), {
                'boxes': torch.zeros((0, 4)),
                'labels': torch.zeros((0,), dtype=torch.int64),
                'image_id': torch.tensor([image_info['id']]),
                'area': torch.zeros((0,)),
                'iscrowd': torch.zeros((0,), dtype=torch.int64)
            }
        
        # Apply transforms
        if self.transforms:
            image, boxes, labels = self.transforms(image, boxes, labels)
        
        # Convert to tensors
        if not isinstance(image, torch.Tensor):
            image = transforms.ToTensor()(image)
        
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        
        # Calculate areas
        areas = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        
        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([image_info['id']]),
            'area': areas,
            'iscrowd': torch.zeros((len(boxes),), dtype=torch.int64)
        }
        
        return image, target

class CompleteEnhancedDetector:
    """Complete enhanced human and animal detector with advanced training"""
    
    def __init__(self, num_classes=12):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_classes = num_classes
        
        # Class names
        self.class_names = [
            'background', 'human', 'bird', 'cat', 'dog', 'horse', 
            'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe'
        ]
        
        # Colors for visualization (BGR format for OpenCV)
        self.colors = {
            'human': (0, 255, 0),      # Green for humans
            'bird': (255, 0, 0),       # Blue for birds
            'cat': (0, 165, 255),      # Orange for cats
            'dog': (0, 0, 255),        # Red for dogs
            'horse': (255, 255, 0),    # Cyan for horses
            'sheep': (255, 255, 255),  # White for sheep
            'cow': (128, 0, 128),      # Purple for cows
            'elephant': (128, 128, 128), # Gray for elephants
            'bear': (0, 100, 0),       # Dark green for bears
            'zebra': (255, 0, 255),    # Magenta for zebras
            'giraffe': (0, 255, 255)   # Yellow for giraffes
        }
        
        # Initialize model
        self.model = self._create_model()
        self.model.to(self.device)
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.learning_rates = []
        
    def _create_model(self):
        """Create enhanced Faster R-CNN model"""
        model = fasterrcnn_mobilenet_v3_large_320_fpn(
            pretrained=True,
            trainable_backbone_layers=3  # Fine-tune more layers
        )
        
        # Replace classifier
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, self.num_classes)
        
        return model
    
    def create_optimizer(self, learning_rate=0.001):
        """Create AdamW optimizer with weight decay"""
        params = []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                # Different learning rates for different parts
                if 'backbone' in name:
                    params.append({'params': param, 'lr': learning_rate * 0.1})
                else:
                    params.append({'params': param, 'lr': learning_rate})
        
        optimizer = optim.AdamW(params, weight_decay=0.0001)
        return optimizer
    
    def create_scheduler(self, optimizer, total_steps):
        """Create learning rate scheduler with warmup"""
        def lr_lambda(step):
            if step < 1000:  # Warmup for first 1000 steps
                return step / 1000.0
            else:
                # Cosine annealing after warmup
                return 0.5 * (1 + math.cos(math.pi * (step - 1000) / (total_steps - 1000)))
        
        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    def train_epoch(self, dataloader, optimizer, scheduler, scaler, epoch):
        """Train for one epoch with advanced features"""
        self.model.train()
        epoch_loss = 0
        num_batches = 0
        
        progress_bar = tqdm(dataloader, desc=f'Epoch {epoch}')
        
        for batch_idx, (images, targets) in enumerate(progress_bar):
            # Move to device
            images = [img.to(self.device) for img in images]
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
            
            # Forward pass with mixed precision
            optimizer.zero_grad()
            
            with torch.cuda.amp.autocast():
                loss_dict = self.model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
            
            # Backward pass
            scaler.scale(losses).backward()
            
            # Gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            # Update metrics
            epoch_loss += losses.item()
            num_batches += 1
            
            # Update progress bar
            current_lr = scheduler.get_last_lr()[0]
            progress_bar.set_postfix({
                'Loss': f'{losses.item():.4f}',
                'Avg Loss': f'{epoch_loss/num_batches:.4f}',
                'LR': f'{current_lr:.6f}'
            })
            
            self.learning_rates.append(current_lr)
        
        avg_loss = epoch_loss / num_batches
        self.train_losses.append(avg_loss)
        return avg_loss
    
    def validate_epoch(self, dataloader):
        """Validate model performance"""
        self.model.eval()
        val_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for images, targets in tqdm(dataloader, desc='Validation'):
                images = [img.to(self.device) for img in images]
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
                
                # Forward pass
                loss_dict = self.model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                
                val_loss += losses.item()
                num_batches += 1
        
        avg_val_loss = val_loss / num_batches
        self.val_losses.append(avg_val_loss)
        return avg_val_loss
    
    def train_model(self, train_dataset, val_dataset, epochs=50, batch_size=8, 
                   learning_rate=0.001, save_dir='checkpoints'):
        """Complete training pipeline"""
        os.makedirs(save_dir, exist_ok=True)
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=4,
            collate_fn=MixUpCollate(),
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=4,
            collate_fn=lambda x: tuple(zip(*x)),
            pin_memory=True
        )
        
        # Setup training components
        optimizer = self.create_optimizer(learning_rate)
        total_steps = len(train_loader) * epochs
        scheduler = self.create_scheduler(optimizer, total_steps)
        scaler = torch.cuda.amp.GradScaler()
        
        best_val_loss = float('inf')
        patience = 10
        patience_counter = 0
        
        print(f"Starting training for {epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Batch size: {batch_size}")
        print(f"Learning rate: {learning_rate}")
        
        for epoch in range(1, epochs + 1):
            print(f"\nEpoch {epoch}/{epochs}")
            print("-" * 50)
            
            # Training
            train_loss = self.train_epoch(train_loader, optimizer, scheduler, scaler, epoch)
            
            # Validation
            val_loss = self.validate_epoch(val_loader)
            
            print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            # Save checkpoint
            if epoch % 5 == 0:
                checkpoint_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch}.pth')
                self.save_checkpoint(checkpoint_path, epoch, optimizer, scheduler)
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_path = os.path.join(save_dir, 'best_model.pth')
                self.save_model(best_model_path)
                print(f"New best model saved! Val Loss: {val_loss:.4f}")
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch} epochs")
                break
            
            # Plot progress
            if epoch % 10 == 0:
                self.plot_training_progress(save_dir)
        
        print("Training completed!")
        return best_model_path
    
    def save_checkpoint(self, path, epoch, optimizer, scheduler):
        """Save training checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'learning_rates': self.learning_rates
        }
        torch.save(checkpoint, path)
    
    def save_model(self, path):
        """Save model weights only"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'num_classes': self.num_classes,
            'class_names': self.class_names
        }, path)
    
    def load_model(self, path):
        """Load trained model"""
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Model loaded from {path}")
            return True
        else:
            print(f"Model file {path} not found")
            return False
    
    def plot_training_progress(self, save_dir):
        """Plot training progress"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss curves
        axes[0, 0].plot(self.train_losses, label='Train Loss', color='blue')
        axes[0, 0].plot(self.val_losses, label='Val Loss', color='red')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Learning rate
        axes[0, 1].plot(self.learning_rates, color='green')
        axes[0, 1].set_title('Learning Rate Schedule')
        axes[0, 1].set_xlabel('Step')
        axes[0, 1].set_ylabel('Learning Rate')
        axes[0, 1].grid(True)
        
        # Loss difference
        if len(self.train_losses) > 1 and len(self.val_losses) > 1:
            loss_diff = [val - train for train, val in zip(self.train_losses, self.val_losses)]
            axes[1, 0].plot(loss_diff, color='purple')
            axes[1, 0].set_title('Validation - Training Loss (Overfitting Indicator)')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Loss Difference')
            axes[1, 0].grid(True)
        
        # Recent loss trend
        if len(self.train_losses) > 10:
            recent_train = self.train_losses[-10:]
            recent_val = self.val_losses[-10:]
            axes[1, 1].plot(recent_train, label='Recent Train Loss', color='blue')
            axes[1, 1].plot(recent_val, label='Recent Val Loss', color='red')
            axes[1, 1].set_title('Recent Loss Trend (Last 10 Epochs)')
            axes[1, 1].set_xlabel('Recent Epochs')
            axes[1, 1].set_ylabel('Loss')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'training_progress.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def detect_image(self, image_path, confidence_threshold=0.5):
        """Detect objects in image"""
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        transform = transforms.Compose([
            transforms.Resize((800, 800)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        image_tensor = transform(image).unsqueeze(0).to(self.device)
        
        # Run inference
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(image_tensor)
        
        # Process predictions
        boxes = predictions[0]['boxes'].cpu().numpy()
        scores = predictions[0]['scores'].cpu().numpy()
        labels = predictions[0]['labels'].cpu().numpy()
        
        # Filter by confidence
        mask = scores > confidence_threshold
        boxes = boxes[mask]
        scores = scores[mask]
        labels = labels[mask]
        
        return boxes, scores, labels, image
    
    def draw_detections(self, image, boxes, scores, labels, save_path=None):
        """Draw bounding boxes and labels on image"""
        # Convert PIL to OpenCV format
        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        for box, score, label in zip(boxes, scores, labels):
            if label < len(self.class_names):
                class_name = self.class_names[label]
                color = self.colors.get(class_name, (255, 255, 255))
                
                # Draw bounding box
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(img_cv, (x1, y1), (x2, y2), color, 3)
                
                # Draw label with confidence
                label_text = f"{class_name}: {score:.2f}"
                
                # Get text size for background rectangle
                (text_width, text_height), baseline = cv2.getTextSize(
                    label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2
                )
                
                # Draw background rectangle for text
                cv2.rectangle(img_cv, (x1, y1 - text_height - 15), 
                            (x1 + text_width, y1), color, -1)
                
                # Draw text
                cv2.putText(img_cv, label_text, (x1, y1 - 5),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        
        if save_path:
            cv2.imwrite(save_path, img_cv)
            print(f"Detection result saved to {save_path}")
        
        return img_cv
    
    def detect_and_display(self, image_path, confidence_threshold=0.5, save_path=None):
        """Complete detection pipeline with visualization"""
        print(f"Processing: {image_path}")
        
        # Detect objects
        boxes, scores, labels, original_image = self.detect_image(image_path, confidence_threshold)
        
        # Print detections
        print(f"Found {len(boxes)} detections:")
        for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
            if label < len(self.class_names):
                class_name = self.class_names[label]
                print(f"  {i+1}. {class_name}: {score:.3f}")
        
        # Draw and save result
        result_image = self.draw_detections(original_image, boxes, scores, labels, save_path)
        
        return result_image, boxes, scores, labels
    
    def detect_webcam(self, confidence_threshold=0.3):
        """Real-time detection from webcam"""
        cap = cv2.VideoCapture(0)
        
        print("Starting webcam detection. Press 'q' to quit.")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Save frame temporarily
            temp_path = 'temp_frame.jpg'
            cv2.imwrite(temp_path, frame)
            
            # Detect
            try:
                boxes, scores, labels, _ = self.detect_image(temp_path, confidence_threshold)
                
                # Draw on original frame
                for box, score, label in zip(boxes, scores, labels):
                    if label < len(self.class_names):
                        class_name = self.class_names[label]
                        color = self.colors.get(class_name, (255, 255, 255))
                        
                        x1, y1, x2, y2 = map(int, box)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        
                        label_text = f"{class_name}: {score:.2f}"
                        cv2.putText(frame, label_text, (x1, y1 - 10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            except Exception as e:
                print(f"Detection error: {e}")
            
            cv2.imshow('Enhanced Human & Animal Detection', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        if os.path.exists(temp_path):
            os.remove(temp_path)

# Training script
def train_complete_model():
    """Complete training pipeline"""
    print("Setting up Complete Enhanced Human & Animal Detector")
    
    # Initialize detector
    detector = CompleteEnhancedDetector()
    
    # Setup datasets
    train_dataset = EnhancedCOCODataset(
        root_dir='coco/train2017',
        annotation_file='coco/annotations/instances_train2017.json',
        is_training=True
    )
    
    val_dataset = EnhancedCOCODataset(
        root_dir='coco/val2017',
        annotation_file='coco/annotations/instances_val2017.json',
        is_training=False
    )
    
    print(f"Training dataset: {len(train_dataset)} images")
    print(f"Validation dataset: {len(val_dataset)} images")
    
    # Start training
    best_model_path = detector.train_model(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        epochs=50,
        batch_size=8,
        learning_rate=0.001
    )
    
    print(f"Training completed! Best model saved at: {best_model_path}")
    return detector

# Testing functions
def test_detection():
    """Test the trained model"""
    detector = CompleteEnhancedDetector()
    
    # Load trained model
    if detector.load_model('checkpoints/best_model.pth'):
        print("Model loaded successfully!")
        
        # Test on sample images
        test_images = ['test_image.jpg', 'sample.jpg', 'demo.png']
        
        for img_path in test_images:
            if os.path.exists(img_path):
                print(f"\n{'='*50}")
                result_img, boxes, scores, labels = detector.detect_and_display(
                    img_path, 
                    confidence_threshold=0.5,
                    save_path=f"detected_{os.path.basename(img_path)}"
                )
                
                # Show result
                cv2.imshow(f'Detections - {img_path}', result_img)
                cv2.waitKey(3000)
                cv2.destroyAllWindows()
    else:
        print("No trained model found. Please train the model first.")

if __name__ == "__main__":
    print("Complete Enhanced Human & Animal Detector")
    print("1. Train new model")
    print("2. Test existing model")
    print("3. Real-time webcam detection")
    
    choice = input("Choose option (1, 2, or 3): ")
    
    if choice == '1':
        train_complete_model()
    elif choice == '2':
        test_detection()
    elif choice == '3':
        detector = CompleteEnhancedDetector()
        if detector.load_model('checkpoints/best_model.pth'):
            detector.detect_webcam()
        else:
            print("No trained model found. Please train first.")
    else:
        print("Invalid choice. Please run again.")