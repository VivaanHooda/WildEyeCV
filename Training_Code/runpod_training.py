#!/usr/bin/env python3
import torch
import os
import time
from datetime import datetime

# Import your detector
from enhanced_animal_detector import CompleteEnhancedDetector, EnhancedCOCODataset

def main():
    print("🚀 Starting RunPod Training Session")
    print("=" * 60)
    
    # Check GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory // 1024**3} GB")
    
    # Initialize detector
    detector = CompleteEnhancedDetector(num_classes=12)
    print("✓ Model initialized")
    
    # Setup datasets
    print("📁 Setting up datasets...")
    train_dataset = EnhancedCOCODataset(
        root_dir='coco/images/train2017',
        annotation_file='coco/annotations/instances_train2017.json',
        is_training=True
    )
    
    val_dataset = EnhancedCOCODataset(
    root_dir='coco/images/train2017',  # Use same as training
    annotation_file='coco/annotations/instances_train2017.json',  # Use same file
    is_training=False
)    
    print(f"✓ Training samples: {len(train_dataset)}")
    print(f"✓ Validation samples: {len(val_dataset)}")
    
    # Training configuration for RunPod
    config = {
        'epochs': 30,  # Reduced for $30 budget
        'batch_size': 4,  # Adjust based on GPU memory
        'learning_rate': 0.001,
        'save_dir': 'checkpoints'
    }
    
    print(f"🔧 Training config: {config}")
    
    # Start training
    start_time = time.time()
    best_model_path = detector.train_model(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        **config
    )
    
    training_time = time.time() - start_time
    print(f"⏱️ Training completed in {training_time/3600:.2f} hours")
    print(f"💾 Best model saved: {best_model_path}")
    
    # Save final model with metadata
    final_model_path = 'animal_detector_final.pth'
    torch.save({
        'model_state_dict': detector.model.state_dict(),
        'num_classes': detector.num_classes,
        'class_names': detector.class_names,
        'training_time': training_time,
        'config': config,
        'timestamp': datetime.now().isoformat()
    }, final_model_path)
    
    print(f"🎉 Final model saved: {final_model_path}")

if __name__ == "__main__":
    main()