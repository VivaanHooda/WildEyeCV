#!/usr/bin/env python3
"""
Quick test to see where dataset loading fails
"""

try:
    print("Testing dataset loading...")
    
    # Test basic imports
    print("1. Testing imports...")
    import torch
    import torchvision
    from enhanced_animal_detector import EnhancedCOCODataset
    print("✓ Imports successful")
    
    # Test annotation file loading
    print("2. Testing annotation file...")
    import json
    with open('coco/annotations/instances_train2017.json', 'r') as f:
        data = json.load(f)
    print(f"✓ Annotation file loaded: {len(data['images'])} images, {len(data['annotations'])} annotations")
    
    # Test dataset creation with minimal settings
    print("3. Testing dataset creation...")
    dataset = EnhancedCOCODataset(
        root_dir='coco/images/train2017',
        annotation_file='coco/annotations/instances_train2017.json',
        is_training=True
    )
    print(f"✓ Dataset created: {len(dataset)} samples")
    
    # Test loading first sample
    print("4. Testing first sample...")
    sample = dataset[0]
    print(f"✓ First sample loaded: image shape {sample[0].shape}")
    
    print("🎉 Dataset loading test PASSED!")
    
except Exception as e:
    print(f"❌ Error at step: {e}")
    import traceback
    traceback.print_exc()