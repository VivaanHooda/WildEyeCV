#!/usr/bin/env python3
"""
Simple COCO cleaner to fix zero-dimension bounding boxes
"""
import json
import shutil
import os

def clean_coco_annotations(input_file):
    """Clean COCO annotation file by fixing/removing invalid bboxes."""
    
    # Create backup
    backup_file = input_file + '.backup'
    if not os.path.exists(backup_file):
        shutil.copy2(input_file, backup_file)
        print(f"✓ Created backup: {backup_file}")
    
    # Load data
    print(f"Loading {input_file}...")
    with open(input_file, 'r') as f:
        coco_data = json.load(f)
    
    print(f"Original annotations: {len(coco_data['annotations'])}")
    
    # Create image info lookup
    img_info = {img['id']: (img['width'], img['height']) for img in coco_data['images']}
    
    # Clean annotations
    cleaned_annotations = []
    fixed_count = 0
    removed_count = 0
    
    for ann in coco_data['annotations']:
        if 'bbox' not in ann:
            cleaned_annotations.append(ann)
            continue
        
        x, y, w, h = ann['bbox']
        img_id = ann['image_id']
        
        # Get image dimensions
        if img_id not in img_info:
            print(f"Warning: Missing image info for annotation {ann['id']}")
            removed_count += 1
            continue
        
        img_w, img_h = img_info[img_id]
        
        # Fix invalid dimensions
        original_bbox = ann['bbox'][:]
        
        # Fix negative or zero width/height
        if w <= 0:
            w = 5.0  # minimum 5 pixels
            fixed_count += 1
        
        if h <= 0:
            h = 5.0  # minimum 5 pixels  
            fixed_count += 1
        
        # Fix negative coordinates by making width/height negative
        if w < 0:
            x = x + w
            w = -w
            fixed_count += 1
        
        if h < 0:
            y = y + h
            h = -h
            fixed_count += 1
        
        # Clamp to image boundaries
        x = max(0, min(img_w - w, x))
        y = max(0, min(img_h - h, y))
        w = min(w, img_w - x)
        h = min(h, img_h - y)
        
        # Final validation
        if w > 0 and h > 0 and x >= 0 and y >= 0 and (x + w) <= img_w and (y + h) <= img_h:
            ann['bbox'] = [x, y, w, h]
            ann['area'] = w * h  # Update area
            cleaned_annotations.append(ann)
            
            if original_bbox != ann['bbox']:
                print(f"Fixed bbox {ann['id']}: {original_bbox} -> {ann['bbox']}")
        else:
            removed_count += 1
            print(f"Removed invalid bbox {ann['id']}: {original_bbox}")
    
    # Update data
    coco_data['annotations'] = cleaned_annotations
    
    # Save cleaned file
    with open(input_file, 'w') as f:
        json.dump(coco_data, f, indent=1)  # Use indent=1 for smaller file
    
    print(f"✓ Cleaned annotations: {len(cleaned_annotations)}")
    print(f"✓ Fixed: {fixed_count}")
    print(f"✓ Removed: {removed_count}")
    print(f"✓ Saved: {input_file}")

if __name__ == "__main__":
    # Clean both train and val files
    train_file = '/workspace/coco/annotations/instances_train2017.json'
    val_file = '/workspace/coco/annotations/instances_val2017.json'
    
    if os.path.exists(train_file):
        print("=== Cleaning Training Annotations ===")
        clean_coco_annotations(train_file)
    
    if os.path.exists(val_file):
        print("\n=== Cleaning Validation Annotations ===")
        clean_coco_annotations(val_file)
    
    print("\n✓ All done! Your dataset should now work without bbox errors.")