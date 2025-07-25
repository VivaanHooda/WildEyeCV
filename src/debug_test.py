# debug_test.py - Debug version with more verbose output

import cv2
import torch
from animal_detector_2 import AnimalDetector
from PIL import Image
import numpy as np
import os
import sys

def start_debug_detection():
    print("🔍 Starting debug detection...")
    
    # Initialize detector
    detector = AnimalDetector(num_classes=8)
    
    # Try to load any available model
    model_files = ['animals_trained.pth', 'animals2.pth', 'animals3.pth', 'model.pth']
    model_loaded = False
    
    for model_file in model_files:
        if os.path.exists(model_file):
            try:
                detector.load_model(model_file)
                print(f"✅ Loaded model: {model_file}")
                model_loaded = True
                break
            except Exception as e:
                print(f"❌ Failed to load {model_file}: {e}")
    
    if not model_loaded:
        print("⚠️ Using pretrained model (may not detect animals well)")
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Could not open webcam")
        return
    
    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 15)
    
    print("🎯 Detection settings:")
    print("- Confidence threshold: 0.1 (very low for debugging)")
    print("- Processing every frame")
    print("- Press 'q' to quit, 's' to save frame")
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Convert to RGB for the model
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        
        try:
            # Get predictions
            predictions = detector.predict(pil_image)
            
            # Debug: Print all detections regardless of confidence
            boxes = predictions['boxes']
            labels = predictions['labels']
            scores = predictions['scores']
            
            print(f"\nFrame {frame_count}: Found {len(boxes)} detections")
            
            detection_found = False
            for i, (box, label, score) in enumerate(zip(boxes, labels, scores)):
                label_idx = label.item()
                score_val = score.item()
                
                print(f"  Detection {i}: Label={label_idx}, Score={score_val:.3f}")
                
                # Draw ALL detections, even very low confidence ones
                if score_val >= 0.1:  # Very low threshold for debugging
                    detection_found = True
                    
                    # Get animal name
                    if 0 <= label_idx < len(detector.category_names):
                        animal_name = detector.category_names[label_idx]
                    else:
                        animal_name = f"class_{label_idx}"
                    
                    # Draw bounding box
                    box_coords = box.cpu().numpy().astype(int)
                    
                    # Color based on confidence
                    if score_val >= 0.5:
                        color = (0, 255, 0)  # Green for high confidence
                    elif score_val >= 0.3:
                        color = (0, 255, 255)  # Yellow for medium confidence
                    else:
                        color = (0, 0, 255)  # Red for low confidence
                    
                    cv2.rectangle(frame, (box_coords[0], box_coords[1]), 
                                (box_coords[2], box_coords[3]), color, 2)
                    
                    # Draw label
                    label_text = f"{animal_name}: {score_val:.3f}"
                    cv2.putText(frame, label_text, 
                              (box_coords[0], box_coords[1] - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    
                    print(f"    -> Drew box for {animal_name} (confidence: {score_val:.3f})")
            
            if not detection_found:
                print("    -> No detections above threshold 0.1")
                
        except Exception as e:
            print(f"❌ Error during prediction: {e}")
        
        # Add debug info to frame
        cv2.putText(frame, f"Frame: {frame_count}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, "Debug Mode - Press 'q' to quit, 's' to save", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Display frame
        cv2.imshow('Debug Animal Detection', frame)
        
        # Handle keys
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            filename = f"debug_frame_{frame_count}.jpg"
            cv2.imwrite(filename, frame)
            print(f"💾 Saved {filename}")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("🏁 Debug detection finished")

if __name__ == "__main__":
    try:
        start_debug_detection()
    except KeyboardInterrupt:
        print("\n👋 Stopped by user")
    except Exception as e:
        print(f"❌ Error: {e}")