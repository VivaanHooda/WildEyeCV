# mac_animal_test.py - Optimized for Mac
import cv2
import torch
import sys
import os
from animal_detector_2 import AnimalDetector
from PIL import Image
import numpy as np

def check_camera_permissions():
    """Check if camera is accessible"""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Camera not accessible!")
        print("Please check:")
        print("1. Camera permissions in System Preferences → Security & Privacy → Camera")
        print("2. No other apps are using the camera")
        print("3. Camera is properly connected")
        return False
    cap.release()
    return True

def start_detection():
    print("🐾 Animal Detection for Mac")
    print("=" * 40)
    
    # Check camera first
    if not check_camera_permissions():
        return
    
    # Initialize detector
    print("🔄 Loading AI model...")
    detector = AnimalDetector(num_classes=8)
    
    # Try to load trained model
    model_files = ['animals_trained.pth', 'animals2.pth', 'animals3.pth']
    model_loaded = False
    
    for model_file in model_files:
        if os.path.exists(model_file):
            try:
                detector.load_model(model_file)
                print(f"✅ Loaded trained model: {model_file}")
                model_loaded = True
                break
            except Exception as e:
                print(f"⚠️  Failed to load {model_file}: {e}")
    
    if not model_loaded:
        print("⚠️  Using pretrained model (no custom animal training)")
    
    # Initialize camera with Mac-optimized settings
    print("📷 Initializing camera...")
    cap = cv2.VideoCapture(0)
    
    # Mac-optimized camera settings
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 15)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce latency
    
    if not cap.isOpened():
        print("❌ Failed to open camera!")
        return
    
    print("✅ Camera ready!")
    print("\n🎯 Controls:")
    print("- Press 'q' to quit")
    print("- Press 's' to save current frame")
    print("- Keep animals in view for 3+ seconds for detection")
    print("\n🔍 Detecting: dog, horse, sheep, cow, elephant, bear, zebra, giraffe")
    print("\nStarting detection...")
    
    # Detection variables
    detection_count = {}
    frame_count = 0
    save_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Failed to read frame")
                break
            
            frame_count += 1
            
            # Process every 3rd frame for better performance on Mac
            if frame_count % 3 != 0:
                # Just display the frame without processing
                cv2.putText(frame, "Animal Detection - Press 'q' to quit", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow('🐾 Animal Detection', frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                continue
            
            # Convert for ML processing
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            
            try:
                # Get predictions
                with torch.no_grad():  # Optimize for inference
                    predictions = detector.predict(pil_image)
                
                # Process detections
                boxes = predictions['boxes']
                labels = predictions['labels']
                scores = predictions['scores']
                
                current_detections = []
                
                # Draw detections
                for box, label, score in zip(boxes, labels, scores):
                    if score >= 0.4:  # Confidence threshold
                        label_idx = label.item()
                        if 0 <= label_idx < len(detector.category_names):
                            animal_name = detector.category_names[label_idx]
                            current_detections.append(animal_name)
                            
                            # Update detection count
                            detection_count[animal_name] = detection_count.get(animal_name, 0) + 1
                            
                            # Draw bounding box
                            box_coords = box.cpu().numpy().astype(int)
                            cv2.rectangle(frame, (box_coords[0], box_coords[1]), 
                                        (box_coords[2], box_coords[3]), (0, 255, 0), 2)
                            
                            # Draw label
                            label_text = f"{animal_name}: {score:.2f}"
                            cv2.putText(frame, label_text, 
                                      (box_coords[0], box_coords[1] - 10),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                            
                            # Print detection
                            if detection_count[animal_name] % 10 == 1:  # Print every 10th detection
                                print(f"🎯 Detected: {animal_name} (confidence: {score:.2f})")
                
                # Reset counts for animals no longer detected
                for animal in list(detection_count.keys()):
                    if animal not in current_detections:
                        if detection_count[animal] > 0:
                            detection_count[animal] = 0
                
            except Exception as e:
                print(f"⚠️  Detection error: {e}")
            
            # Add status information
            y_pos = 30
            cv2.putText(frame, "Animal Detection - Press 'q' to quit, 's' to save", 
                       (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Show detection counts
            y_pos = 60
            for animal, count in detection_count.items():
                if count > 0:
                    cv2.putText(frame, f"{animal}: {count} detections", 
                               (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                    y_pos += 20
            
            # Display frame
            cv2.imshow('🐾 Animal Detection', frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("👋 Quitting...")
                break
            elif key == ord('s'):
                # Save current frame
                save_count += 1
                filename = f"detection_frame_{save_count}.jpg"
                cv2.imwrite(filename, frame)
                print(f"💾 Saved frame as {filename}")
    
    except KeyboardInterrupt:
        print("\n👋 Detection stopped by user")
    
    except Exception as e:
        print(f"❌ Error during detection: {e}")
    
    finally:
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        
        # Print summary
        print("\n📊 Detection Summary:")
        if detection_count:
            for animal, count in detection_count.items():
                if count > 0:
                    print(f"  {animal}: {count} detections")
        else:
            print("  No animals detected")
        
        print("✅ Camera released, windows closed")

def main():
    """Main function with error handling"""
    try:
        # Check dependencies
        required_modules = ['torch', 'cv2', 'PIL', 'numpy']
        missing_modules = []
        
        for module in required_modules:
            try:
                __import__(module)
            except ImportError:
                missing_modules.append(module)
        
        if missing_modules:
            print(f"❌ Missing required modules: {', '.join(missing_modules)}")
            print("Install with: pip3 install torch torchvision opencv-python pillow numpy")
            return
        
        # Check if model file exists
        if not os.path.exists('animal_detector_2.py'):
            print("❌ animal_detector_2.py not found!")
            print("Make sure you're in the correct directory with all project files.")
            return
        
        print("🚀 Starting Animal Detection...")
        start_detection()
        
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        print("Please check your setup and try again.")

if __name__ == "__main__":
    main()