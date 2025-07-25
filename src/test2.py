# realtime_detection.py

import cv2
import torch
from animal_detector_2 import AnimalDetector
from PIL import Image
import numpy as np
import subprocess
import os
import sys

def start_webcam_detection(model_path, confidence_threshold=0.5, trigger_threshold=5):
    # Initialize detector
    detector = AnimalDetector(num_classes=8)  # 8 animal categories
    
    # Load model if it exists
    if os.path.exists(model_path):
        detector.load_model(model_path)
        print(f"Loaded model from {model_path}")
    else:
        print(f"Model file {model_path} not found! Using pretrained model without animal-specific training.")
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)  # Use 0 for default webcam
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
        
    cap.set(cv2.CAP_PROP_FPS, 10)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("Starting webcam detection... Press 'q' to quit")
    
    # Object tracking dictionary: {object_label: consecutive_frame_count}
    object_frame_count = {}
    
    frame_count = 0
    
    while True:
        # Read frame
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        
        frame_count += 1
        
        # Process every 3rd frame to improve performance
        if frame_count % 3 != 0:
            cv2.imshow('Animal Detection', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue
            
        # Convert BGR (OpenCV) to RGB (PIL)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        
        try:
            # Get predictions
            predictions = detector.predict(pil_image)

            # Extract information from predictions
            boxes = predictions['boxes']
            labels = predictions['labels']
            scores = predictions['scores']
            
            # Track objects
            current_objects = []
            for label, score in zip(labels, scores):
                if score >= confidence_threshold:  # Filter by confidence
                    label_idx = label.item()
                    if 0 <= label_idx < len(detector.category_names):
                        object_name = detector.category_names[label_idx]
                        current_objects.append(object_name)
                        # Update frame count for each object
                        object_frame_count[object_name] = object_frame_count.get(object_name, 0) + 1
                        
                        # Trigger action if object has been detected for enough consecutive frames
                        if object_frame_count[object_name] >= trigger_threshold:
                            print(f"Triggering action for object: {object_name}")
                            try:
                                # Try to call the notification script
                                if os.path.exists("process_object.py"):
                                    subprocess.Popen([sys.executable, "process_object.py", object_name])
                                else:
                                    print(f"process_object.py not found, detected: {object_name}")
                            except Exception as e:
                                print(f"Error calling process_object.py: {e}")
                            object_frame_count[object_name] = 0  # Reset count after triggering
            
            # Reset counts for objects no longer in frame
            for obj in list(object_frame_count.keys()):
                if obj not in current_objects:
                    object_frame_count[obj] = 0
            
            # Draw predictions
            result_pil = detector.draw_boxes(pil_image, predictions, confidence_threshold)
            
            # Convert back to OpenCV format for display
            result_frame = cv2.cvtColor(np.array(result_pil), cv2.COLOR_RGB2BGR)
            
        except Exception as e:
            print(f"Error during prediction: {e}")
            result_frame = frame
        
        # Add detection info
        y_offset = 30
        for obj, count in object_frame_count.items():
            if count > 0:
                text = f"{obj}: {count}/{trigger_threshold}"
                cv2.putText(result_frame, text, (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                y_offset += 25
        
        # Add instructions
        cv2.putText(result_frame, "Press 'q' to quit", (10, result_frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Display the result
        cv2.imshow('Animal Detection', result_frame)
        
        # Break loop on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        # First check if model exists, if not, use a default name
        model_path = 'animals_trained.pth'
        if not os.path.exists(model_path):
            # Try other possible model names
            possible_models = ['animals2.pth', 'animals3.pth', 'model.pth']
            for model in possible_models:
                if os.path.exists(model):
                    model_path = model
                    break
        
        start_webcam_detection(
            model_path=model_path,
            confidence_threshold=0.3,  # Lower threshold for better detection
            trigger_threshold=10  # Number of consecutive frames required to trigger action
        )
    except KeyboardInterrupt:
        print("\nStopping webcam detection...")
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure you have a webcam connected and the required dependencies installed.")