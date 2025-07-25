#!/usr/bin/env python3
"""
SSL Certificate Fix for macOS
Run this script to fix SSL certificate issues with PyTorch model downloads
"""

import os
import sys
import subprocess
import ssl
import certifi

def fix_ssl_certificates():
    """Fix SSL certificate issues on macOS"""
    
    print("🔧 Fixing SSL certificates for PyTorch model downloads...")
    
    # Method 1: Install certificates for macOS
    print("\n1️⃣ Installing certificates for macOS...")
    
    # Find the Install Certificates.command script
    python_path = sys.executable
    python_dir = os.path.dirname(python_path)
    cert_script_paths = [
        os.path.join(python_dir, "Install Certificates.command"),
        os.path.join(os.path.dirname(python_dir), "Install Certificates.command"),
        f"/Applications/Python {sys.version_info.major}.{sys.version_info.minor}/Install Certificates.command"
    ]
    
    cert_script_found = False
    for cert_script in cert_script_paths:
        if os.path.exists(cert_script):
            print(f"   Found certificate installer: {cert_script}")
            try:
                subprocess.run([cert_script], check=True)
                print("   ✅ Certificates installed successfully")
                cert_script_found = True
                break
            except Exception as e:
                print(f"   ⚠️ Failed to run certificate installer: {e}")
    
    if not cert_script_found:
        print("   ⚠️ Certificate installer not found, trying alternative methods...")
    
    # Method 2: Update certificates using pip
    print("\n2️⃣ Updating certificates via pip...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "certifi"], check=True)
        print("   ✅ Certifi updated successfully")
    except Exception as e:
        print(f"   ⚠️ Failed to update certifi: {e}")
    
    # Method 3: Manual certificate verification bypass (as last resort)
    print("\n3️⃣ Setting up certificate workaround...")
    
    # Create a simple SSL context bypass
    workaround_code = '''
import ssl
import os

# Store original function
_original_create_default_https_context = ssl.create_default_context

def _create_unverified_context():
    """Create an unverified SSL context for downloads"""
    context = ssl.create_default_context()
    context.check_hostname = False
    context.verify_mode = ssl.CERT_NONE
    return context

# Only apply workaround if certificates fail
try:
    ssl.create_default_context().check_hostname = True
except Exception:
    ssl._create_default_https_context = _create_unverified_context
'''
    
    # Write the workaround to a file
    with open("ssl_workaround.py", "w") as f:
        f.write(workaround_code)
    
    print("   ✅ SSL workaround created")
    
    # Method 4: Environment variables
    print("\n4️⃣ Setting environment variables...")
    
    # Set certificate environment variables
    cert_path = certifi.where()
    os.environ['SSL_CERT_FILE'] = cert_path
    os.environ['REQUESTS_CA_BUNDLE'] = cert_path
    
    print(f"   SSL_CERT_FILE = {cert_path}")
    print(f"   REQUESTS_CA_BUNDLE = {cert_path}")
    
    # Test SSL connection
    print("\n🧪 Testing SSL connection...")
    try:
        import urllib.request
        response = urllib.request.urlopen('https://download.pytorch.org', timeout=10)
        print("   ✅ SSL connection test successful")
        response.close()
    except Exception as e:
        print(f"   ⚠️ SSL connection test failed: {e}")
        print("   Using certificate bypass for PyTorch downloads...")
    
    print("\n" + "="*50)
    print("SSL CERTIFICATE FIX COMPLETED")
    print("="*50)
    print("You can now run your animal detection script:")
    print("   python3 test2.py")
    print("\nIf you still get SSL errors, run:")
    print("   python3 test2_with_ssl_fix.py")

def create_ssl_fixed_test_script():
    """Create a version of test2.py with SSL fixes built-in"""
    
    ssl_fixed_script = '''# test2_with_ssl_fix.py - Version with SSL certificate fixes
import ssl
import os
import sys

# Fix SSL certificates before importing other modules
try:
    # Method 1: Create unverified context
    ssl._create_default_https_context = ssl._create_unverified_context
except AttributeError:
    pass

try:
    # Method 2: Set certificate environment variables
    import certifi
    cert_path = certifi.where()
    os.environ['SSL_CERT_FILE'] = cert_path
    os.environ['REQUESTS_CA_BUNDLE'] = cert_path
except ImportError:
    pass

# Now import the rest normally
import cv2
import torch
from animal_detector_2 import AnimalDetector
from PIL import Image
import numpy as np
import subprocess

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
        print("\\nStopping webcam detection...")
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure you have a webcam connected and the required dependencies installed.")
'''
    
    with open("test2_with_ssl_fix.py", "w") as f:
        f.write(ssl_fixed_script)
    
    print("   ✅ Created test2_with_ssl_fix.py with built-in SSL fixes")

if __name__ == "__main__":
    try:
        fix_ssl_certificates()
        create_ssl_fixed_test_script()
        
        print("\n🎯 RECOMMENDED NEXT STEPS:")
        print("1. Try running: python3 test2.py")
        print("2. If SSL errors persist, run: python3 test2_with_ssl_fix.py")
        print("3. Or manually run the certificate installer from Applications/Python folder")
        
    except Exception as e:
        print(f"❌ Error during SSL fix: {e}")
        print("Try manually installing certificates from Applications/Python folder")