#!/usr/bin/env python3
"""
Mac-optimized Animal Detection System Runner
Integrates your trained model with webcam detection, API server, and dashboard
Modified to send Pushbullet notifications when humans are detected
"""

import os
import sys
import subprocess
import threading
import time
import json
import requests
from datetime import datetime
import cv2 
import torch
import numpy as np
from PIL import Image

# Import your detector class
try:
    from enhanced_animal_detector import CompleteEnhancedDetector
except ImportError:
    print("Error: enhanced_animal_detector.py not found in current directory")
    sys.exit(1)

# Pushbullet integration
try:
    from pushbullet import Pushbullet
    PUSHBULLET_AVAILABLE = True
except ImportError:
    print("Warning: pushbullet.py not installed. Run: pip install pushbullet.py")
    PUSHBULLET_AVAILABLE = False

# Add your Pushbullet API key here
PUSHBULLET_API_KEY = "o.0fO4N5ybzVV9k4ESDT2oRt06BexvYPKu"

class MacAnimalDetectionRunner:
    def __init__(self, model_path='checkpoints/best_model.pth'):
        self.model_path = model_path
        self.detector = None
        self.server_process = None
        self.detection_thread = None
        self.running = False
        self.api_url = 'http://localhost:5001'
        self.output_dir = 'detections'
        
        # Pushbullet setup
        self.pushbullet = None
        self.last_human_notification = 0
        self.notification_cooldown = 30  # seconds between human notifications
        
        if PUSHBULLET_AVAILABLE and PUSHBULLET_API_KEY != "YOUR_API_KEY_HERE":
            try:
                self.pushbullet = Pushbullet(PUSHBULLET_API_KEY)
                print("✓ Pushbullet connected for human detection alerts")
            except Exception as e:
                print(f"Warning: Could not connect to Pushbullet: {e}")
        else:
            print("Warning: Pushbullet not configured. Human detection alerts disabled.")
        
        # Create output directory for saving detection images
        os.makedirs(self.output_dir, exist_ok=True)
        
    def is_human_detection(self, class_name):
        """Check if detected class is a human/person"""
        human_classes = ['person', 'human', 'people']
        return class_name.lower() in human_classes
    
    def send_human_alert(self, confidence, image_path=None):
        """Send Pushbullet notification for human detection"""
        if not self.pushbullet:
            return False
            
        current_time = time.time()
        
        # Check cooldown period
        if current_time - self.last_human_notification < self.notification_cooldown:
            return False
        
        try:
            title = "🚨 HUMAN DETECTED"
            message = f"Human detected with {confidence:.1%} confidence\nTime: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            
            # Send text notification
            self.pushbullet.push_note(title, message)
            
            # Send image if available
            if image_path and os.path.exists(image_path):
                try:
                    with open(image_path, "rb") as pic:
                        file_data = self.pushbullet.upload_file(pic, "human_detection.jpg")
                    self.pushbullet.push_file(**file_data)
                except Exception as e:
                    print(f"Could not send detection image: {e}")
            
            self.last_human_notification = current_time
            print(f"✓ Human detection alert sent via Pushbullet")
            return True
            
        except Exception as e:
            print(f"✗ Failed to send Pushbullet notification: {e}")
            return False
        
    def check_dependencies(self):
        """Check if all required dependencies are installed"""
        print("Checking dependencies...")
        
        # Map package names to their import names
        package_imports = {
            'torch': 'torch',
            'torchvision': 'torchvision', 
            'opencv-python': 'cv2',
            'pillow': 'PIL',
            'numpy': 'numpy',
            'flask': 'flask',
            'flask-cors': 'flask_cors',
            'requests': 'requests'
        }
        
        missing_packages = []
        for package, import_name in package_imports.items():
            try:
                __import__(import_name)
                print(f"✓ {package}")
            except ImportError:
                missing_packages.append(package)
                print(f"✗ {package}")
        
        if missing_packages:
            print(f"\nMissing packages: {', '.join(missing_packages)}")
            print("Install with: pip install " + " ".join(missing_packages))
            return False
        
        return True
    def check_camera_permissions(self):
        """Check camera permissions on Mac"""
        print("Checking camera access...")
        try:
            cap = cv2.VideoCapture(0)
            if cap.isOpened():
                ret, frame = cap.read()
                cap.release()
                if ret:
                    print("✓ Camera access granted")
                    return True
                else:
                    print("✗ Camera opened but couldn't read frame")
                    return False
            else:
                print("✗ Could not open camera")
                print("On Mac, you may need to:")
                print("1. Go to System Preferences > Security & Privacy > Camera")
                print("2. Allow Terminal/Python to access camera")
                return False
        except Exception as e:
            print(f"✗ Camera check failed: {e}")
            return False
    
    def load_model(self):
        """Load the trained model"""
        print(f"Loading trained model from {self.model_path}...")
        
        # Check if model file exists
        if not os.path.exists(self.model_path):
            print(f"✗ Model file not found: {self.model_path}")
            
            # Look for alternative model files
            potential_paths = [
                'best_model.pth',
                'checkpoints/best_model.pth',
                'model/best_model.pth',
                'animals_trained.pth'
            ]
            
            print("Looking for alternative model files...")
            for path in potential_paths:
                if os.path.exists(path):
                    print(f"Found alternative model: {path}")
                    self.model_path = path
                    break
            else:
                print("No model files found. Please ensure you have:")
                print("1. Trained a model using enhanced_animal_detector.py")
                print("2. Or have a valid .pth file in the correct location")
                return False
        
        # Initialize detector
        self.detector = CompleteEnhancedDetector()
        
        # Load model
        if self.detector.load_model(self.model_path):
            print(f"✓ Model loaded successfully from {self.model_path}")
            return True
        else:
            print(f"✗ Failed to load model from {self.model_path}")
            return False
    
    def start_server(self):
        """Start the Flask API server"""
        print("Starting API server...")
        
        # Check if server script exists
        server_script = 'torch server.py'
        if not os.path.exists(server_script):
            print(f"✗ Server script not found: {server_script}")
            return False
        
        try:
            # Start server in background
            self.server_process = subprocess.Popen([
                sys.executable, server_script
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            # Wait for server to start
            print("Waiting for server to start...")
            for i in range(10):
                try:
                    response = requests.get(f'{self.api_url}/api/detections', timeout=2)
                    if response.status_code == 200:
                        print("✓ API server started successfully")
                        return True
                except requests.RequestException:
                    time.sleep(1)
            
            print("✗ Server failed to start within 10 seconds")
            return False
            
        except Exception as e:
            print(f"✗ Error starting server: {e}")
            return False
    
    def send_detection_to_api(self, animal, confidence, image_path):
        """Send detection data to API server"""
        try:
            detection_data = {
                'timestamp': time.time(),
                'animal': animal,
                'location': 'Webcam',
                'confidence': float(confidence),
                'image_path': image_path
            }
            
            response = requests.post(
                f'{self.api_url}/api/add_detection',
                json=detection_data,
                timeout=2
            )
            
            if response.status_code == 200:
                print(f"✓ Detection sent to API: {animal} ({confidence:.2f})")
            else:
                print(f"✗ API error: {response.status_code}")
                
        except Exception as e:
            print(f"✗ Failed to send detection to API: {e}")
    
    def run_webcam_detection(self, confidence_threshold=0.5, save_detections=True):
        """Run real-time webcam detection"""
        print("Starting webcam detection...")
        print("Press 'q' to quit, 's' to save current frame")
        if self.pushbullet:
            print("🚨 Human detection alerts enabled via Pushbullet")
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("✗ Could not open webcam")
            return False
        
        # Set camera properties for better performance on Mac
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        detection_count = 0
        human_detection_count = 0
        last_detection_time = 0
        detection_cooldown = 2.0  # seconds between API calls
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("✗ Failed to read frame from webcam")
                break
            
            # Save frame temporarily for detection
            temp_path = os.path.join(self.output_dir, 'temp_frame.jpg')
            cv2.imwrite(temp_path, frame)
            
            try:
                # Run detection
                boxes, scores, labels, _ = self.detector.detect_image(
                    temp_path, confidence_threshold
                )
                
                # Draw detections on frame
                for box, score, label in zip(boxes, scores, labels):
                    if label < len(self.detector.class_names):
                        class_name = self.detector.class_names[label]
                        
                        # Check if this is a human detection
                        is_human = self.is_human_detection(class_name)
                        
                        # Use red color for humans, original colors for others
                        if is_human:
                            color = (0, 0, 255)  # Red for humans
                            human_detection_count += 1
                            
                            # Send Pushbullet alert for human detection
                            if save_detections:
                                # Save human detection image
                                human_filename = f"human_detection_{human_detection_count:04d}_{score:.2f}.jpg"
                                human_path = os.path.join(self.output_dir, human_filename)
                                
                                # Add timestamp overlay for human detections
                                human_frame = frame.copy()
                                timestamp_text = f"HUMAN DETECTED: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                                cv2.putText(human_frame, timestamp_text, (10, 30),
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                                cv2.imwrite(human_path, human_frame)
                                
                                # Send Pushbullet notification
                                self.send_human_alert(score, human_path)
                        else:
                            color = self.detector.colors.get(class_name, (255, 255, 255))
                        
                        # Draw bounding box
                        x1, y1, x2, y2 = map(int, box)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3 if is_human else 2)
                        
                        # Draw label
                        label_text = f"{'HUMAN' if is_human else class_name}: {score:.2f}"
                        cv2.putText(frame, label_text, (x1, y1 - 10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.8 if is_human else 0.7, color, 2)
                        
                        # Send to API (with cooldown)
                        current_time = time.time()
                        if (current_time - last_detection_time) > detection_cooldown:
                            if save_detections:
                                # Save detection image
                                detection_filename = f"detection_{detection_count:04d}_{class_name}_{score:.2f}.jpg"
                                detection_path = os.path.join(self.output_dir, detection_filename)
                                cv2.imwrite(detection_path, frame)
                                
                                # Send to API
                                self.send_detection_to_api(class_name, score, detection_path)
                                
                                detection_count += 1
                                last_detection_time = current_time
                
                # Add status text
                status_text = f"Detections: {detection_count} | Humans: {human_detection_count} | Press 'q' to quit"
                cv2.putText(frame, status_text, (10, frame.shape[0] - 20),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
            except Exception as e:
                print(f"Detection error: {e}")
                # Add error text to frame
                cv2.putText(frame, f"Detection Error: {str(e)[:50]}", (10, 60),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            
            # Display frame
            cv2.imshow('Animal Detection - Mac', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Save current frame
                save_path = os.path.join(self.output_dir, f'saved_frame_{int(time.time())}.jpg')
                cv2.imwrite(save_path, frame)
                print(f"Frame saved: {save_path}")
        
        cap.release()
        cv2.destroyAllWindows()
        
        # Clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        print(f"Detection completed. Total detections: {detection_count}")
        print(f"Human detections: {human_detection_count}")
        return True
    
    def open_dashboard(self):
        """Open the web dashboard"""
        dashboard_url = "http://localhost:3000"  # Assuming React dashboard runs on port 3000
        
        try:
            # Try to open in default browser
            import webbrowser
            webbrowser.open(dashboard_url)
            print(f"✓ Dashboard opened in browser: {dashboard_url}")
        except Exception as e:
            print(f"Could not open dashboard automatically: {e}")
            print(f"Please open manually: {dashboard_url}")
    
    def run_complete_system(self):
        """Run the complete detection system"""
        print("=" * 60)
        print("MAC ANIMAL DETECTION SYSTEM")
        print("=" * 60)
        
        # Step 1: Check dependencies
        if not self.check_dependencies():
            return False
        
        # Step 2: Check camera permissions
        if not self.check_camera_permissions():
            return False
        
        # Step 3: Load model
        if not self.load_model():
            return False
        
        # Step 4: Start API server
        if not self.start_server():
            print("Warning: API server failed to start. Continuing with webcam detection only.")
        
        # Step 5: Run webcam detection
        try:
            self.run_webcam_detection()
        except KeyboardInterrupt:
            print("\nDetection stopped by user")
        
        # Cleanup
        self.cleanup()
        
        return True
    
    def cleanup(self):
        """Clean up processes and resources"""
        print("Cleaning up...")
        
        if self.server_process:
            try:
                self.server_process.terminate()
                self.server_process.wait(timeout=5)
                print("✓ Server process terminated")
            except Exception as e:
                print(f"Warning: Could not terminate server process: {e}")
    
    def run_test_detection(self, image_path):
        """Test detection on a single image"""
        if not self.load_model():
            return False
        
        if not os.path.exists(image_path):
            print(f"Test image not found: {image_path}")
            return False
        
        print(f"Testing detection on: {image_path}")
        
        try:
            result_img, boxes, scores, labels = self.detector.detect_and_display(
                image_path, 
                confidence_threshold=0.5,
                save_path=f"test_result_{os.path.basename(image_path)}"
            )
            
            # Check if any humans were detected in the test image
            for score, label in zip(scores, labels):
                if label < len(self.detector.class_names):
                    class_name = self.detector.class_names[label]
                    if self.is_human_detection(class_name):
                        print(f"Human detected in test image with confidence: {score:.2f}")
                        # Send test notification
                        self.send_human_alert(score, f"test_result_{os.path.basename(image_path)}")
            
            print("Test completed successfully!")
            return True
            
        except Exception as e:
            print(f"Test failed: {e}")
            return False

def main():
    """Main function with menu options"""
    runner = MacAnimalDetectionRunner()
    
    print("WildEye Animal Detection System")
    if runner.pushbullet:
        print("🚨 Human detection alerts: ENABLED")
    else:
        print("⚠️ Human detection alerts: DISABLED (Configure Pushbullet API key)")
    
    print("1. Run complete system (webcam + API + dashboard)")
    print("2. Test detection on image")
    print("3. Webcam detection only")
    print("4. Start API server only")
    
    try:
        choice = input("Choose option (1-4): ").strip()
        
        if choice == '1':
            runner.run_complete_system()
        elif choice == '2':
            image_path = input("Enter image path: ").strip()
            runner.run_test_detection(image_path)
        elif choice == '3':
            if runner.load_model():
                runner.run_webcam_detection()
        elif choice == '4':
            runner.start_server()
            print("Server running. Press Ctrl+C to stop.")
            while True:
                time.sleep(1)
        else:
            print("Invalid choice")
            
    except KeyboardInterrupt:
        print("\nExiting...")
        runner.cleanup()
    except Exception as e:
        print(f"Error: {e}")
        runner.cleanup()

if __name__ == "__main__":
    main()