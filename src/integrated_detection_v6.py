#!/usr/bin/env python3
"""
Enhanced Unified Animal & Gunshot Detection System
Combines the robust visual detection from run3.py with audio gunshot detection
Maintains all original features while adding audio threat detection
"""

import os
import sys
import subprocess
import threading
import time
import json
import queue
from datetime import datetime
import cv2
import numpy as np
from PIL import Image
import geocoder

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings

# Audio detection imports
try:
    import sounddevice as sd
    import soundfile as sf
    import librosa
    import tensorflow as tf
    from keras.models import load_model
    AUDIO_DETECTION_AVAILABLE = True
except ImportError:
    print("Warning: Audio detection libraries not installed.")
    print("For audio detection, install: pip install sounddevice soundfile librosa tensorflow keras")
    AUDIO_DETECTION_AVAILABLE = False

# Pushbullet and other imports
import requests
try:
    from pushbullet import Pushbullet
    PUSHBULLET_AVAILABLE = True
except ImportError:
    print("Warning: pushbullet.py not installed. Run: pip install pushbullet.py")
    PUSHBULLET_AVAILABLE = False

# Visual detection imports
try:
    from enhanced_animal_detector import CompleteEnhancedDetector
    VISUAL_DETECTOR_AVAILABLE = True
except ImportError:
    print("Error: enhanced_animal_detector.py not found in current directory")
    VISUAL_DETECTOR_AVAILABLE = False

class EnhancedUnifiedDetectionSystem:
    def __init__(self, visual_model_path='checkpoints/best_model.pth', audio_model_path='gunshot_full.h5'):
        # Configuration
        self.config = {
            'pushbullet_api_key': "o.0fO4N5ybzVV9k4ESDT2oRt06BexvYPKu",
            'visual_model_path': visual_model_path,
            'audio_model_path': audio_model_path,
            'visual_confidence_threshold': 0.5,
            'audio_confidence_threshold': 0.5,
            'notification_cooldown': 30,  # seconds
            'audio_duration': 2,  # seconds per recording
            'audio_sample_rate': 22050,
            'output_dir': 'unified_detections',
            'visual_output_dir': 'visual_detections',
            'audio_output_dir': 'audio_detections'
        }
        
        # Initialize components
        self.pushbullet = None
        self.visual_detector = None
        self.audio_model = None
        self.server_process = None
        
        # Threading controls
        self.running = False
        self.visual_thread = None
        self.audio_thread = None
        self.detection_history = []
        
        # API configuration
        self.api_url = 'http://localhost:5001'
        
        # Notification tracking
        self.last_notifications = {
            'human': 0,
            'gunshot': 0,
            'animal': 0
        }
        
        # Audio classes (from your original audio detection)
        self.audio_classes = [
            'Background noise', 'Background noise', 'Background noise', 
            'Background noise', 'gunshot', 'Background noise', 'gunshot', 
            'gunshot', 'Background noise', 'Background noise'
        ]
        
        # Detection counters
        self.detection_counts = {
            'visual': 0,
            'human': 0,
            'animal': 0,
            'gunshot': 0
        }
        
        # Setup directories
        self.setup_directories()
        
        # Initialize systems
        self.initialize_pushbullet()
        self.initialize_visual_detector()
        self.initialize_audio_detector()
    
    def setup_directories(self):
        """Create necessary directories"""
        for dir_path in [self.config['output_dir'], 
                        self.config['visual_output_dir'], 
                        self.config['audio_output_dir']]:
            os.makedirs(dir_path, exist_ok=True)
    
    def get_gps_location(self):
        """Get current GPS location"""
        try:
            g = geocoder.ip('me')
            if g.ok:
                return {
                    'latitude': g.latlng[0],
                    'longitude': g.latlng[1],
                    'address': g.address,
                    'city': g.city,
                    'country': g.country
                }
            else:
                return None
        except Exception as e:
            print(f"GPS location error: {e}")
            return None
    
    def initialize_pushbullet(self):
        """Initialize Pushbullet for notifications"""
        if not PUSHBULLET_AVAILABLE:
            print("Warning: Pushbullet not available")
            return False
        
        if self.config['pushbullet_api_key'] == "":
            print("Warning: Pushbullet API key not configured")
            return False
        
        try:
            self.pushbullet = Pushbullet(self.config['pushbullet_api_key'])
            print("✓ Pushbullet connected for threat detection alerts")
            return True
        except Exception as e:
            print(f"Warning: Could not connect to Pushbullet: {e}")
            return False
    
    def initialize_visual_detector(self):
        """Initialize visual detection system"""
        if not VISUAL_DETECTOR_AVAILABLE:
            print("✗ Visual detector not available")
            return False
        
        try:
            self.visual_detector = CompleteEnhancedDetector()
            if os.path.exists(self.config['visual_model_path']):
                if self.visual_detector.load_model(self.config['visual_model_path']):
                    print("✓ Visual detector initialized successfully")
                    return True
            
            # Look for alternative model files (from original run3.py logic)
            potential_paths = [
                'best_model.pth',
                'checkpoints/best_model.pth',
                'model/best_model.pth',
                'animals_trained.pth'
            ]
            
            print("Looking for alternative visual model files...")
            for path in potential_paths:
                if os.path.exists(path):
                    print(f"Found alternative model: {path}")
                    self.config['visual_model_path'] = path
                    if self.visual_detector.load_model(path):
                        print("✓ Visual detector initialized with alternative model")
                        return True
            
            print("✗ No visual model files found")
            return False
            
        except Exception as e:
            print(f"✗ Visual detector initialization failed: {e}")
            return False
    
    def initialize_audio_detector(self):
        """Initialize audio detection system"""
        if not AUDIO_DETECTION_AVAILABLE:
            print("Warning: Audio detection not available - missing libraries")
            return False
        
        try:
            if os.path.exists(self.config['audio_model_path']):
                self.audio_model = load_model(self.config['audio_model_path'])
                print("✓ Audio detector initialized successfully")
                return True
            else:
                print(f"Warning: Audio model not found: {self.config['audio_model_path']}")
                return False
        except Exception as e:
            print(f"Warning: Audio detector initialization failed: {e}")
            return False
    
    def is_human_detection(self, class_name):
        """Check if detected class is a human/person"""
        human_classes = ['person', 'human', 'people']
        return class_name.lower() in human_classes
    
    def send_unified_alert(self, threat_type, details):
        """Send unified notification via Pushbullet"""
        if not self.pushbullet:
            print(f"Warning: Notification not sent - Pushbullet unavailable: {threat_type}")
            return False
        
        # Skip animal notifications
        if threat_type == 'animal':
            return False
        
        current_time = time.time()
        
        # Check cooldown
        if current_time - self.last_notifications[threat_type] < self.config['notification_cooldown']:
            return False
        
        try:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            # Get GPS location for human detections
            location_info = ""
            if threat_type == 'human':
                gps_data = self.get_gps_location()
                if gps_data:
                    location_info = f"\nGPS: {gps_data['latitude']:.6f}, {gps_data['longitude']:.6f}\nAddress: {gps_data['address']}"
                    # Display GPS on terminal
                    print(f"🌍 GPS Location: {gps_data['latitude']:.6f}, {gps_data['longitude']:.6f}")
                    print(f"📍 Address: {gps_data['address']}")
            
            # Prepare notification content based on threat type
            if threat_type == 'human':
                title = "🚨 HUMAN DETECTED"
                message = f"Human detected with {details['confidence']:.1%} confidence\nTime: {timestamp}\nLocation: Visual System{location_info}"
            elif threat_type == 'gunshot':
                title = "🔫 GUNSHOT DETECTED"
                message = f"Gunshot detected with {details['confidence']:.1%} confidence\nTime: {timestamp}\nLocation: Audio System"
            
            # Send text notification
            self.pushbullet.push_note(title, message)
            
            # Send file if available
            if 'file_path' in details and details['file_path'] and os.path.exists(details['file_path']):
                try:
                    with open(details['file_path'], "rb") as file:
                        if threat_type == 'gunshot':
                            file_data = self.pushbullet.upload_file(file, "gunshot_audio.wav")
                        else:
                            file_data = self.pushbullet.upload_file(file, f"{threat_type}_detection.jpg")
                        self.pushbullet.push_file(**file_data)
                except Exception as e:
                    print(f"Could not send detection file: {e}")
            
            self.last_notifications[threat_type] = current_time
            print(f"✓ {threat_type.upper()} detection alert sent via Pushbullet")
            
            # Log detection
            self.log_detection(threat_type, details, timestamp)
            
            return True
        
        except Exception as e:
            print(f"✗ Failed to send {threat_type} notification: {e}")
            return False
    
    def log_detection(self, threat_type, details, timestamp):
        """Log detection to history"""
        detection_entry = {
            'timestamp': timestamp,
            'type': threat_type,
            'details': details
        }
        self.detection_history.append(detection_entry)
        
        # Keep only last 100 detections
        if len(self.detection_history) > 100:
            self.detection_history.pop(0)
        
        # Save to file
        try:
            log_file = os.path.join(self.config['output_dir'], 'detection_log.json')
            with open(log_file, 'w') as f:
                json.dump(self.detection_history, f, indent=2)
        except Exception as e:
            print(f"Could not save detection log: {e}")
    
    def send_detection_to_api(self, animal, confidence, image_path):
        """Send detection data to API server (from original run3.py)"""
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
    
    def audio_detection_loop(self):
        """Audio detection loop running in separate thread"""
        if not self.audio_model:
            print("Audio detection not available")
            return
        
        print("🎙️ Starting audio detection...")
        
        while self.running:
            try:
                # Record audio
                audio = sd.rec(
                    int(self.config['audio_duration'] * self.config['audio_sample_rate']), 
                    samplerate=self.config['audio_sample_rate'], 
                    channels=1
                )
                sd.wait()
                audio = audio.flatten()
                
                # Extract features
                mels = librosa.feature.melspectrogram(y=audio, sr=self.config['audio_sample_rate'])
                mels_db = np.mean(mels.T, axis=0)
                features = np.expand_dims(mels_db, axis=0)
                
                # Predict
                prediction = self.audio_model.predict(features, verbose=0)
                predicted_class = self.audio_classes[np.argmax(prediction)]
                confidence = float(prediction[0][np.argmax(prediction)])
                
                # Check for gunshot
                if predicted_class == 'gunshot' and confidence > self.config['audio_confidence_threshold']:
                    self.detection_counts['gunshot'] += 1
                    
                    # Save audio file
                    audio_filename = f"gunshot_{self.detection_counts['gunshot']:04d}_{confidence:.2f}.wav"
                    audio_path = os.path.join(self.config['audio_output_dir'], audio_filename)
                    sf.write(audio_path, audio, self.config['audio_sample_rate'])
                    
                    # Send notification
                    self.send_unified_alert('gunshot', {
                        'confidence': confidence,
                        'file_path': audio_path
                    })
                    
                    print(f"🔫 GUNSHOT DETECTED: {confidence:.2f} confidence")
                
            except Exception as e:
                print(f"Audio detection error: {e}")
                time.sleep(1)
        
        print("🎙️ Audio detection stopped")
    
    def run_unified_webcam_detection(self, save_detections=True):
        """Enhanced webcam detection with audio integration"""
        if not self.visual_detector:
            print("✗ Visual detector not available")
            return False
        
        print("Starting unified detection system...")
        print("🎥 Visual: Animal & Human Detection")
        if self.audio_model:
            print("🎙️ Audio: Gunshot Detection")
        if self.pushbullet:
            print("📱 Notifications: Pushbullet Alerts")
        print("Press 'q' to quit, 's' to save current frame")
        
        # Start audio detection thread
        if self.audio_model:
            self.running = True
            self.audio_thread = threading.Thread(target=self.audio_detection_loop)
            self.audio_thread.daemon = True
            self.audio_thread.start()
        
        # Initialize camera
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("✗ Could not open webcam")
            return False
        
        # Set camera properties (from original run3.py)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        last_detection_time = 0
        detection_cooldown = 2.0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("✗ Failed to read frame from webcam")
                    break
                
                # Save frame temporarily for detection
                temp_path = os.path.join(self.config['visual_output_dir'], 'temp_frame.jpg')
                cv2.imwrite(temp_path, frame)
                
                try:
                    # Run visual detection
                    boxes, scores, labels, _ = self.visual_detector.detect_image(
                        temp_path, self.config['visual_confidence_threshold']
                    )
                    
                    current_time = time.time()
                    
                    # Process detections
                    for box, score, label in zip(boxes, scores, labels):
                        if label < len(self.visual_detector.class_names):
                            class_name = self.visual_detector.class_names[label]
                            
                            # Check if this is a human detection
                            is_human = self.is_human_detection(class_name)
                            
                            # Use red color for humans, original colors for others
                            if is_human:
                                color = (0, 0, 255)  # Red for humans
                                self.detection_counts['human'] += 1
                            else:
                                color = self.visual_detector.colors.get(class_name, (255, 255, 255))
                                self.detection_counts['animal'] += 1
                            
                            # Draw bounding box
                            x1, y1, x2, y2 = map(int, box)
                            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3 if is_human else 2)
                            
                            # Draw label
                            label_text = f"{'HUMAN' if is_human else class_name}: {score:.2f}"
                            cv2.putText(frame, label_text, (x1, y1 - 10),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.8 if is_human else 0.7, color, 2)
                            
                            # Send notifications and API calls (with cooldown)
                            if (current_time - last_detection_time) > detection_cooldown:
                                if save_detections:
                                    self.detection_counts['visual'] += 1
                                    
                                    # Save detection image
                                    if is_human:
                                        detection_filename = f"human_detection_{self.detection_counts['human']:04d}_{score:.2f}.jpg"
                                        # Add timestamp overlay for human detections
                                        human_frame = frame.copy()
                                        timestamp_text = f"HUMAN DETECTED: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                                        cv2.putText(human_frame, timestamp_text, (10, 30),
                                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                                        detection_path = os.path.join(self.config['visual_output_dir'], detection_filename)
                                        cv2.imwrite(detection_path, human_frame)
                                        
                                        # Animal detection logged but no notification sent
                                        print(f"Subject detected: {class_name} ({score:.2f})")
                                    else:
                                        detection_filename = f"animal_detection_{self.detection_counts['animal']:04d}_{class_name}_{score:.2f}.jpg"
                                        detection_path = os.path.join(self.config['visual_output_dir'], detection_filename)
                                        cv2.imwrite(detection_path, frame)
                                        
                                        # Send animal alert
                                        self.send_unified_alert('animal', {
                                            'animal_type': class_name,
                                            'confidence': score,
                                            'file_path': detection_path
                                        })
                                    
                                    # Send to API
                                    self.send_detection_to_api(class_name, score, detection_path)
                                    
                                    last_detection_time = current_time
                    
                    # Add comprehensive status overlay
                    status_lines = [
                        f"Visual: {self.detection_counts['visual']} | Humans: {self.detection_counts['human']} | Animals: {self.detection_counts['animal']}",
                        f"Gunshots: {self.detection_counts['gunshot']} | Audio: {'ON' if self.audio_model else 'OFF'} | API: {'ON' if self.server_process else 'OFF'}",
                        "Press 'q' to quit, 's' to save frame"
                    ]
                    
                    for i, line in enumerate(status_lines):
                        cv2.putText(frame, line, (10, frame.shape[0] - 60 + i * 20),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    
                except Exception as e:
                    print(f"Detection error: {e}")
                    cv2.putText(frame, f"Detection Error: {str(e)[:50]}", (10, 60),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                
                # Display frame
                cv2.imshow('Unified Detection System - Enhanced', frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    # Save current frame
                    save_path = os.path.join(self.config['output_dir'], f'saved_frame_{int(time.time())}.jpg')
                    cv2.imwrite(save_path, frame)
                    print(f"Frame saved: {save_path}")
        
        except KeyboardInterrupt:
            print("\nDetection stopped by user")
        
        finally:
            # Cleanup
            self.running = False
            cap.release()
            cv2.destroyAllWindows()
            
            # Wait for audio thread to finish
            if self.audio_thread and self.audio_thread.is_alive():
                self.audio_thread.join(timeout=2)
            
            # Clean up temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)
            
            print(f"Detection completed:")
            print(f"  Total visual detections: {self.detection_counts['visual']}")
            print(f"  Human detections: {self.detection_counts['human']}")
            print(f"  Animal detections: {self.detection_counts['animal']}")
            print(f"  Gunshot detections: {self.detection_counts['gunshot']}")
        
        return True
    
    def check_dependencies(self):
        """Check if all required dependencies are installed"""
        print("Checking dependencies...")
        
        # Visual detection dependencies
        visual_packages = {
            'torch': 'torch',
            'torchvision': 'torchvision', 
            'opencv-python': 'cv2',
            'pillow': 'PIL',
            'numpy': 'numpy',
            'requests': 'requests',
            'geocoder': 'geocoder'
        }
        
        # Audio detection dependencies  
        audio_packages = {
            'sounddevice': 'sounddevice',
            'soundfile': 'soundfile',
            'librosa': 'librosa',
            'tensorflow': 'tensorflow'
        }
        
        missing_packages = []
        
        # Check visual packages
        for package, import_name in visual_packages.items():
            try:
                __import__(import_name)
                print(f"✓ {package}")
            except ImportError:
                missing_packages.append(package)
                print(f"✗ {package}")
        
        # Check audio packages
        for package, import_name in audio_packages.items():
            try:
                __import__(import_name)
                print(f"✓ {package}")
            except ImportError:
                missing_packages.append(package)
                print(f"⚠️ {package} (audio detection will be disabled)")
        
        if missing_packages:
            print(f"\nMissing packages: {', '.join(missing_packages)}")
            print("Install with: pip install " + " ".join(missing_packages))
        
        return len([p for p in visual_packages.keys() if p not in missing_packages]) >= 5
    
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
    
    def start_server(self):
        """Start the Flask API server"""
        print("Starting API server...")
        
        server_script = 'torch server.py'
        if not os.path.exists(server_script):
            print(f"✗ Server script not found: {server_script}")
            return False
        
        try:
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
    
    def run_complete_system(self):
        """Run the complete unified detection system"""
        print("=" * 80)
        print("ENHANCED UNIFIED DETECTION SYSTEM")
        print("🎥 Visual: Animal & Human Detection")
        print("🎙️ Audio: Gunshot Detection")
        print("📱 Notifications: Pushbullet Alerts")
        print("🌐 API: Detection Logging")
        print("=" * 80)
        
        # Step 1: Check dependencies
        if not self.check_dependencies():
            return False
        
        # Step 2: Check camera permissions
        if not self.check_camera_permissions():
            return False
        
        # Step 3: Start API server (optional)
        if not self.start_server():
            print("Warning: API server failed to start. Continuing without API integration.")
        
        # Step 4: Run unified detection
        try:
            self.run_unified_webcam_detection()
        except KeyboardInterrupt:
            print("\nDetection stopped by user")
        
        # Cleanup
        self.cleanup()
        
        return True
    
    def cleanup(self):
        """Clean up processes and resources"""
        print("Cleaning up...")
        
        self.running = False
        
        if self.server_process:
            try:
                self.server_process.terminate()
                self.server_process.wait(timeout=5)
                print("✓ Server process terminated")
            except Exception as e:
                print(f"Warning: Could not terminate server process: {e}")
    
    def run_test_detection(self, image_path):
        """Test visual detection on a single image"""
        if not self.visual_detector:
            print("Visual detector not available")
            return False
        
        if not os.path.exists(image_path):
            print(f"Test image not found: {image_path}")
            return False
        
        print(f"Testing visual detection on: {image_path}")
        
        try:
            result_img, boxes, scores, labels = self.visual_detector.detect_and_display(
                image_path, 
                confidence_threshold=self.config['visual_confidence_threshold'],
                save_path=f"test_result_{os.path.basename(image_path)}"
            )
            
            # Check detections
            for score, label in zip(scores, labels):
                if label < len(self.visual_detector.class_names):
                    class_name = self.visual_detector.class_names[label]
                    if self.is_human_detection(class_name):
                        print(f"Human detected in test image with confidence: {score:.2f}")
                        self.send_unified_alert('human', {
                            'confidence': score,
                            'file_path': f"test_result_{os.path.basename(image_path)}"
                        })
            
            print("Test completed successfully!")
            return True
            
        except Exception as e:
            print(f"Test failed: {e}")
            return False
    
    def run_audio_test(self):
        """Test audio detection"""
        if not self.audio_model:
            print("Audio detection not available")
            return False
        
        print("Testing audio detection - make a sound...")
        
        try:
            # Record test audio
            audio = sd.rec(
                int(self.config['audio_duration'] * self.config['audio_sample_rate']), 
                samplerate=self.config['audio_sample_rate'], 
                channels=1
            )
            sd.wait()
            audio = audio.flatten()
            
            # Process audio
            mels = librosa.feature.melspectrogram(y=audio, sr=self.config['audio_sample_rate'])
            mels_db = np.mean(mels.T, axis=0)
            features = np.expand_dims(mels_db, axis=0)
            
            # Predict
            prediction = self.audio_model.predict(features, verbose=0)
            predicted_class = self.audio_classes[np.argmax(prediction)]
            confidence = float(prediction[0][np.argmax(prediction)])
            
            print(f"Audio test result: {predicted_class} ({confidence:.2f})")
            
            # Save test audio
            test_path = os.path.join(self.config['audio_output_dir'], 'test_audio.wav')
            sf.write(test_path, audio, self.config['audio_sample_rate'])
            
            if predicted_class == 'gunshot':
                print("Gunshot detected in test!")
                self.send_unified_alert('gunshot', {
                    'confidence': confidence,
                    'file_path': test_path
                })
            
            return True
            
        except Exception as e:
            print(f"Audio test failed: {e}")
            return False

def main():
    """Main function with enhanced menu options"""
    system = EnhancedUnifiedDetectionSystem()
    
    print("🔍 Enhanced Unified Detection System")
    print("=" * 50)
    
    # Show system status
    status = []
    if system.visual_detector:
        status.append("Visual Detection: ✓")
    else:
        status.append("Visual Detection: ✗")
        
    if system.audio_model:
        status.append("Audio Detection: ✓")
    else:
        status.append("Audio Detection: ✗")
        
    if system.pushbullet:
        status.append("Pushbullet Alerts: ✓")
    else:
        status.append("Pushbullet Alerts: ✗")
    
    print("\n".join(status))
    print("=" * 50)
    
    while True:
        print("\nChoose an option:")
        print("1. Run Complete Detection System (Visual + Audio)")
        print("2. Run Visual Detection Only")
        print("3. Run Audio Detection Only")
        print("4. Test Visual Detection on Image")
        print("5. Test Audio Detection")
        print("6. View Detection History")
        print("7. System Status")
        print("8. Configuration")
        print("9. Exit")
        
        choice = input("\nEnter your choice (1-9): ").strip()
        
        if choice == '1':
            print("\n🚀 Starting Complete Detection System...")
            system.run_complete_system()
            
        elif choice == '2':
            print("\n🎥 Starting Visual Detection Only...")
            if system.visual_detector:
                system.run_unified_webcam_detection()
            else:
                print("Visual detection not available")
                
        elif choice == '3':
            print("\n🎙️ Starting Audio Detection Only...")
            if system.audio_model:
                system.running = True
                print("Audio detection running... Press Ctrl+C to stop")
                try:
                    system.audio_detection_loop()
                except KeyboardInterrupt:
                    print("\nAudio detection stopped")
                    system.running = False
            else:
                print("Audio detection not available")
                
        elif choice == '4':
            image_path = input("Enter path to test image: ").strip()
            if image_path:
                system.run_test_detection(image_path)
            else:
                print("No image path provided")
                
        elif choice == '5':
            print("\n🔊 Testing Audio Detection...")
            system.run_audio_test()
            
        elif choice == '6':
            print("\n📊 Detection History:")
            if system.detection_history:
                for i, detection in enumerate(system.detection_history[-10:], 1):
                    print(f"{i}. {detection['timestamp']} - {detection['type'].upper()}")
                    if detection['type'] == 'animal':
                        print(f"   Animal: {detection['details'].get('animal_type', 'Unknown')}")
                    print(f"   Confidence: {detection['details'].get('confidence', 0):.2f}")
                    print()
            else:
                print("No detections recorded yet")
                
        elif choice == '7':
            print("\n🔍 System Status:")
            print(f"Visual Detector: {'✓ Ready' if system.visual_detector else '✗ Not Available'}")
            print(f"Audio Detector: {'✓ Ready' if system.audio_model else '✗ Not Available'}")
            print(f"Pushbullet: {'✓ Connected' if system.pushbullet else '✗ Not Connected'}")
            print(f"API Server: {'✓ Running' if system.server_process else '✗ Not Running'}")
            print(f"Detection Counts:")
            print(f"  Visual: {system.detection_counts['visual']}")
            print(f"  Human: {system.detection_counts['human']}")
            print(f"  Animal: {system.detection_counts['animal']}")
            print(f"  Gunshot: {system.detection_counts['gunshot']}")
            
        elif choice == '8':
            print("\n⚙️ Configuration:")
            print(f"Visual Confidence Threshold: {system.config['visual_confidence_threshold']}")
            print(f"Audio Confidence Threshold: {system.config['audio_confidence_threshold']}")
            print(f"Notification Cooldown: {system.config['notification_cooldown']}s")
            print(f"Audio Duration: {system.config['audio_duration']}s")
            print(f"Output Directory: {system.config['output_dir']}")
            
            if input("\nModify configuration? (y/n): ").lower() == 'y':
                try:
                    new_visual_threshold = float(input(f"Visual confidence threshold ({system.config['visual_confidence_threshold']}): ") or system.config['visual_confidence_threshold'])
                    new_audio_threshold = float(input(f"Audio confidence threshold ({system.config['audio_confidence_threshold']}): ") or system.config['audio_confidence_threshold'])
                    new_cooldown = int(input(f"Notification cooldown ({system.config['notification_cooldown']}s): ") or system.config['notification_cooldown'])
                    
                    system.config['visual_confidence_threshold'] = new_visual_threshold
                    system.config['audio_confidence_threshold'] = new_audio_threshold
                    system.config['notification_cooldown'] = new_cooldown
                    
                    print("✓ Configuration updated")
                except ValueError:
                    print("✗ Invalid input, configuration unchanged")
            
        elif choice == '9':
            print("\n👋 Exiting...")
            system.cleanup()
            break
            
        else:
            print("Invalid choice. Please enter 1-9.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nProgram interrupted by user")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("Program terminated")