#!/usr/bin/env python3
"""
Unified Animal & Gunshot Detection System V4
Integrates visual animal/human detection with audio gunshot detection
Sends coordinated Pushbullet notifications for all threat types
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

# Audio detection imports
import sounddevice as sd
import soundfile as sf
import librosa
import tensorflow as tf
from keras.models import load_model

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
    print("Warning: enhanced_animal_detector.py not found. Visual detection disabled.")
    VISUAL_DETECTOR_AVAILABLE = False

class UnifiedDetectionSystem:
    def __init__(self):
        # Configuration
        self.config = {
            'pushbullet_api_key': "o.0fO4N5ybzVV9k4ESDT2oRt06BexvYPKu",
            'visual_model_path': 'checkpoints/best_model.pth',
            'audio_model_path': 'gunshot_full.h5',
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
        
        # Threading controls
        self.running = False
        self.visual_thread = None
        self.audio_thread = None
        self.notification_queue = queue.Queue()
        self.detection_history = []
        
        # Notification tracking
        self.last_notifications = {
            'human': 0,
            'gunshot': 0,
            'animal': 0
        }
        
        # Audio classes (from your original code)
        self.audio_classes = [
            'Background noise', 'Background noise', 'Background noise', 
            'Background noise', 'gunshot', 'Background noise', 'gunshot', 
            'gunshot', 'Background noise', 'Background noise'
        ]
        
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
            print(f"✓ Directory ready: {dir_path}")
    
    def initialize_pushbullet(self):
        """Initialize Pushbullet for notifications"""
        if not PUSHBULLET_AVAILABLE:
            print("✗ Pushbullet not available")
            return False
        
        try:
            self.pushbullet = Pushbullet(self.config['pushbullet_api_key'])
            print("✓ Pushbullet initialized successfully")
            return True
        except Exception as e:
            print(f"✗ Pushbullet initialization failed: {e}")
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
            print("✗ Visual model not found or failed to load")
            return False
        except Exception as e:
            print(f"✗ Visual detector initialization failed: {e}")
            return False
    
    def initialize_audio_detector(self):
        """Initialize audio detection system"""
        try:
            if os.path.exists(self.config['audio_model_path']):
                self.audio_model = load_model(self.config['audio_model_path'])
                print("✓ Audio detector initialized successfully")
                return True
            else:
                print(f"✗ Audio model not found: {self.config['audio_model_path']}")
                return False
        except Exception as e:
            print(f"✗ Audio detector initialization failed: {e}")
            return False
    
    def send_notification(self, threat_type, details):
        """Send unified notification via Pushbullet"""
        if not self.pushbullet:
            print(f"⚠️ Notification not sent - Pushbullet unavailable: {threat_type}")
            return False
        
        current_time = time.time()
        
        # Check cooldown
        if current_time - self.last_notifications[threat_type] < self.config['notification_cooldown']:
            print(f"⏳ Notification cooldown active for {threat_type}")
            return False
        
        try:
            # Prepare notification content
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            if threat_type == 'human':
                title = "🚨 HUMAN DETECTED"
                message = f"Human detected with {details['confidence']:.1%} confidence\nTime: {timestamp}\nLocation: Visual System"
            elif threat_type == 'gunshot':
                title = "🔫 GUNSHOT DETECTED"
                message = f"Gunshot detected with {details['confidence']:.1%} confidence\nTime: {timestamp}\nLocation: Audio System"
            elif threat_type == 'animal':
                title = f"🦌 ANIMAL DETECTED: {details['animal_type']}"
                message = f"{details['animal_type']} detected with {details['confidence']:.1%} confidence\nTime: {timestamp}\nLocation: Visual System"
            
            # Send text notification
            self.pushbullet.push_note(title, message)
            
            # Send file if available
            if 'file_path' in details and os.path.exists(details['file_path']):
                try:
                    with open(details['file_path'], "rb") as file:
                        if threat_type == 'gunshot':
                            file_data = self.pushbullet.upload_file(file, "gunshot_audio.wav")
                        else:
                            file_data = self.pushbullet.upload_file(file, f"{threat_type}_detection.jpg")
                        self.pushbullet.push_file(**file_data)
                except Exception as e:
                    print(f"Could not send file: {e}")
            
            self.last_notifications[threat_type] = current_time
            print(f"✓ {threat_type.upper()} notification sent successfully")
            
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
    
    def visual_detection_loop(self):
        """Main visual detection loop"""
        print("🎥 Starting visual detection...")
        
        if not self.visual_detector:
            print("✗ Visual detector not available")
            return
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("✗ Could not open camera")
            return
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        detection_count = 0
        last_detection_time = 0
        detection_cooldown = 2.0
        
        while self.running:
            ret, frame = cap.read()
            if not ret:
                print("✗ Failed to read frame")
                break
            
            # Save frame temporarily
            temp_path = os.path.join(self.config['visual_output_dir'], 'temp_frame.jpg')
            cv2.imwrite(temp_path, frame)
            
            try:
                # Run detection
                boxes, scores, labels, _ = self.visual_detector.detect_image(
                    temp_path, self.config['visual_confidence_threshold']
                )
                
                current_time = time.time()
                
                # Process detections
                for box, score, label in zip(boxes, scores, labels):
                    if label < len(self.visual_detector.class_names):
                        class_name = self.visual_detector.class_names[label]
                        
                        # Determine threat type
                        is_human = self.is_human_detection(class_name)
                        
                        # Draw on frame
                        color = (0, 0, 255) if is_human else (0, 255, 0)
                        x1, y1, x2, y2 = map(int, box)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3 if is_human else 2)
                        
                        label_text = f"{'HUMAN' if is_human else class_name}: {score:.2f}"
                        cv2.putText(frame, label_text, (x1, y1 - 10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.8 if is_human else 0.7, color, 2)
                        
                        # Send notification (with cooldown)
                        if (current_time - last_detection_time) > detection_cooldown:
                            # Save detection image
                            detection_filename = f"detection_{detection_count:04d}_{class_name}_{score:.2f}.jpg"
                            detection_path = os.path.join(self.config['visual_output_dir'], detection_filename)
                            cv2.imwrite(detection_path, frame)
                            
                            # Prepare notification
                            if is_human:
                                self.send_notification('human', {
                                    'confidence': score,
                                    'file_path': detection_path
                                })
                            else:
                                self.send_notification('animal', {
                                    'animal_type': class_name,
                                    'confidence': score,
                                    'file_path': detection_path
                                })
                            
                            detection_count += 1
                            last_detection_time = current_time
                
                # Add status overlay
                status_text = f"Visual Detections: {detection_count} | Press 'q' to quit"
                cv2.putText(frame, status_text, (10, frame.shape[0] - 20),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
            except Exception as e:
                print(f"Visual detection error: {e}")
                cv2.putText(frame, f"Error: {str(e)[:50]}", (10, 60),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            
            # Display frame
            cv2.imshow('Unified Detection System V4', frame)
            
            # Handle key press
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.running = False
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        # Clean up
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        print(f"🎥 Visual detection stopped. Total detections: {detection_count}")
    
    def audio_detection_loop(self):
        """Main audio detection loop"""
        print("🎙️ Starting audio detection...")
        
        if not self.audio_model:
            print("✗ Audio model not available")
            return
        
        audio_detection_count = 0
        
        while self.running:
            try:
                # Record audio
                audio = self.record_audio()
                
                # Save audio file
                audio_filename = f"audio_{audio_detection_count:04d}.wav"
                audio_path = os.path.join(self.config['audio_output_dir'], audio_filename)
                sf.write(audio_path, audio, self.config['audio_sample_rate'])
                
                # Extract features and predict
                mel_features = self.extract_mel_features(audio)
                prediction = self.audio_model.predict(mel_features, verbose=0)
                predicted_class = self.audio_classes[np.argmax(prediction)]
                confidence = float(prediction[0][np.argmax(prediction)])
                
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"🎙️ Audio {audio_detection_count}: {predicted_class} ({confidence:.2f}) at {timestamp}")
                
                # Check for gunshot
                if predicted_class == 'gunshot' and confidence > self.config['audio_confidence_threshold']:
                    self.send_notification('gunshot', {
                        'confidence': confidence,
                        'file_path': audio_path
                    })
                
                audio_detection_count += 1
                
            except Exception as e:
                print(f"Audio detection error: {e}")
                time.sleep(1)
        
        print(f"🎙️ Audio detection stopped. Total checks: {audio_detection_count}")
    
    def record_audio(self):
        """Record audio from microphone"""
        audio = sd.rec(
            int(self.config['audio_duration'] * self.config['audio_sample_rate']), 
            samplerate=self.config['audio_sample_rate'], 
            channels=1
        )
        sd.wait()
        return audio.flatten()
    
    def extract_mel_features(self, audio):
        """Extract mel-spectrogram features from audio"""
        mels = librosa.feature.melspectrogram(y=audio, sr=self.config['audio_sample_rate'])
        mels_db = np.mean(mels.T, axis=0)  # shape (128,)
        return np.expand_dims(mels_db, axis=0)  # shape (1, 128)
    
    def is_human_detection(self, class_name):
        """Check if detected class is human"""
        human_classes = ['person', 'human', 'people']
        return class_name.lower() in human_classes
    
    def run_system(self):
        """Run the complete unified detection system"""
        print("=" * 80)
        print("🔍 UNIFIED DETECTION SYSTEM V4")
        print("🎥 Visual: Animals & Human Detection")
        print("🎙️ Audio: Gunshot Detection")
        print("📱 Notifications: Pushbullet")
        print("=" * 80)
        
        # Check system readiness
        systems_ready = []
        
        if self.visual_detector:
            systems_ready.append("Visual Detection")
        if self.audio_model:
            systems_ready.append("Audio Detection")
        if self.pushbullet:
            systems_ready.append("Pushbullet Notifications")
        
        if not systems_ready:
            print("✗ No detection systems available. Please check your setup.")
            return False
        
        print(f"✓ Active Systems: {', '.join(systems_ready)}")
        print("Press Ctrl+C to stop all systems")
        
        # Start detection threads
        self.running = True
        
        if self.visual_detector:
            self.visual_thread = threading.Thread(target=self.visual_detection_loop)
            self.visual_thread.daemon = True
            self.visual_thread.start()
        
        if self.audio_model:
            self.audio_thread = threading.Thread(target=self.audio_detection_loop)
            self.audio_thread.daemon = True
            self.audio_thread.start()
        
        try:
            # Keep main thread alive
            while self.running:
                time.sleep(1)
                
                # Print status every 30 seconds
                if int(time.time()) % 30 == 0:
                    print(f"📊 System Status: {len(self.detection_history)} total detections logged")
                    
        except KeyboardInterrupt:
            print("\n🛑 Stopping detection systems...")
            self.running = False
            
            # Wait for threads to finish
            if self.visual_thread and self.visual_thread.is_alive():
                self.visual_thread.join(timeout=2)
            if self.audio_thread and self.audio_thread.is_alive():
                self.audio_thread.join(timeout=2)
            
            print("✓ All systems stopped")
        
        return True
    
    def run_test_mode(self):
        """Run system in test mode"""
        print("🧪 Running in test mode...")
        
        # Test visual detection on sample image
        if self.visual_detector:
            print("Testing visual detection...")
            # Add test image logic here
        
        # Test audio detection
        if self.audio_model:
            print("Testing audio detection...")
            # Record a short test audio
            test_audio = self.record_audio()
            test_path = os.path.join(self.config['audio_output_dir'], 'test_audio.wav')
            sf.write(test_path, test_audio, self.config['audio_sample_rate'])
            
            # Process test audio
            mel_features = self.extract_mel_features(test_audio)
            prediction = self.audio_model.predict(mel_features, verbose=0)
            predicted_class = self.audio_classes[np.argmax(prediction)]
            confidence = float(prediction[0][np.argmax(prediction)])
            
            print(f"Test audio result: {predicted_class} ({confidence:.2f})")
        
        # Test notification
        if self.pushbullet:
            print("Testing notification system...")
            self.send_notification('human', {
                'confidence': 0.95,
                'file_path': None
            })
        
        print("✓ Test mode completed")

def main():
    """Main function with menu"""
    system = UnifiedDetectionSystem()
    
    print("🔍 Unified Detection System V4")
    print("1. Run complete system")
    print("2. Run test mode")
    print("3. Visual detection only")
    print("4. Audio detection only")
    print("5. Show detection history")
    
    try:
        choice = input("Choose option (1-5): ").strip()
        
        if choice == '1':
            system.run_system()
        elif choice == '2':
            system.run_test_mode()
        elif choice == '3':
            if system.visual_detector:
                system.running = True
                system.visual_detection_loop()
            else:
                print("Visual detection not available")
        elif choice == '4':
            if system.audio_model:
                system.running = True
                system.audio_detection_loop()
            else:
                print("Audio detection not available")
        elif choice == '5':
            if system.detection_history:
                print("\n📋 Detection History:")
                for entry in system.detection_history[-10:]:  # Show last 10
                    print(f"  {entry['timestamp']}: {entry['type']} - {entry['details']}")
            else:
                print("No detection history available")
        else:
            print("Invalid choice")
            
    except KeyboardInterrupt:
        print("\nExiting...")
        system.running = False
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()