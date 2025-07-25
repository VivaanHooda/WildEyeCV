#!/usr/bin/env python3
"""
Standalone Gunshot Detection System
Renamed from main2.py for integration purposes
This can be used independently or as reference for the unified system
"""

import sounddevice as sd
import soundfile as sf
import numpy as np
import librosa
import tensorflow as tf
from keras.models import load_model
import os
import shutil
from pushbullet import Pushbullet
import time

# Configuration
CONFIG = {
    'pushbullet_token': "o.0fO4N5ybzVV9k4ESDT2oRt06BexvYPKu",
    'model_path': "gunshot_full.h5",
    'recording_folder': "recording",
    'duration': 2,  # seconds
    'sample_rate': 22050,
    'classes': ['Background noise', 'Background noise', 'Background noise', 
                'Background noise', 'gunshot', 'Background noise', 'gunshot', 
                'gunshot', 'Background noise', 'Background noise']
}

class StandaloneGunShotDetector:
    def __init__(self):
        self.pb = Pushbullet(CONFIG['pushbullet_token'])
        self.model = None
        self.setup_recording_folder()
        self.load_model()
    
    def setup_recording_folder(self):
        """Setup and clean recording folder"""
        folder = CONFIG['recording_folder']
        os.makedirs(folder, exist_ok=True)
        
        # Clear existing contents
        for file in os.listdir(folder):
            file_path = os.path.join(folder, file)
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        
        print(f"✅ '{folder}' is ready and clean.")
    
    def load_model(self):
        """Load the trained gunshot detection model"""
        try:
            self.model = load_model(CONFIG['model_path'])
            print(f"✅ Gunshot detection model loaded from {CONFIG['model_path']}")
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            raise
    
    def record_audio(self):
        """Record audio from microphone"""
        print("🎙️ Recording...")
        audio = sd.rec(
            int(CONFIG['duration'] * CONFIG['sample_rate']), 
            samplerate=CONFIG['sample_rate'], 
            channels=1
        )
        sd.wait()
        return audio.flatten()
    
    def extract_mel_features(self, audio):
        """Extract mel-spectrogram features from audio"""
        mels = librosa.feature.melspectrogram(y=audio, sr=CONFIG['sample_rate'])
        mels_db = np.mean(mels.T, axis=0)  # shape (128,)
        return np.expand_dims(mels_db, axis=0)  # shape (1, 128)
    
    def detect_gunshot(self, audio):
        """Detect gunshot in audio"""
        mel_features = self.extract_mel_features(audio)
        prediction = self.model.predict(mel_features, verbose=0)
        predicted_class = CONFIG['classes'][np.argmax(prediction)]
        confidence = float(prediction[0][np.argmax(prediction)])
        
        return predicted_class, confidence, np.argmax(prediction)
    
    def send_gunshot_alert(self, confidence, audio_file):
        """Send gunshot alert via Pushbullet"""
        try:
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            
            # Send text notification
            push = self.pb.push_note(
                "🚨 Gunshot Detected", 
                f"A gunshot was detected by the audio monitoring system!\n"
                f"Time: {timestamp}\n"
                f"Confidence: {confidence:.2f}"
            )
            
            # Send audio file
            with open(audio_file, "rb") as f:
                file_data = self.pb.upload_file(f, "gunshot_clip.wav")
                self.pb.push_file(**file_data)
            
            print(f"🚨 Gunshot alert sent! Confidence: {confidence:.2f}")
            
        except Exception as e:
            print(f"❌ Failed to send alert: {e}")
    
    def run_continuous_detection(self):
        """Run continuous gunshot detection"""
        print("🔍 Starting continuous gunshot detection...")
        print("Press Ctrl+C to stop")
        
        detection_count = 0
        
        try:
            while True:
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                
                # Record audio
                audio = self.record_audio()
                
                # Save audio file
                audio_filename = f'recorded_audio{detection_count}.wav'
                audio_path = os.path.join(CONFIG['recording_folder'], audio_filename)
                sf.write(audio_path, audio, CONFIG['sample_rate'])
                
                # Detect gunshot
                predicted_class, confidence, class_index = self.detect_gunshot(audio)
                
                print(f"🎙️ Detection {detection_count}: {predicted_class}:{class_index} "
                      f"(Confidence: {confidence:.2f}) at {timestamp}")
                
                # Send alert if gunshot detected
                if predicted_class == 'gunshot':
                    self.send_gunshot_alert(confidence, audio_path)
                
                detection_count += 1
                
        except KeyboardInterrupt:
            print(f"\n🛑 Detection stopped. Total detections: {detection_count}")
        except Exception as e:
            print(f"❌ Detection error: {e}")
    
    def test_detection(self):
        """Test detection with a single recording"""
        print("🧪 Testing gunshot detection...")
        
        # Record test audio
        audio = self.record_audio()
        
        # Save test audio
        test_path = os.path.join(CONFIG['recording_folder'], 'test_audio.wav')
        sf.write(test_path, audio, CONFIG['sample_rate'])
        
        # Run detection
        predicted_class, confidence, class_index = self.detect_gunshot(audio)
        
        print(f"Test Result: {predicted_class} (Confidence: {confidence:.2f})")
        
        if predicted_class == 'gunshot':
            print("🚨 Gunshot detected in test!")
            self.send_gunshot_alert(confidence, test_path)
        else:
            print("✅ No gunshot detected in test")

def main():
    """Main function"""
    detector = StandaloneGunShotDetector()
    
    print("🔫 Standalone Gunshot Detection System")
    print("1. Start continuous detection")
    print("2. Test detection")
    print("3. Exit")
    
    try:
        choice = input("Choose option (1-3): ").strip()
        
        if choice == '1':
            detector.run_continuous_detection()
        elif choice == '2':
            detector.test_detection()
        elif choice == '3':
            print("Goodbye!")
        else:
            print("Invalid choice")
            
    except KeyboardInterrupt:
        print("\nExiting...")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()