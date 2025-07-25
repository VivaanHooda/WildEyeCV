#!/usr/bin/env python3
"""
Complete setup and run script for Animal Detection System
"""

import os
import sys
import subprocess
import time

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"\n{'='*50}")
    print(f"STEP: {description}")
    print(f"{'='*50}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed")
        print(f"Error: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("Error: Python 3.8 or higher is required")
        return False
    print(f"✓ Python {version.major}.{version.minor}.{version.micro} is compatible")
    return True

def install_requirements():
    """Install required packages"""
    print("Installing required packages...")
    
    # Install PyTorch first (CPU version for compatibility)
    torch_command = "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu"
    if not run_command(torch_command, "Installing PyTorch"):
        return False
    
    # Install other requirements
    requirements = [
        "opencv-python",
        "pillow",
        "numpy",
        "pandas", 
        "tqdm",
        "requests",
        "pycocotools",
        "pushbullet.py",
        "flask",
        "flask-cors"
    ]
    
    for req in requirements:
        if not run_command(f"pip install {req}", f"Installing {req}"):
            print(f"Warning: Failed to install {req}, continuing anyway...")
    
    return True

def download_dataset():
    """Download COCO dataset"""
    if os.path.exists("coco/annotations/instances_train2017.json"):
        print("✓ COCO dataset already exists, skipping download")
        return True
    
    print("Downloading COCO dataset (this may take a while)...")
    return run_command("python download_coco.py", "Downloading COCO dataset")

def train_model():
    """Train the model"""
    if os.path.exists("animals_trained.pth"):
        print("✓ Trained model already exists, skipping training")
        return True
    
    print("Training the model (this may take 10-30 minutes)...")
    return run_command("python main.py", "Training animal detection model")

def test_webcam():
    """Test webcam detection"""
    print("Starting webcam detection...")
    print("This will open a window showing your webcam feed with animal detection.")
    print("Press 'q' in the webcam window to quit.")
    
    try:
        subprocess.run("python test2.py", shell=True, check=True)
        return True
    except KeyboardInterrupt:
        print("\n✓ Webcam detection stopped by user")
        return True
    except subprocess.CalledProcessError:
        print("✗ Webcam detection failed")
        return False

def main():
    """Main setup and run function"""
    print("Animal Detection System - Complete Setup")
    print("=" * 50)
    
    # Step 1: Check Python version
    if not check_python_version():
        return
    
    # Step 2: Install requirements
    if not install_requirements():
        print("Failed to install requirements. Please install manually using:")
        print("pip install -r requirements.txt")
        return
    
    # Step 3: Download dataset
    if not download_dataset():
        print("Failed to download dataset. Please check your internet connection.")
        return
    
    # Step 4: Train model
    if not train_model():
        print("Model training failed. You can still try webcam detection with pretrained weights.")
    
    # Step 5: Test webcam
    print("\n" + "="*50)
    print("SETUP COMPLETE!")
    print("="*50)
    print("Ready to start webcam detection!")
    
    input("Press Enter to start webcam detection (or Ctrl+C to exit)...")
    test_webcam()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nSetup interrupted by user")
    except Exception as e:
        print(f"Setup failed with error: {e}")
        print("Please check the error messages above and try manual setup.")