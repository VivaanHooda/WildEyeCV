#!/bin/bash
# mac_setup.sh - Animal Detection Setup for Mac

echo "🐾 Animal Detection Setup for Mac"
echo "=================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running on Mac
if [[ "$OSTYPE" != "darwin"* ]]; then
    print_error "This script is designed for macOS only"
    exit 1
fi

print_status "Checking system requirements..."

# Check Python version
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
    print_success "Python $PYTHON_VERSION found"
    
    # Check if version is 3.8 or higher
    if python3 -c 'import sys; exit(0 if sys.version_info >= (3, 8) else 1)'; then
        print_success "Python version is compatible"
    else
        print_error "Python 3.8 or higher is required"
        print_status "Install with: brew install python@3.11"
        exit 1
    fi
else
    print_error "Python 3 not found"
    print_status "Install with: brew install python@3.11"
    exit 1
fi

# Check if pip is available
if ! command -v pip3 &> /dev/null; then
    print_error "pip3 not found"
    exit 1
fi

# Create virtual environment (optional but recommended)
read -p "Create virtual environment? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    print_status "Creating virtual environment..."
    python3 -m venv animal_detection_env
    source animal_detection_env/bin/activate
    print_success "Virtual environment created and activated"
fi

# Install dependencies
print_status "Installing dependencies..."

# Install PyTorch (CPU version for compatibility)
print_status "Installing PyTorch..."
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

if [ $? -eq 0 ]; then
    print_success "PyTorch installed successfully"
else
    print_error "Failed to install PyTorch"
    exit 1
fi

# Install other dependencies
print_status "Installing other dependencies..."
dependencies=(
    "opencv-python"
    "pillow"
    "numpy"
    "pandas"
    "tqdm"
    "requests"
    "pycocotools"
    "flask"
    "flask-cors"
)

for dep in "${dependencies[@]}"; do
    print_status "Installing $dep..."
    pip3 install "$dep"
    if [ $? -eq 0 ]; then
        print_success "$dep installed"
    else
        print_warning "Failed to install $dep - continuing anyway"
    fi
done

# Optional: Install pushbullet for notifications
read -p "Install pushbullet for notifications? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    pip3 install pushbullet.py
    print_success "Pushbullet installed"
fi

# Check camera permissions
print_status "Checking camera permissions..."
print_warning "If camera access fails, go to:"
print_warning "System Preferences → Security & Privacy → Camera"
print_warning "And add Terminal to allowed applications"

# Test basic imports
print_status "Testing imports..."
python3 -c "
import torch
import cv2
import numpy as np
from PIL import Image
print('✅ All imports successful')
" 2>/dev/null

if [ $? -eq 0 ]; then
    print_success "All dependencies imported successfully"
else
    print_error "Some imports failed - check installation"
fi

# Check if model files exist
print_status "Checking project files..."
required_files=("animal_detector_2.py" "test2.py")
missing_files=()

for file in "${required_files[@]}"; do
    if [ ! -f "$file" ]; then
        missing_files+=("$file")
    fi
done

if [ ${#missing_files[@]} -eq 0 ]; then
    print_success "All required files found"
else
    print_warning "Missing files: ${missing_files[*]}"
    print_status "Make sure all project files are in the current directory"
fi

print_success "Setup completed!"
echo
echo "🚀 To start detection:"
echo "   python3 test2.py"
echo
echo "📱 To test with the optimized Mac version:"
echo "   python3 mac_animal_test.py"
echo
echo "🔧 Optional: Download dataset for training:"
echo "   python3 download_coco.py"
echo
echo "⚠️  Remember to enable camera permissions if prompted"