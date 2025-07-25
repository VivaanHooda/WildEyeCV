#!/usr/bin/env python3
"""
Enhanced Animal Detection Training Script
Supports full COCO dataset with better performance and monitoring
"""

import torch
import torch.multiprocessing as mp
from enhanced_animal_detector import EnhancedAnimalDetector, get_enhanced_data_loaders
import os
import json
import matplotlib.pyplot as plt
from datetime import datetime
import argparse
import sys

def setup_training_environment():
    """Setup optimal training environment"""
    # Set multiprocessing start method
    if mp.get_start_method(allow_none=True) != 'spawn':
        mp.set_start_method('spawn', force=True)