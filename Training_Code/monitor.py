import time
import psutil
import torch
import matplotlib.pyplot as plt
from datetime import datetime

def monitor_resources():
    """Monitor system resources during training"""
    start_time = time.time()
    
    while True:
        try:
            # GPU monitoring
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.memory_allocated() / 1024**3
                gpu_max_memory = torch.cuda.max_memory_allocated() / 1024**3
                gpu_util = torch.cuda.utilization()
            
            # CPU/RAM monitoring  
            cpu_percent = psutil.cpu_percent()
            ram_percent = psutil.virtual_memory().percent
            
            elapsed = (time.time() - start_time) / 3600  # hours
            
            print(f"[{datetime.now().strftime('%H:%M:%S')}] "
                  f"GPU: {gpu_memory:.1f}GB/{gpu_max_memory:.1f}GB | "
                  f"CPU: {cpu_percent:.1f}% | RAM: {ram_percent:.1f}% | "
                  f"Time: {elapsed:.2f}h")
            
            time.sleep(60)  # Update every minute
            
        except KeyboardInterrupt:
            print("Monitoring stopped")
            break

if __name__ == "__main__":
    monitor_resources()