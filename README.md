# 🛡️ Anti-Poaching Surveillance System using AI & IoT

This project is a **real-time AI-powered anti-poaching surveillance system** that integrates **image-based human and animal detection** with **audio-based gunshot detection** using edge-deployable CNN models. Built with affordability and deployment-readiness in mind, it’s designed to help **forest rangers detect intrusions and threats instantly** using sensor pods.

> 🔖 Latest Stable Release: **v3**  
> 📁 Core Detection Script: `integrated_detection_v5.py`

---

## 📸 Project Overview

Illegal poaching is a major threat to biodiversity. This system uses **camera feeds** and **environmental audio** from forest-installed pods to identify suspicious human activity and gunshots in real-time, sending instant alerts via a backend notification system.

Each **sensor pod** contains:
- A camera module
- A microphone
- GPS for location tagging
- Ultrasonic sensors for motion tracking
- An Arduino-controlled servo motor for camera panning

These pods feed data to integrated AI models that detect and analyze:
- **Humans & animals** in video
- **Gunshots** in audio

Alerts are automatically pushed to rangers via **Pushbullet** notifications, containing:
- Location (GPS)
- Time of detection
- Visual evidence (image)
- Audio clip (if relevant)

---

## 🧠 AI Models

### 1. Human & Animal Detector (Camera-based)
- Built using **PyTorch**, inspired by **YOLOv5** principles
- Trained for **30 epochs**; **Epoch 20** gave best results
- Dataset: COCO + curated animal images
- Accuracy: **>98%** in simulated conditions

### 2. Gunshot Detector (Audio-based)
- Built using **CNN** with **MFCC**-based spectrogram inputs
- Dataset: Gunshot sounds from **Kaggle**
- Accuracy: **~90%** in classification

---

## 🗂️ Repository Structure

```bash
📁 Training_Code/          # Contains all training scripts
└── src/
    └── unified_detections/
        ├── integrated_detection_v5.py   # ✅ Latest working version (tag: v3)
        ├── gunshot_detector.py          # Gunshot model
        ├── enhanced_training_script.py  # Model training script
        ├── final_detector.py            # Unified detection logic
        ├── flask_server.py              # Backend API for alerts
        ├── torch_server.py              # Torch-based detection server
        ├── training_graphs.py           # Model performance graphs
        ├── AnimalDetectionDashboard.js  # Dashboard UI (if used)
        └── ...
```

---

## 📥 Download the Model

Due to the relatively large size of the trained AI model, it is not included in this repository. Please download the **best performing model** (Epoch 20) from the link below:

📌 **[Download `best_model.pth`](https://drive.google.com/file/d/1jVHQoEx7L2ZPFtwcHV8DU7CziB6z2F6e/view?usp=sharing)**  
📝 Ensure this file is placed in the **same directory** as `integrated_detection_v5.py`.

---

## ▶️ Running the Detector

```bash
# Step 1: Install dependencies
pip install -r requirements.txt

# Step 2: Run the integrated detection system
python integrated_detection_v5.py
```

For SSL-related issues on some networks, refer to:
- `ssl_workaround.py`
- `test2_with_ssl_fix.py`

---

## 🌐 Backend Alerts

- **Flask Server**: Handles API triggers
- **Pushbullet**: Sends real-time alerts to ranger devices
- Triggered on gunshot/human detection
- Alert contains image, timestamp, GPS, and audio clip

---

## 📊 Training

All model training scripts are available under `Training_Code/`.

- Training was done on **RunPod Cloud** (NVIDIA RTX 4090)
- 30 epochs, best performance at **epoch 20**
- Metrics and loss curves available in:
  - `training_graphs.py`
  - `torch_server.py`

---

## 📦 Dependencies

- PyTorch
- OpenCV
- Flask
- NumPy, Pandas
- Librosa
- Pushbullet.py
- Matplotlib
- TQDM

Install with:

```bash
pip install -r requirements.txt
```

---

## 🏷️ GitHub Tags

- **v1**: Initial working prototype
- **v2**: Modular pod integration, improved gunshot detection
- **v3**: Latest stable version using `integrated_detection_v5.py`

---

## 🧪 Testing and Performance

| Feature            | Accuracy     | Latency |
|-------------------|--------------|---------|
| Human Detection   | >98%         | <2 sec  |
| Animal Detection  | >98%         | <2 sec  |
| Gunshot Detection | ~90%         | <2 sec  |

Model is optimized for **low power**, **offline deployment**, and **edge inference**.

---

## 📚 Documentation

Full technical report and implementation details can be found here:
- [📄 Unpublished Paper PDF](https://drive.google.com/file/d/1_s0hC6Il7NhcbeWVox9EWxUbcrciseXP/view?usp=sharing)
- [📘 EL Report PDF](https://drive.google.com/file/d/1B7D9sXXb16WTmadQxiPt68nnwwCTdMVP/view?usp=sharing)
- [📀 Video Demonstration](https://drive.google.com/file/d/1ExfGBStDfSps7XRbbWfGsixl_-ZQynjX/view?usp=sharing)
- [📜 Poster](https://drive.google.com/file/d/1i6KM6y8oWH_YLwYJXIkCGD7_lwVzMAEK/view?usp=sharing)

---

## 📌 License

This project is for academic and non-commercial research use only. For commercial licensing or deployments, contact the authors.

---
