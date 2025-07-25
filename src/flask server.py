#!/usr/bin/env python3
"""
Enhanced Flask API Server for React Dashboard Integration
Provides REST API endpoints for the React frontend to communicate with the detection system
"""

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import os
import json
import threading
import time
from datetime import datetime
import base64
import cv2
import numpy as np
from io import BytesIO
import traceback

# Import your detection system
try:
    from integrated_detection_v5 import EnhancedUnifiedDetectionSystem
    DETECTION_SYSTEM_AVAILABLE = True
except ImportError:
    print("Warning: Detection system not found. Some features will be limited.")
    DETECTION_SYSTEM_AVAILABLE = False

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
CORS(app, origins=['http://localhost:3000', 'http://localhost:5173'])  # Common React dev ports
socketio = SocketIO(app, cors_allowed_origins="*")

# Global variables
detection_system = None
detection_running = False
detection_thread = None
system_stats = {
    'fps': 0,
    'cpu_usage': 0,
    'memory_usage': 0,
    'uptime': 0
}
start_time = time.time()

# Initialize detection system
if DETECTION_SYSTEM_AVAILABLE:
    try:
        detection_system = EnhancedUnifiedDetectionSystem()
        print("✓ Detection system initialized")
    except Exception as e:
        print(f"✗ Failed to initialize detection system: {e}")
        detection_system = None

def get_system_stats():
    """Get current system statistics"""
    try:
        import psutil
        cpu_percent = psutil.cpu_percent(interval=1)
        memory_percent = psutil.virtual_memory().percent
    except ImportError:
        cpu_percent = np.random.randint(40, 60)  # Mock data
        memory_percent = np.random.randint(50, 80)
    
    return {
        'fps': np.random.randint(28, 32),  # Mock FPS
        'cpu_usage': cpu_percent,
        'memory_usage': memory_percent,
        'uptime': int(time.time() - start_time)
    }

def detection_worker():
    """Background worker for detection system"""
    global detection_running, system_stats
    
    if not detection_system:
        return
    
    print("🚀 Starting detection worker...")
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("✗ Could not open webcam")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    frame_count = 0
    start_time = time.time()
    
    try:
        while detection_running:
            ret, frame = cap.read()
            if not ret:
                continue
            
            frame_count += 1
            
            # Calculate FPS
            elapsed_time = time.time() - start_time
            if elapsed_time > 1.0:
                fps = frame_count / elapsed_time
                system_stats['fps'] = int(fps)
                frame_count = 0
                start_time = time.time()
            
            # Save frame temporarily for detection
            temp_path = 'temp_frame.jpg'
            cv2.imwrite(temp_path, frame)
            
            try:
                # Run detection
                if detection_system.visual_detector:
                    boxes, scores, labels, _ = detection_system.visual_detector.detect_image(
                        temp_path, detection_system.config['visual_confidence_threshold']
                    )
                    
                    # Process detections
                    for box, score, label in zip(boxes, scores, labels):
                        if label < len(detection_system.visual_detector.class_names):
                            class_name = detection_system.visual_detector.class_names[label]
                            
                            # Determine detection type
                            is_human = detection_system.is_human_detection(class_name)
                            detection_type = 'human' if is_human else 'animal'
                            
                            # Create detection data
                            detection_data = {
                                'id': int(time.time() * 1000),
                                'type': detection_type,
                                'confidence': float(score),
                                'timestamp': datetime.now().isoformat(),
                                'class_name': class_name,
                                'bounding_box': [int(x) for x in box]
                            }
                            
                            # Update counters
                            if is_human:
                                detection_system.detection_counts['human'] += 1
                            else:
                                detection_system.detection_counts['animal'] += 1
                            
                            detection_system.detection_counts['total'] = (
                                detection_system.detection_counts['human'] + 
                                detection_system.detection_counts['animal'] + 
                                detection_system.detection_counts['gunshot']
                            )
                            
                            # Save detection image
                            detection_filename = f"{detection_type}_{int(time.time())}.jpg"
                            detection_path = os.path.join(detection_system.config['visual_output_dir'], detection_filename)
                            
                            # Draw bounding box on frame
                            x1, y1, x2, y2 = map(int, box)
                            color = (0, 0, 255) if is_human else (0, 255, 0)
                            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                            cv2.putText(frame, f"{class_name}: {score:.2f}", (x1, y1 - 10),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                            
                            cv2.imwrite(detection_path, frame)
                            detection_data['image_path'] = detection_path
                            
                            # Emit detection to React frontend
                            socketio.emit('new_detection', detection_data)
                            
                            # Send notification
                            if is_human:
                                detection_system.send_unified_alert('human', {
                                    'confidence': score,
                                    'file_path': detection_path
                                })
                            else:
                                detection_system.send_unified_alert('animal', {
                                    'animal_type': class_name,
                                    'confidence': score,
                                    'file_path': detection_path
                                })
                            
                            print(f"🔍 {detection_type.upper()} detected: {class_name} ({score:.2f})")
                
            except Exception as e:
                print(f"Detection error: {e}")
            
            # Clean up temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)
            
            # Update system stats
            system_stats.update(get_system_stats())
            socketio.emit('system_stats', system_stats)
            
            time.sleep(0.1)  # Small delay to prevent overwhelming
    
    except Exception as e:
        print(f"Detection worker error: {e}")
        traceback.print_exc()
    
    finally:
        cap.release()
        if os.path.exists('temp_frame.jpg'):
            os.remove('temp_frame.jpg')
        print("🛑 Detection worker stopped")

# API Routes

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'detection_system_available': detection_system is not None
    })

@app.route('/api/system/status', methods=['GET'])
def get_system_status():
    """Get system status"""
    global system_stats
    
    status = {
        'detection_running': detection_running,
        'visual_detector_available': detection_system and detection_system.visual_detector is not None,
        'audio_detector_available': detection_system and detection_system.audio_model is not None,
        'pushbullet_available': detection_system and detection_system.pushbullet is not None,
        'uptime': int(time.time() - start_time),
        'stats': system_stats
    }
    
    return jsonify(status)

@app.route('/api/detection/start', methods=['POST'])
def start_detection():
    """Start detection system"""
    global detection_running, detection_thread
    
    if not detection_system:
        return jsonify({'error': 'Detection system not available'}), 400
    
    if detection_running:
        return jsonify({'error': 'Detection already running'}), 400
    
    try:
        detection_running = True
        detection_thread = threading.Thread(target=detection_worker)
        detection_thread.daemon = True
        detection_thread.start()
        
        # Start audio detection if available
        if detection_system.audio_model:
            detection_system.running = True
            detection_system.audio_thread = threading.Thread(target=detection_system.audio_detection_loop)
            detection_system.audio_thread.daemon = True
            detection_system.audio_thread.start()
        
        return jsonify({'message': 'Detection started successfully'})
    
    except Exception as e:
        detection_running = False
        return jsonify({'error': str(e)}), 500

@app.route('/api/detection/stop', methods=['POST'])
def stop_detection():
    """Stop detection system"""
    global detection_running, detection_thread
    
    try:
        detection_running = False
        
        if detection_system:
            detection_system.running = False
        
        if detection_thread and detection_thread.is_alive():
            detection_thread.join(timeout=5)
        
        return jsonify({'message': 'Detection stopped successfully'})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/detection/counts', methods=['GET'])
def get_detection_counts():
    """Get detection counts"""
    if not detection_system:
        return jsonify({'error': 'Detection system not available'}), 400
    
    return jsonify(detection_system.detection_counts)

@app.route('/api/detection/history', methods=['GET'])
def get_detection_history():
    """Get detection history"""
    if not detection_system:
        return jsonify({'error': 'Detection system not available'}), 400
    
    limit = request.args.get('limit', 50, type=int)
    history = detection_system.detection_history[-limit:] if detection_system.detection_history else []
    
    return jsonify(history)

@app.route('/api/detection/clear', methods=['POST'])
def clear_detection_history():
    """Clear detection history"""
    if not detection_system:
        return jsonify({'error': 'Detection system not available'}), 400
    
    detection_system.detection_history = []
    detection_system.detection_counts = {
        'total': 0,
        'human': 0,
        'animal': 0,
        'gunshot': 0
    }
    
    return jsonify({'message': 'History cleared successfully'})

@app.route('/api/settings', methods=['GET'])
def get_settings():
    """Get current settings"""
    if not detection_system:
        return jsonify({'error': 'Detection system not available'}), 400
    
    return jsonify(detection_system.config)

@app.route('/api/settings', methods=['POST'])
def update_settings():
    """Update settings"""
    if not detection_system:
        return jsonify({'error': 'Detection system not available'}), 400
    
    try:
        data = request.get_json()
        
        # Update only allowed settings
        allowed_settings = [
            'visual_confidence_threshold',
            'audio_confidence_threshold',
            'notification_cooldown',
            'audio_duration'
        ]
        
        for key, value in data.items():
            if key in allowed_settings:
                detection_system.config[key] = value
        
        return jsonify({'message': 'Settings updated successfully'})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/detection/image/<path:filename>', methods=['GET'])
def get_detection_image(filename):
    """Get detection image"""
    if not detection_system:
        return jsonify({'error': 'Detection system not available'}), 400
    
    try:
        image_path = os.path.join(detection_system.config['visual_output_dir'], filename)
        if os.path.exists(image_path):
            return send_file(image_path, mimetype='image/jpeg')
        else:
            return jsonify({'error': 'Image not found'}), 404
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/test/detection', methods=['POST'])
def test_detection():
    """Test detection on uploaded image"""
    if not detection_system:
        return jsonify({'error': 'Detection system not available'}), 400
    
    try:
        # Get uploaded file
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No image selected'}), 400
        
        # Save uploaded image
        filename = f"test_{int(time.time())}.jpg"
        filepath = os.path.join(detection_system.config['visual_output_dir'], filename)
        file.save(filepath)
        
        # Run detection
        boxes, scores, labels, _ = detection_system.visual_detector.detect_image(
            filepath, detection_system.config['visual_confidence_threshold']
        )
        
        detections = []
        for box, score, label in zip(boxes, scores, labels):
            if label < len(detection_system.visual_detector.class_names):
                class_name = detection_system.visual_detector.class_names[label]
                detections.append({
                    'class_name': class_name,
                    'confidence': float(score),
                    'bounding_box': [int(x) for x in box],
                    'is_human': detection_system.is_human_detection(class_name)
                })
        
        return jsonify({
            'message': 'Test completed',
            'detections': detections,
            'image_path': filename
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/camera/snapshot', methods=['POST'])
def take_snapshot():
    """Take a snapshot from camera"""
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            return jsonify({'error': 'Camera not available'}), 400
        
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            return jsonify({'error': 'Failed to capture frame'}), 400
        
        # Save snapshot
        filename = f"snapshot_{int(time.time())}.jpg"
        filepath = os.path.join(detection_system.config['output_dir'], filename)
        cv2.imwrite(filepath, frame)
        
        return jsonify({
            'message': 'Snapshot saved',
            'filename': filename
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# WebSocket Events

@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    print('Client connected')
    emit('connected', {'message': 'Connected to detection system'})

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    print('Client disconnected')

@socketio.on('request_status')
def handle_status_request():
    """Handle status request"""
    status = {
        'detection_running': detection_running,
        'system_stats': system_stats,
        'detection_counts': detection_system.detection_counts if detection_system else {}
    }
    emit('status_update', status)

# Background task to send periodic updates
def background_task():
    """Send periodic updates to connected clients"""
    while True:
        if detection_running:
            socketio.emit('system_stats', system_stats)
            if detection_system:
                socketio.emit('detection_counts', detection_system.detection_counts)
        socketio.sleep(2)

# Start background task
socketio.start_background_task(background_task)

if __name__ == '__main__':
    print("🚀 Starting Enhanced Detection API Server")
    print("📡 WebSocket support enabled")
    print("🎯 CORS enabled for React frontend")
    print("=" * 50)
    
    # Create output directories
    if detection_system:
        detection_system.setup_directories()
    
    try:
        # Run with SocketIO support
        socketio.run(app, host='0.0.0.0', port=5001, debug=True, allow_unsafe_werkzeug=True)
    except KeyboardInterrupt:
        print("\nServer stopped by user")
    except Exception as e:
        print(f"Server error: {e}")
    finally:
        # Cleanup
        if detection_running:
            detection_running = False
        print("Server terminated")