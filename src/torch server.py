from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
from datetime import datetime

app = Flask(__name__)
CORS(app)

detections = []

@app.route('/api/images/<path:image_path>')
def serve_image(image_path):
    try:
        image_path = image_path.replace('\\', '/')
        return send_file(image_path, mimetype='image/jpeg')
    except Exception as e:
        return jsonify({"error": "Image not found"}), 404

@app.route('/api/detections', methods=['GET'])
def get_detections():
    return jsonify(detections)

@app.route('/api/add_detection', methods=['POST'])
def add_detection():
    detection_data = request.json
    new_detection = {
        'id': len(detections) + 1,
        'timestamp': datetime.fromtimestamp(detection_data.get('timestamp', 0)).isoformat(),
        'animal': detection_data.get('animal', 'Unknown'),
        'location': detection_data.get('location', 'Unknown'),
        'confidence': detection_data.get('confidence', 0.0),
        'image_path': detection_data.get('image_path', None),
        'status': 'Active'
    }
    detections.append(new_detection)
    print("Added new detection:", new_detection)
    return jsonify({"status": "success", "detection": new_detection})

@app.route('/api/update_status/<int:detection_id>', methods=['POST'])
def update_status(detection_id):
    try:
        status = request.json.get('status', 'Cleared')
        for detection in detections:
            if detection['id'] == detection_id:
                detection['status'] = status
                return jsonify({"status": "success", "detection": detection})
        return jsonify({"status": "error", "message": "Detection not found"}), 404
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)

