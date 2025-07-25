import React, { useState, useEffect } from 'react';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card';
import { Bell, Camera, X } from 'lucide-react';
import { Alert, AlertTitle, AlertDescription } from '@/components/ui/alert';

const API_URL = 'http://localhost:5000';

const AnimalDetectionDashboard = () => {
  const [detections, setDetections] = useState([]);
  const [activeAlert, setActiveAlert] = useState(null);
  const [error, setError] = useState(null);
  const [selectedDetection, setSelectedDetection] = useState(null);

  const fetchDetections = async () => {
    try {
      const response = await fetch('http://localhost:5000/api/detections');
      if (!response.ok) throw new Error('Failed to fetch detections');
      const data = await response.json();
      setDetections(data);
      const active = data.find(d => d.status === "Active");
      setActiveAlert(active);
    } catch (err) {
      setError(err.message);
      console.error('Error fetching detections:', err);
    }
  };

  const handleClearDetection = async (detectionId) => {
    try {
      const response = await fetch(`http://localhost:5000/api/update_status/${detectionId}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ status: 'Cleared' })
      });
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.message || 'Failed to update status');
      }
      
      // Refresh the detections list
      await fetchDetections();
    } catch (err) {
      setError(err.message);
      console.error('Error updating detection:', err);
    }
  };


  useEffect(() => {
    fetchDetections();
    const interval = setInterval(fetchDetections, 5000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="p-4 max-w-4xl mx-auto space-y-4">
      <h1 className="text-2xl font-bold mb-6">Animal Detection Dashboard</h1>

      {activeAlert && (
        <Alert className="bg-red-50 border-red-200">
          <Bell className="h-4 w-4" />
          <AlertTitle>Active Detection!</AlertTitle>
          <AlertDescription>
            {activeAlert.animal} detected at {activeAlert.location}
          </AlertDescription>
        </Alert>
      )}

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* Detections List */}
        <Card>
          <CardHeader>
            <CardTitle>Recent Detections ({detections.length})</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
            {detections.slice().reverse().map((detection) => (
                <div
                  key={detection.id}
                  onClick={() => setSelectedDetection(detection)}
                  className={`p-4 rounded-lg border cursor-pointer hover:shadow-md transition-shadow ${
                    detection.status === 'Active'
                      ? 'border-red-200 bg-red-50'
                      : 'border-gray-200 bg-gray-50'
                  }`}
                >
                  <div className="flex items-center justify-between">
                    <div className="flex items-center space-x-2">
                      <Camera className="h-5 w-5" />
                      <span className="font-medium">{detection.animal}</span>
                    </div>
                    <div className="flex items-center space-x-2">
                      <span className={`px-2 py-1 rounded-full text-sm ${
                        detection.status === 'Active'
                          ? 'bg-red-100 text-red-800'
                          : 'bg-green-100 text-green-800'
                      }`}>
                        {detection.status}
                      </span>
                      {detection.status === 'Active' && (
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            handleClearDetection(detection.id);
                          }}
                          className="p-1 hover:bg-red-100 rounded-full"
                        >
                          <X className="h-4 w-4 text-red-600" />
                        </button>
                      )}
                    </div>
                  </div>
                  <div className="mt-2 text-sm text-gray-600">
                    <p>Location: {detection.location}</p>
                    <p>Confidence: {(detection.confidence * 100).toFixed(1)}%</p>
                    <p>Time: {new Date(detection.timestamp).toLocaleString()}</p>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>

        {/* Selected Detection Details */}
        {selectedDetection && (
          <Card>
            <CardHeader>
              <CardTitle>Detection Details</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
              <div className="aspect-video rounded-lg overflow-hidden bg-gray-100">
                  {selectedDetection.image_path ? (
                    <img 
                      src={`http://localhost:5000/api/images/${encodeURIComponent(selectedDetection.image_path)}`}
                      alt="Detection frame" 
                      className="w-full h-full object-contain"
                      onError={(e) => {
                        e.target.onerror = null;
                        e.target.src = ''; // Clear the broken image
                        e.target.alt = 'Failed to load image';
                      }}
                    />
                  ) : (
                    <div className="w-full h-full flex items-center justify-center text-gray-500">
                      No image available
                    </div>
                  )}
                </div>

                <div className="text-sm space-y-2">
                  <p><strong>Animal:</strong> {selectedDetection.animal}</p>
                  <p><strong>Location:</strong> {selectedDetection.location}</p>
                  <p><strong>Confidence:</strong> {(selectedDetection.confidence * 100).toFixed(1)}%</p>
                  <p><strong>Time:</strong> {new Date(selectedDetection.timestamp).toLocaleString()}</p>
                  <p><strong>Status:</strong> {selectedDetection.status}</p>
                </div>
              </div>
            </CardContent>
          </Card>
        )}
      </div>
    </div>
  );
};

export default AnimalDetectionDashboard;