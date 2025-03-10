import os
from flask import Flask, render_template, Response, jsonify
import cv2
import numpy as np
from deepface import DeepFace
from mtcnn import MTCNN

app = Flask(__name__)

# Initialize webcam
cap = cv2.VideoCapture(0)
detector = MTCNN()

# Global variables to store analysis results
latest_results = {"age": "Detecting...", "gender": "Detecting...", "emotion": "Detecting..."}

def generate_frames():
    global latest_results

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        faces = detector.detect_faces(frame)

        for face in faces:
            x, y, w, h = face["box"]
            face_roi = frame[y:y + h, x:x + w]

            try:
                result = DeepFace.analyze(face_roi, actions=["age", "gender", "emotion"], enforce_detection=False)
                latest_results = {
                    "age": result[0].get("age", "N/A"),
                    "gender": result[0].get("dominant_gender", "N/A"),
                    "emotion": result[0].get("dominant_emotion", "N/A")
                }

                # Draw bounding box and labels
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, f"Age: {latest_results['age']}", (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(frame, f"Gender: {latest_results['gender']}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(frame, f"Emotion: {latest_results['emotion']}", (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            except Exception as e:
                print("Error:", str(e))

        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/get_facial_data')
def get_facial_data():
    """API endpoint to provide real-time facial analysis results."""
    return jsonify(latest_results)


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))  # Required for Cloud Run
    app.run(debug=True, host="0.0.0.0", port=port)
