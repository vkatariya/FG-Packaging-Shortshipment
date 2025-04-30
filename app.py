from ultralytics import YOLO
import cv2
from flask import Flask, render_template, Response, request, redirect, url_for, jsonify
import os
from datetime import datetime
import base64
import threading
import time
import random
import pandas as pd
import winsound
import openpyxl
import queue
import signal
import sys
import numpy as np

app = Flask(__name__, template_folder='templates', static_folder='static')
#DIRECTORY_PATH = "C:/Users/tv239/Downloads/SHIPMENT/"
DIRECTORY_PATH = os.path.dirname(os.path.abspath(__file__))
EXCEL_FILE_PATH = os.path.join(DIRECTORY_PATH, "data.xlsx")

# Ensure required directories exist
os.makedirs(os.path.join(DIRECTORY_PATH, '/static/CaptureImages'), exist_ok=True)
os.makedirs(os.path.join(DIRECTORY_PATH, '/static/predict'), exist_ok=True)

# Load YOLO model
model = YOLO(DIRECTORY_PATH + "/weights/best_v3.pt")

# Global variables for thread synchronization and camera handling
camera_running = True
frame_queue = queue.Queue(maxsize=1)
capture_lock = threading.Lock()

def signal_handler(sig, frame):
    global camera_running
    print("Shutting down gracefully...")
    camera_running = False
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

def read_excel_data():
    """Read data from the Excel file."""
    if os.path.exists(EXCEL_FILE_PATH):
        df = pd.read_excel(EXCEL_FILE_PATH)
        return df.to_dict(orient='records')
    return []

def write_excel_data(data):
    """Write data to the Excel file."""
    df = pd.DataFrame(data)
    df.to_excel(EXCEL_FILE_PATH, index=False)

def camera_capture(save_path):
    """Production-ready camera capture function"""
    global camera_running
    image_path = os.path.join(save_path, "camera_image.jpg")
    
    # For production, we'll use a dummy frame if camera is not available
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(dummy_frame, 'Camera Unavailable', (50, 240),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    while camera_running:
        try:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                # Use dummy frame when camera is not available
                frame_queue.put(dummy_frame)
                time.sleep(1)
                continue

            cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
            cap.set(cv2.CAP_PROP_FOCUS, 100)
            cap.set(3, 1280)

            while camera_running:
                ret, frame = cap.read()
                if not ret:
                    print("Error reading frame. Reconnecting camera...")
                    break

                # Update the frame in the queue
                try:
                    if frame_queue.full():
                        frame_queue.get_nowait()  # Remove old frame
                    frame_queue.put_nowait(frame)
                except queue.Full:
                    pass

                with capture_lock:
                    cv2.imwrite(image_path, frame)

                time.sleep(0.1)  # Reduce CPU usage

        except Exception as e:
            print(f"Camera error: {str(e)}")
            frame_queue.put(dummy_frame)
            time.sleep(1)
        finally:
            if cap is not None:
                cap.release()

# Start the camera capture in a separate thread
capture_thread = threading.Thread(
    target=camera_capture,
    args=(os.path.join(DIRECTORY_PATH, '/static/CaptureImages'),),
    daemon=True
)
capture_thread.start()

# Function to generate frames for video streaming
def generate_frames():
    while True:
        try:
            frame = frame_queue.get(timeout=1.0)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_data = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n')
        except queue.Empty:
            continue
        except Exception as e:
            print(f"Frame generation error: {str(e)}")
            time.sleep(0.1)

@app.route('/')
def index():
    data = read_excel_data()
    return render_template('index.html', object_count=0, selected_number=16, status="Empty", fill_count=0, empty_count=0, data=data)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/capture', methods=['POST'])
def capture():
    try:
        selected_number = int(request.form.get('number'))
        image_path = os.path.join(DIRECTORY_PATH, '/static/CaptureImages/camera_image.jpg')
        
        if not os.path.exists(image_path):
            return jsonify(error="Image file not found"), 404

        # Add timeout for file reading
        start_time = time.time()
        while time.time() - start_time < 5:  # 5 second timeout
            try:
                with capture_lock:
                    frame = cv2.imread(image_path)
                if frame is not None:
                    break
                time.sleep(0.1)
            except Exception as e:
                print(f"Error reading image: {str(e)}")
                continue
        
        if frame is None:
            return jsonify(error="Failed to read image"), 500

        # Process image with timeout
        try:
            results = model(frame, conf=0.6)
            fill_count = 0
            empty_count = 0

            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    class_id = int(box.cls[0])
                    class_name = model.names[class_id]
                    color = (0, 0, 255) if class_name == "empty" else (0, 255, 0)
                    
                    if class_name == "empty":
                        empty_count += 1
                    else:
                        fill_count += 1

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                    cv2.putText(frame, f'{class_name} {box.conf.item():.2f}',
                              (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            status = "Completed" if fill_count == selected_number else "Not Complete"
            
            # Save results with error handling
            if status == "Completed":
                try:
                    unique_filename = datetime.now().strftime("%Y%m%d") + "_" + str(random.randint(1000, 9999)) + ".jpg"
                    predict_folder = os.path.join(DIRECTORY_PATH, '/static/predict')
                    os.makedirs(predict_folder, exist_ok=True)
                    cv2.imwrite(os.path.join(predict_folder, unique_filename), frame)

                    data = read_excel_data()
                    data.append({
                        'id': len(data) + 1,
                        'status': status,
                        'fill_quantity': fill_count,
                        'empty_quantity': empty_count,
                        'datetime': datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
                        'image_path': unique_filename
                    })
                    write_excel_data(data)
                except Exception as e:
                    print(f"Error saving results: {str(e)}")

            # Encode image with error handling
            _, buffer = cv2.imencode('.jpg', frame)
            image_base64 = base64.b64encode(buffer).decode('utf-8')

            return jsonify(
                object_count=fill_count + empty_count,
                selected_number=selected_number,
                status=status,
                fill_count=fill_count,
                empty_count=empty_count,
                play_alarm=empty_count == 1,
                image_base64=image_base64
            )

        except Exception as e:
            return jsonify(error=f"Processing error: {str(e)}"), 500

    except Exception as e:
        return jsonify(error=f"Capture error: {str(e)}"), 500

@app.route('/confirm', methods=['POST'])
def confirm():
    # Dummy confirmation action for testing with Excel file
    return redirect(url_for('index'))

@app.route('/fetch_records')
def fetch_records():
    data = read_excel_data()
    # Ensure the data structure matches the expected format
    formatted_data = [
        {
            'id': record.get('id', ''),
            'status': record.get('status', ''),
            'fill_quantity': record.get('fill_quantity', ''),
            'empty_quantity': record.get('empty_quantity', ''),
            'datetime': record.get('datetime', ''),
            'image_path': record.get('image_path', '')
        }
        for record in data
    ]
    return jsonify(formatted_data)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
