from ultralytics import YOLO
import cv2
from flask import Flask, render_template, Response, request, redirect, url_for, jsonify
import os
from datetime import datetime
import base64  # Import base64 for encoding image
import threading  # Import threading for continuous capture
import time  # Import time for sleep
import random  # Import random for generating random numbers
import winsound  # Import winsound for playing alarm sound

app = Flask(__name__, template_folder='templates', static_folder='static')

DIRECTORY_PATH = "C:/Users/tv239/Downloads/SHIPMENT/"

# Load YOLO model
model = YOLO(DIRECTORY_PATH + "weights/best_v3.pt")

def camera_capture(save_path):
    """Capture images from the camera every 10 seconds."""
    image_path = os.path.join(save_path, "camera_image.jpg")
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    cap = cv2.VideoCapture(0)  # Camera 0

    if not cap.isOpened():
        print("Error: Could not open the camera.")
        return

    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)  # 0 means manual focus
    cap.set(cv2.CAP_PROP_FOCUS, 100)  # Adjust the focus value
    cap.set(3, 1280)  # Set width

    capture_interval = 10  # Time delay for capturing image

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from the camera.")
            break

        cv2.imwrite(image_path, frame)  # Save frame from camera
        print(f"Image from camera saved at {image_path}")

        time.sleep(capture_interval)

    cap.release()
    cv2.destroyAllWindows()

# Start the camera capture in a separate thread
capture_thread = threading.Thread(target=camera_capture, args=(DIRECTORY_PATH + 'static/CaptureImages',))
capture_thread.start()

# Function to generate frames for video streaming and save captured images
def generate_frames():
    cap = cv2.VideoCapture(0)  # Capture from the default camera
    if not cap.isOpened():
        print("Error: Could not open the camera.")
        return

    while True:
        success, frame = cap.read()
        if not success:
            break

        # Encode frame as JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        # Yield frame data for streaming
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    # Dummy data for testing without database
    data = [
        {'id': 1, 'status': 'Completed', 'fill_quantity': 16, 'empty_quantity': 0, 'datetime': '01/01/2023 12:00:00', 'image_path': 'image1.jpg'},
        {'id': 2, 'status': 'Not Complete', 'fill_quantity': 10, 'empty_quantity': 6, 'datetime': '01/01/2023 12:10:00', 'image_path': 'image2.jpg'}
    ]
    return render_template('index.html', object_count=0, selected_number=16, status="Empty", fill_count=0, empty_count=0, data=data)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/capture', methods=['POST'])
def capture():
    selected_number = int(request.form.get('number'))
    print(f"Selected number: {selected_number}")  # Log the selected number
    object_count = 0
    fill_count = 0
    empty_count = 0

    # Read the saved image from the static/CaptureImages folder
    image_path = DIRECTORY_PATH + 'static/CaptureImages/camera_image.jpg'
    if not os.path.exists(image_path):
        return jsonify(error="Image file not found"), 404

    frame = cv2.imread(image_path)
    if frame is not None:
        results = model(frame, conf=0.6)  # Set confidence threshold to 0.6

        # Process YOLO results
        for i, result in enumerate(results):
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                class_id = int(box.cls[0])
                class_name = model.names[class_id]

                # Determine the color of the bounding box
                if class_name == "empty":
                    color = (0, 0, 255)  # Red color for "empty" class
                    empty_count += 1
                else:
                    color = (0, 255, 0)  # Green color for "fill" class
                    fill_count += 1

                object_count += 1  # Increment count for objects
                # Draw a bounding box and add class name and confidence score
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                cv2.putText(frame, f'{class_name} {box.conf.item():.2f}', (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Generate a unique filename based on the date and a random number
        unique_filename = datetime.now().strftime("%Y%m%d") + "_" + str(random.randint(1000, 9999)) + ".jpg"
        predict_folder = DIRECTORY_PATH + 'static/predict'
        if not os.path.exists(predict_folder):
            os.makedirs(predict_folder)
        save_path = os.path.join(predict_folder, unique_filename)

        # Save the processed frame only if fill_count equals selected_number
        if fill_count == selected_number:
            cv2.imwrite(save_path, frame)  # Save the processed frame

        # Encode the captured frame as base64
        _, buffer = cv2.imencode('.jpg', frame)
        image_base64 = base64.b64encode(buffer).decode('utf-8')

    status = "Completed" if fill_count == selected_number else "Not Complete"
    play_alarm = empty_count == 1

    # Play alarm sound if empty_count is exactly 1
    if play_alarm:
        alarm_sound = DIRECTORY_PATH + "static/alarm.wav"
        winsound.PlaySound(alarm_sound, winsound.SND_FILENAME | winsound.SND_LOOP)
        winsound.PlaySound(None, winsound.SND_PURGE)  # Stop the sound

    return jsonify(object_count=object_count, selected_number=selected_number, status=status, fill_count=fill_count, empty_count=empty_count, play_alarm=play_alarm, image_base64=image_base64)

@app.route('/confirm', methods=['POST'])
def confirm():
    # Dummy confirmation action for testing without database
    return redirect(url_for('index'))

@app.route('/fetch_records')
def fetch_records():
    # Dummy data for testing without database
    data = [
        {'id': 1, 'status': 'Completed', 'fill_quantity': 16, 'empty_quantity': 0, 'datetime': '01/01/2023 12:00:00', 'image_path': 'image1.jpg'},
        {'id': 2, 'status': 'Not Complete', 'fill_quantity': 10, 'empty_quantity': 6, 'datetime': '01/01/2023 12:10:00', 'image_path': 'image2.jpg'}
    ]
    return jsonify(data)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)  # Ensure the app runs on the correct host
