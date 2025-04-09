import cv2
import os
import datetime
import numpy as np
import pandas as pd
import json
import re
import pickle
from flask import Flask, render_template, Response, request, redirect, url_for
import threading
from PIL import Image
import mediapipe as mp
import math
from scipy.spatial import distance as dist

app = Flask(__name__)

# Constants
MIN_CONFIDENCE = 0.1
TEMPORAL_THRESHOLD = 3
FRAME_SKIP = 4
EAR_THRESHOLD = 0.25  # Eye Aspect Ratio threshold for blink detection
EAR_CONSECUTIVE_FRAMES = 3  # Number of consecutive frames with low EAR to count as a blink
BLINK_REQUIRED = False

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Face mesh indices for eyes
LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
# Key points for EAR calculation (vertical and horizontal points)
LEFT_EYE_POINTS = [362, 385, 387, 373, 380, 374]
RIGHT_EYE_POINTS = [33, 160, 158, 133, 153, 144]

# Create directories and files
if not os.path.exists("known_faces"):
    os.makedirs("known_faces")
if not os.path.exists("models"):
    os.makedirs("models")

attendance_file = "attendance.csv"
# Define the expected header with 6 columns
expected_header = "Name,Registration Number,Time,Confidence,Blink Detected,Motion Verified\n"

# Check and fix the attendance.csv file
if not os.path.exists(attendance_file):
    with open(attendance_file, "w") as f:
        f.write(expected_header)
else:
    # Check if the file has the old 4-column header
    with open(attendance_file, "r") as f:
        first_line = f.readline().strip()
        if first_line != expected_header.strip():
            # Read the existing data
            try:
                df = pd.read_csv(attendance_file)
                # Add missing columns with default values
                if 'Blink Detected' not in df.columns:
                    df['Blink Detected'] = False
                if 'Motion Verified' not in df.columns:
                    df['Motion Verified'] = False
                # Write back with the correct header
                df.to_csv(attendance_file, index=False)
                print("Updated attendance.csv to include Blink Detected and Motion Verified columns.")
            except Exception as e:
                print(f"Error fixing attendance.csv: {str(e)}. Recreating file.")
                with open(attendance_file, "w") as f:
                    f.write(expected_header)

students_file = "students.json"
if not os.path.exists(students_file):
    with open(students_file, "w") as f:
        json.dump({}, f)

embeddings_file = "models/face_embeddings.pkl"
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Use cv2.face_LBPHFaceRecognizer.create() from opencv-contrib-python
try:
    face_recognizer = cv2.face_LBPHFaceRecognizer.create()
except AttributeError as e:
    print("Error: LBPHFaceRecognizer not found. Ensure you have installed 'opencv-contrib-python'.")
    print("Run: pip install opencv-contrib-python")
    raise e

model_file = "models/face_model.yml"
recognizer_loaded = False

students = {}
label_to_id = {}
id_to_label = {}
next_label = 0

def load_students():
    global students
    try:
        with open(students_file, "r") as f:
            students = json.load(f)
    except:
        students = {}

def save_students():
    with open(students_file, "w") as f:
        json.dump(students, f, indent=4)

def load_face_recognizer():
    global face_recognizer, recognizer_loaded, label_to_id, id_to_label, next_label
    if os.path.exists(model_file):
        try:
            face_recognizer.read(model_file)
            recognizer_loaded = True
            if os.path.exists(embeddings_file):
                with open(embeddings_file, 'rb') as f:
                    data = pickle.load(f)
                    label_to_id = data.get('label_to_id', {})
                    id_to_label = data.get('id_to_label', {})
                    next_label = data.get('next_label', 0)
            print("Face recognizer model loaded successfully")
            print(f"Loaded mappings: {label_to_id}, {id_to_label}")
        except Exception as e:
            print(f"Error loading face recognizer: {str(e)}")
            recognizer_loaded = False
    else:
        print("No face recognizer model found yet. Register a student to train.")
        recognizer_loaded = False

def train_face_recognizer():
    global face_recognizer, recognizer_loaded, label_to_id, id_to_label, next_label
    faces = []
    labels = []
    
    for reg_number, label in label_to_id.items():
        dir_path = os.path.join("known_faces", reg_number)
        if os.path.exists(dir_path):
            for img_name in os.listdir(dir_path):
                img_path = os.path.join(dir_path, img_name)
                try:
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        faces.append(img)
                        labels.append(label)
                except Exception as e:
                    print(f"Error loading image {img_path}: {str(e)}")
    
    if len(faces) > 0 and len(labels) > 0:
        print(f"Training with {len(faces)} images")
        try:
            face_recognizer.train(faces, np.array(labels))
            face_recognizer.write(model_file)
            print(f"Model saved to {model_file}")
        except Exception as e:
            print(f"Error training or saving model: {str(e)}")
        data = {
            'label_to_id': label_to_id,
            'id_to_label': id_to_label,
            'next_label': next_label
        }
        with open(embeddings_file, 'wb') as f:
            pickle.dump(data, f)
        recognizer_loaded = True
        print("Face recognizer trained successfully")
        load_face_recognizer()
    else:
        print("No faces found for training")

load_students()
load_face_recognizer()

cap = None
running = threading.Event()
last_attendance = {}
recognition_buffer = {}
registering = threading.Event()
register_count = 0
register_max_samples = 50
register_data = {}
face_display_duration = {}  # To store face coordinates and timestamps

# Blink detection variables
blink_counter = 0
ear_consecutive_frames = 0
blink_detected = False
blink_start_time = None
blink_in_progress = False
last_blink_time = None
# For motion detection
prev_gray = None
motion_verified = False

def get_available_camera():
    global cap
    index = 0
    while index < 20:
        print(f"Trying camera index {index}...")
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            print(f"Camera found at index {index}")
            return cap, index
        cap.release()
        index += 1
    print("No camera found after trying indices 0 to 19")
    return None, -1

def preprocess_face(frame, face_rect):
    x, y, w, h = face_rect
    face = frame[y:y+h, x:x+w]
    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (100, 100))
    equalized = cv2.equalizeHist(resized)
    return equalized

def calculate_ear(eye_landmarks):
    """
    Calculate Eye Aspect Ratio (EAR) using the 6 eye landmarks.
    EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
    """
    # Vertical eye landmarks (top to bottom)
    A = dist.euclidean(eye_landmarks[1], eye_landmarks[5])
    B = dist.euclidean(eye_landmarks[2], eye_landmarks[4])
    # Horizontal eye landmarks (left to right)
    C = dist.euclidean(eye_landmarks[0], eye_landmarks[3])
    
    # Calculate EAR
    ear = (A + B) / (2.0 * C)
    
    # Handle potential division by zero
    if C == 0:
        return 1.0
    
    return ear

def detect_blink(landmarks):
    """
    Enhanced blink detection using Eye Aspect Ratio (EAR) method.
    Returns True if a blink is detected, False otherwise.
    Also updates blink_counter and blink_detected global variables.
    """
    global blink_counter, ear_consecutive_frames, blink_detected, blink_start_time, blink_in_progress, last_blink_time
    
    # If no landmarks detected, return current blink status
    if not landmarks:
        return blink_detected
    
    # Get current time for blink timing
    current_time = datetime.datetime.now()
    
    # Extract eye landmarks
    try:
        left_eye_points = [landmarks[point] for point in LEFT_EYE_POINTS]
        right_eye_points = [landmarks[point] for point in RIGHT_EYE_POINTS]
        
        # Calculate EAR for both eyes
        left_ear = calculate_ear(left_eye_points)
        right_ear = calculate_ear(right_eye_points)
        
        # Average EAR from both eyes
        ear = (left_ear + right_ear) / 2.0
        
        # Detect blink stages
        if ear < EAR_THRESHOLD:
            ear_consecutive_frames += 1
            
            # Start tracking a potential blink
            if not blink_in_progress and ear_consecutive_frames >= 1:
                blink_in_progress = True
                blink_start_time = current_time
        else:
            # If eyes were closed for enough consecutive frames, count as a blink
            if blink_in_progress and ear_consecutive_frames >= EAR_CONSECUTIVE_FRAMES:
                blink_counter += 1
                blink_detected = True
                last_blink_time = current_time
                print(f"Blink #{blink_counter} detected at {current_time}")
                
                # Reset blink tracking
                blink_in_progress = False
                ear_consecutive_frames = 0
            elif blink_in_progress:
                # Reset if eyes opened too quickly (not a real blink)
                blink_in_progress = False
                ear_consecutive_frames = 0
            else:
                # Just reset counter if no blink in progress
                ear_consecutive_frames = 0
        
        # Reset blink detection after 30 seconds to require new blinks for verification
        if last_blink_time and (current_time - last_blink_time).total_seconds() > 30:
            if blink_counter > 0:  # Only reset if we've detected at least one blink
                print("Resetting blink detection after 30 seconds of inactivity")
                blink_counter = 0
                blink_detected = False
                last_blink_time = None
    
    except Exception as e:
        print(f"Error in blink detection: {str(e)}")
    
    return blink_detected

def calculate_optical_flow(prev, curr):
    global motion_verified
    flow = cv2.calcOpticalFlowFarneback(prev, curr, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    avg_magnitude = np.mean(mag)
    motion_verified = avg_magnitude > 0.5
    return motion_verified

def process_face_mesh(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)
    landmarks_dict = {}
    
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
            )
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_IRISES,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style()
            )
            for idx, landmark in enumerate(face_landmarks.landmark):
                h, w, _ = frame.shape
                x, y = int(landmark.x * w), int(landmark.y * h)
                landmarks_dict[idx] = (x, y)
    
    return frame, landmarks_dict

def gen_frames():
    global cap, recognizer_loaded, blink_counter, blink_detected, prev_gray, motion_verified, last_attendance, face_display_duration
    frame_count = 0
    
    while running.is_set() and cap and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame from camera")
            break

        frame_count += 1
        frame = cv2.resize(frame, (640, 480))
        
        if prev_gray is None:
            prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            continue
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if frame_count % 5 == 0:
            motion_detected = calculate_optical_flow(prev_gray, gray)
            prev_gray = gray.copy()
        
        processed_frame, landmarks = process_face_mesh(frame)
        
        if landmarks:
            is_blink = detect_blink(landmarks)
            cv2.putText(processed_frame, f"Blinks: {blink_counter}", (20, 80), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            if blink_detected:
                cv2.putText(processed_frame, "BLINK DETECTED", (20, 110), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        if motion_verified:
            cv2.putText(processed_frame, "MOTION VERIFIED", (20, 140), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        if frame_count % FRAME_SKIP != 0:
            # Draw any persistent face boxes before yielding the frame
            current_time = datetime.datetime.now()
            for reg_number, (coords, timestamp) in list(face_display_duration.items()):
                if (current_time - timestamp).total_seconds() <= 3:
                    x, y, w, h = coords
                    name = students.get(reg_number, "Unknown")
                    label_text = f"{name} ({reg_number.replace('_', '/')})"
                    cv2.rectangle(processed_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(processed_frame, label_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                else:
                    del face_display_duration[reg_number]  # Remove expired entries
            
            _, buffer = cv2.imencode('.jpg', processed_frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            continue
        
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)
        print(f"Detected {len(faces)} faces in attendance frame")
        
        current_time = datetime.datetime.now()
        detected_faces = {}  # Temporary storage for current frame detections
        
        for (x, y, w, h) in faces:
            if recognizer_loaded:
                face_img = preprocess_face(frame, (x, y, w, h))
                try:
                    label, confidence = face_recognizer.predict(face_img)
                    confidence = (100 - confidence) / 100
                    print(f"Predicted label: {label}, Confidence: {confidence:.2f}, Recognizer loaded: {recognizer_loaded}")
                    if confidence > MIN_CONFIDENCE and label in id_to_label:
                        reg_number = id_to_label[label]
                        recognition_buffer.setdefault(reg_number, 0)
                        recognition_buffer[reg_number] += 1
                        
                        print(f"Recognition buffer for {reg_number}: {recognition_buffer[reg_number]}")
                        if recognition_buffer[reg_number] >= TEMPORAL_THRESHOLD:
                            authentication_passed = (not BLINK_REQUIRED) or blink_detected
                            
                            print(f"Authentication passed: {authentication_passed}, Blink detected: {blink_detected}")
                            if authentication_passed and (reg_number not in last_attendance or 
                                                         (current_time - last_attendance[reg_number]).seconds > 300):
                                name = students.get(reg_number, "Unknown")
                                with open(attendance_file, "a") as f:
                                    f.write(f"{name},{reg_number.replace('_', '/')},{current_time.strftime('%Y-%m-%d %H:%M:%S')},{confidence:.2f},{blink_detected},{motion_verified}\n")
                                last_attendance[reg_number] = current_time
                                recognition_buffer[reg_number] = 0  # Reset buffer after recording
                                print(f"Attendance recorded for {name} ({reg_number}) with confidence {confidence:.2f}")
                                blink_detected = False
                                blink_counter = 0
                        
                        # Store face data for persistent display
                        detected_faces[reg_number] = ((x, y, w, h), current_time)
                        color = (0, 255, 0)
                        name = students.get(reg_number, "Unknown")
                        label_text = f"{name} ({reg_number.replace('_', '/')}) ({confidence:.2f})"
                    else:
                        reason = "Confidence too low" if confidence <= MIN_CONFIDENCE else "Label not in id_to_label"
                        print(f"Recognition failed: {reason}, Confidence: {confidence:.2f}, Label: {label}")
                        color = (0, 0, 255)
                        label_text = f"Unknown (Conf: {confidence:.2f})"
                except Exception as e:
                    print(f"Prediction error: {str(e)}")
                    color = (0, 0, 255)
                    label_text = "Error"
            else:
                color = (255, 0, 0)
                label_text = "Train model first"
            
            cv2.rectangle(processed_frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(processed_frame, label_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        # Update face_display_duration with current detections
        face_display_duration.update(detected_faces)

        # Draw any persistent face boxes that are still within 3 seconds
        for reg_number, (coords, timestamp) in list(face_display_duration.items()):
            if (current_time - timestamp).total_seconds() <= 3:
                x, y, w, h = coords
                name = students.get(reg_number, "Unknown")
                label_text = f"{name} ({reg_number.replace('_', '/')})"
                cv2.rectangle(processed_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(processed_frame, label_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            else:
                del face_display_duration[reg_number]  # Remove expired entries

        _, buffer = cv2.imencode('.jpg', processed_frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

def gen_register_frames():
    global cap, register_count, registering, register_data
    while registering.is_set() and cap and cap.isOpened() and register_count < register_max_samples:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame from camera during registration")
            break

        frame = cv2.resize(frame, (640, 480))
        processed_frame, landmarks = process_face_mesh(frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)
        print(f"Detected {len(faces)} faces in registration frame")
        
        for (x, y, w, h) in faces:
            face_img = preprocess_face(frame, (x, y, w, h))
            reg_number_dir = register_data.get('reg_number_dir')
            student_dir = os.path.join("known_faces", reg_number_dir)
            if not os.path.exists(student_dir):
                os.makedirs(student_dir)
            img_path = os.path.join(student_dir, f"{reg_number_dir}_{register_count}.jpg")
            cv2.imwrite(img_path, face_img)
            register_count += 1
            print(f"Saved image {register_count}/{register_max_samples}")
            break
        
        cv2.putText(processed_frame, f"Capturing: {register_count}/{register_max_samples}", (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        _, buffer = cv2.imencode('.jpg', processed_frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    
    if register_count >= register_max_samples:
        registering.clear()
        students[register_data['reg_number_dir']] = register_data['name']
        save_students()
        train_face_recognizer()

@app.route('/')
def welcome():
    return render_template('welcome.html')

@app.route('/home')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start', methods=['POST'])
def start():
    global cap, last_attendance
    running.set()
    last_attendance.clear()  # Reset last_attendance on start
    cap, _ = get_available_camera()
    if cap is None:
        return "No camera detected!", 500
    return redirect(url_for('index'))

@app.route('/stop', methods=['POST'])
def stop():
    global cap
    running.clear()
    if cap:
        cap.release()
    return redirect(url_for('index'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    global next_label, label_to_id, id_to_label, register_count, registering, register_data, cap
    
    if request.method == 'POST':
        name = request.form.get('name')
        reg_number = request.form.get('reg_number')

        if not name or not name.strip():
            return "Name cannot be empty!", 400
        if not reg_number or not reg_number.strip():
            return "Registration number cannot be empty!", 400

        if not re.match(r'^D/BCE/23/00(0[3-9]|1[0-6])$', reg_number):
            return "Invalid registration number! Must be in format D/BCE/23/0003 to D/BCE/23/0016.", 400

        reg_number_dir = reg_number.replace('/', '_')
        if reg_number_dir not in label_to_id:
            label_to_id[reg_number_dir] = next_label
            id_to_label[next_label] = reg_number_dir
            next_label += 1
            print(f"Label mappings updated: {label_to_id}, {id_to_label}")

        register_data = {'name': name, 'reg_number_dir': reg_number_dir}
        register_count = 0
        registering.set()
        cap, _ = get_available_camera()
        if cap is None:
            return "No camera detected!", 500
        
        return render_template('register.html')
    
    return redirect(url_for('index'))

@app.route('/register_feed')
def register_feed():
    return Response(gen_register_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stop_register', methods=['POST'])
def stop_register():
    global cap, registering
    registering.clear()
    if cap:
        cap.release()
    return redirect(url_for('index'))

@app.route('/report')
def report():
    try:
        df = pd.read_csv(attendance_file)
        if df.empty:
            return render_template('report.html', records=[])
        # Ensure all expected columns are present
        expected_columns = ['Name', 'Registration Number', 'Time', 'Confidence', 'Blink Detected', 'Motion Verified']
        for col in expected_columns:
            if col not in df.columns:
                df[col] = False if col in ['Blink Detected', 'Motion Verified'] else ''
        records = df.to_dict('records')
        return render_template('report.html', records=records)
    except pd.errors.EmptyDataError:
        return render_template('report.html', records=[])
    except Exception as e:
        print(f"Error in report route: {str(e)}")
        return render_template('report.html', records=[])

@app.route('/export_csv')
def export_csv():
    try:
        # Read the attendance file
        df = pd.read_csv(attendance_file)
        
        # Create the response
        output = df.to_csv(index=False)
        
        # Create the response with appropriate headers
        response = Response(
            output,
            mimetype='text/csv',
            headers={
                "Content-Disposition": f"attachment;filename=attendance_export_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            }
        )
        return response
    except Exception as e:
        print(f"Error exporting CSV: {str(e)}")
        return redirect(url_for('report'))

@app.route('/export_pdf')
def export_pdf():
    try:
        # Read the attendance file
        df = pd.read_csv(attendance_file)
        
        # Create a styled HTML table
        html = '''
        <html>
            <head>
                <style>
                    body { font-family: Arial, sans-serif; }
                    table { width: 100%; border-collapse: collapse; margin: 25px 0; }
                    th, td { border: 1px solid #ddd; padding: 12px; text-align: left; }
                    th { background-color: #4CAF50; color: white; }
                    tr:nth-child(even) { background-color: #f2f2f2; }
                    h1 { text-align: center; color: #333; }
                    .timestamp { text-align: right; color: #666; margin: 20px; }
                </style>
            </head>
            <body>
                <h1>Attendance Report</h1>
                <div class="timestamp">Generated on: ''' + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') + '''</div>
        '''
        
        # Convert DataFrame to HTML table with styling
        html += df.to_html(classes='table', index=False)
        html += '''
            </body>
        </html>
        '''
        
        # Convert HTML to PDF using pdfkit
        try:
            import pdfkit
            pdf = pdfkit.from_string(html, False)
        except ImportError:
            return "Error: pdfkit not installed. Please install wkhtmltopdf and pdfkit.", 500
        except Exception as e:
            return f"Error generating PDF: {str(e)}", 500
        
        # Create the response with appropriate headers
        response = Response(
            pdf,
            mimetype='application/pdf',
            headers={
                "Content-Disposition": f"attachment;filename=attendance_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
            }
        )
        return response
    except Exception as e:
        print(f"Error exporting PDF: {str(e)}")
        return redirect(url_for('report'))

@app.route('/manage')
def manage():
    return render_template('manage.html', students=students)

@app.route('/delete', methods=['POST'])
def delete():
    global label_to_id, id_to_label
    
    reg_number = request.form.get('reg_number')
    if reg_number:
        student_dir = os.path.join("known_faces", reg_number)
        if os.path.exists(student_dir):
            for file in os.listdir(student_dir):
                os.remove(os.path.join(student_dir, file))
            os.rmdir(student_dir)
            
            if reg_number in label_to_id:
                label = label_to_id[reg_number]
                del label_to_id[reg_number]
                if label in id_to_label:
                    del id_to_label[label]
            
            if reg_number in students:
                del students[reg_number]
                save_students()
            
            train_face_recognizer()
    
    return redirect(url_for('manage'))

@app.route('/retrain', methods=['POST'])
def retrain():
    train_face_recognizer()
    return redirect(url_for('manage'))

@app.route('/toggle_blink_requirement', methods=['POST'])
def toggle_blink_requirement():
    global BLINK_REQUIRED
    BLINK_REQUIRED = not BLINK_REQUIRED
    return redirect(url_for('manage'))

@app.route('/settings')
def settings():
    global BLINK_REQUIRED
    return render_template('settings.html', blink_required=BLINK_REQUIRED)

@app.route('/debug')
def debug():
    return f"Students: {students}<br>Label to ID: {label_to_id}<br>ID to Label: {id_to_label}<br>Recognizer Loaded: {recognizer_loaded}"

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)