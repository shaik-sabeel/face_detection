# backend/app.py
from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit
import base64
import numpy as np
import cv2
import mediapipe as mp
import time
from gevent import monkey
monkey.patch_all() # Patch for Flask-SocketIO with gevent

# Initialize MediaPipe Face Mesh (loaded once when app starts)
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1,
                                 min_detection_confidence=0.6, min_tracking_confidence=0.6) # Increased confidence for accuracy

# --- Configuration Parameters for Focus Logic ---
YAW_THRESHOLD = 20        # degrees (e.g., looking significantly left or right)
PITCH_THRESHOLD = 15      # degrees (e.g., looking significantly up or down)
EAR_THRESHOLD = 0.23      # For eye closure (tune this based on observed EAR values)
EAR_CONSEC_FRAMES = 20    # Number of consecutive frames for EAR to trigger distraction (approx 1 second at ~20 FPS)
ALERT_DURATION = 10       # seconds alert stays on screen
COOLDOWN_PERIOD = 5       # seconds before a new alert can be triggered after one clears

# --- Camera Intrinsics for Head Pose (approximate for typical webcams) ---
dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion for simplicity

# 3D model points of a generic head, mapping to MediaPipe landmarks
model_points = np.array([
    (0.0, 0.0, 0.0),      # Nose tip (landmark 1)
    (0.0, -330.0, -65.0), # Chin (landmark 152)
    (-225.0, 170.0, -135.0), # Left eye inner corner (landmark 33)
    (225.0, 170.0, -135.0),  # Right eye inner corner (landmark 263)
    (-150.0, -150.0, -125.0),# Left mouth corner (landmark 61)
    (150.0, -150.0, -125.0)  # Right mouth corner (landmark 291)
], dtype="double")

# MediaPipe landmark indices for Eye Aspect Ratio (EAR) computation
LEFT_EYE_PTS = [
    33,  # P1 - outermost (left corner)
    160, # P2 - top-outer eyelid
    158, # P3 - top-inner eyelid
    133, # P4 - innermost (right corner of left eye)
    153, # P5 - bottom-inner eyelid
    144  # P6 - bottom-outer eyelid
]
RIGHT_EYE_PTS = [
    263, # P1 - outermost (right corner)
    387, # P2 - top-outer eyelid
    385, # P3 - top-inner eyelid
    362, # P4 - innermost (left corner of right eye)
    380, # P5 - bottom-inner eyelid
    373  # P6 - bottom-outer eyelid
]

# Global state variables for each connected user (using dictionary keyed by SocketID if multiple users)
# For a single user, these can be simpler globals. For multi-user, context.
# We'll treat this as single-user for now.
user_data = {
    "distracted_frames_ear": 0,
    "last_alert_trigger_time": 0, # Time when the alert was last *shown*
    "alert_active": False,
    "last_focus_time": time.time() # Time when the user was last explicitly 'focused'
}

def calculate_ear(eye_landmarks_coords):
    if len(eye_landmarks_coords) != 6:
        return 0.0 # Not enough points to calculate EAR

    p1 = np.array(eye_landmarks_coords[0])
    p2 = np.array(eye_landmarks_coords[1])
    p3 = np.array(eye_landmarks_coords[2])
    p4 = np.array(eye_landmarks_coords[3])
    p5 = np.array(eye_landmarks_coords[4])
    p6 = np.array(eye_landmarks_coords[5])

    A = np.linalg.norm(p2 - p6)
    B = np.linalg.norm(p3 - p5)
    C = np.linalg.norm(p1 - p4)

    if C != 0:
        ear = (A + B) / (2.0 * C)
    else:
        ear = 0.0
    return ear

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*") # Allows frontend from any origin to connect

@app.route('/')
def index():
    return "Backend server running. Connect with frontend client."

@socketio.on('connect')
def handle_connect():
    print(f"Client {request.sid} connected.")
    # Reset state for new connection or per-user state management
    user_data["distracted_frames_ear"] = 0
    user_data["last_alert_trigger_time"] = 0
    user_data["alert_active"] = False
    user_data["last_focus_time"] = time.time()

@socketio.on('disconnect')
def handle_disconnect():
    print(f"Client {request.sid} disconnected.")

@socketio.on('video_frame')
def handle_video_frame(data):
    # Decode the image from base64 string
    encoded_image = data.split(',')[1] # Remove "data:image/jpeg;base64," header
    decoded_image = base64.b64decode(encoded_image)
    np_arr = np.frombuffer(decoded_image, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if frame is None:
        return # Skip if frame couldn't be decoded

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    current_time = time.time()
    is_distracted_in_frame = False
    distraction_reason = ""
    
    image_h, image_w, _ = frame.shape

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # --- Head Pose Estimation ---
            img_coords = []
            # Make sure these indices match the model_points defined globally
            mp_indices = [1, 152, 33, 263, 61, 291]
            for idx in mp_indices:
                x = face_landmarks.landmark[idx].x * image_w
                y = face_landmarks.landmark[idx].y * image_h
                img_coords.append([x, y])
            image_points = np.array(img_coords, dtype="double")

            # Dynamically set camera matrix for solvePnP
            focal_length = image_w # Approximation
            camera_matrix = np.array([
                [focal_length, 0, image_w / 2],
                [0, focal_length, image_h / 2],
                [0, 0, 1]
            ], dtype="double")

            try:
                (success, rotation_vector, translation_vector) = cv2.solvePnP(
                    model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)

                if success:
                    R_matrix, _ = cv2.Rodrigues(rotation_vector)
                    sy = np.sqrt(R_matrix[0,0]**2 + R_matrix[1,0]**2)
                    singular = sy < 1e-6
                    if not singular:
                        pitch = np.degrees(np.arctan2(-R_matrix[2,0], sy))
                        yaw = np.degrees(np.arctan2(R_matrix[1,0], R_matrix[0,0]))
                        roll = np.degrees(np.arctan2(R_matrix[2,1], R_matrix[2,2]))
                    else: # Gimmbal lock
                        pitch = np.degrees(np.arctan2(-R_matrix[2,0], sy))
                        yaw = 0
                        roll = np.degrees(np.arctan2(-R_matrix[1,2], R_matrix[1,1]))
                    
                    # Check for head pose distraction
                    if abs(yaw) > YAW_THRESHOLD:
                        is_distracted_in_frame = True
                        distraction_reason += f"Yaw ({yaw:.1f}deg) "
                    if abs(pitch) > PITCH_THRESHOLD:
                        is_distracted_in_frame = True
                        distraction_reason += f"Pitch ({pitch:.1f}deg) "
            except cv2.error as e:
                # print(f"SolvePnP error: {e}")
                pass # Silently fail pose if not enough points, or error

            # --- Eye Aspect Ratio (EAR) Check ---
            left_eye_coords = []
            for idx in LEFT_EYE_PTS:
                l_mp = face_landmarks.landmark[idx]
                left_eye_coords.append((l_mp.x * image_w, l_mp.y * image_h))

            right_eye_coords = []
            for idx in RIGHT_EYE_PTS:
                r_mp = face_landmarks.landmark[idx]
                right_eye_coords.append((r_mp.x * image_w, r_mp.y * image_h))
            
            # Calculate EAR only if we have all 6 points for each eye
            if len(left_eye_coords) == 6 and len(right_eye_coords) == 6:
                left_ear = calculate_ear(left_eye_coords)
                right_ear = calculate_ear(right_eye_coords)
                avg_ear = (left_ear + right_ear) / 2.0

                if avg_ear < EAR_THRESHOLD:
                    user_data["distracted_frames_ear"] += 1
                    if user_data["distracted_frames_ear"] >= EAR_CONSEC_FRAMES:
                        is_distracted_in_frame = True
                        distraction_reason += f"Eyes Closed ({avg_ear:.2f}) "
                else:
                    user_data["distracted_frames_ear"] = 0 # Reset counter if eyes open

    else: # No face detected
        user_data["distracted_frames_ear"] = 0 # Reset EAR if no face
        is_distracted_in_frame = True
        distraction_reason += "No Face Detected "

    # --- Decision Logic and Alerting ---
    if is_distracted_in_frame:
        user_data["last_focus_time"] = current_time # Update 'last_focus_time' ONLY if it's currently focused. If distracted, we wait for refocus.

        # Trigger alert if not active and cooldown passed
        if not user_data["alert_active"] and (current_time - user_data["last_alert_trigger_time"]) > COOLDOWN_PERIOD:
            emit('alert', {'action': 'show_alert', 'reason': distraction_reason})
            user_data["last_alert_trigger_time"] = current_time
            user_data["alert_active"] = True
            print(f"[{request.sid}] Distraction Alert Triggered! Reason: {distraction_reason.strip()}")
        elif user_data["alert_active"] and (current_time - user_data["last_alert_trigger_time"]) < ALERT_DURATION:
            # If alert is active but hasn't run its course, continue to signal "active" (e.g., maintain UI)
            emit('alert', {'action': 'maintain_alert'})
    else: # Student is focused
        user_data["distracted_frames_ear"] = 0 # Reset consecutive ear frames on focus
        if user_data["alert_active"]:
            # If an alert is active, check if its duration has passed AND user has refocused recently
            if (current_time - user_data["last_alert_trigger_time"]) > ALERT_DURATION:
                 emit('alert', {'action': 'hide_alert'})
                 user_data["alert_active"] = False
                 print(f"[{request.sid}] Alert cleared: Student refocused.")
            else: # Alert still showing its course even if currently focused
                emit('alert', {'action': 'maintain_alert'})
        else: # All clear, emit a focused state to client (optional for continuous update)
            emit('alert', {'action': 'focused'})


if __name__ == '__main__':
    # Make sure to install gevent (part of requirements.txt) for a production-ready async server
    # Otherwise, the development server will be single-threaded.
    print("Starting Flask-SocketIO server...")
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)
    # 0.0.0.0 makes it accessible from other machines on network, useful for testing
    # For simple local use, '127.0.0.1' or just 'app' is fine