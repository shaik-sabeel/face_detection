# backend/app.py
from flask import Flask, request
from flask_socketio import SocketIO, emit
import base64
import numpy as np
import cv2
import mediapipe as mp
import time
from gevent import monkey
monkey.patch_all() # Required for Flask-SocketIO with gevent (asynchronous processing)

# Initialize MediaPipe Face Mesh (loaded once when app starts)
mp_face_mesh = mp.solutions.face_mesh
# Adjusted confidence for better detection across conditions
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1,
                                 min_detection_confidence=0.5, min_tracking_confidence=0.5) # Revert to 0.5 to make detection easier

# --- Configuration Parameters for Focus Logic ---
YAW_THRESHOLD = 20        # degrees: how much left/right turn is considered distracted
PITCH_THRESHOLD = 15      # degrees: how much up/down turn is considered distracted
EAR_THRESHOLD = 0.20      # Eye Aspect Ratio threshold for eye closure (slightly lowered for more sensitivity)
EAR_CONSEC_FRAMES = 15    # Number of consecutive frames eyes must be closed (0.75 sec at 20 FPS)
NO_FACE_CONSEC_FRAMES = 20 # Number of consecutive frames without a face (1 sec at 20 FPS)

ALERT_DISPLAY_MAX_DURATION = 10       # seconds: max alert display duration if student remains distracted
ALERT_COOLDOWN_PERIOD = 3       # seconds: Cooldown period before a new alert can be triggered after one clears (Reduced for testing)

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
    33, 160, 158, 133, 153, 144
]
RIGHT_EYE_PTS = [
    263, 387, 385, 362, 380, 373
]

# Global/per-user state variables for managing distraction and alerts.
# For this example, we'll treat it as single-user global.
user_session_state = {
    "ear_distracted_consecutive_frames": 0,
    "no_face_consecutive_frames": 0,
    "alert_is_active_on_client": False, # Is the "DISTRACTED!" alert currently showing on the frontend?
    "alert_activation_time": 0.0,       # Timestamp when the alert was last *shown*
    "alert_deactivation_time": 0.0      # Timestamp when the alert was last *hidden* (for cooldown)
}

# Helper function to calculate Eye Aspect Ratio (EAR)
def calculate_ear(eye_landmarks_coords):
    if len(eye_landmarks_coords) != 6:
        return 0.0 # Return 0 if not enough points to prevent errors

    p1, p2, p3, p4, p5, p6 = [np.array(pt) for pt in eye_landmarks_coords]

    A = np.linalg.norm(p2 - p6)
    B = np.linalg.norm(p3 - p5)
    C = np.linalg.norm(p1 - p4)

    if C != 0:
        ear = (A + B) / (2.0 * C)
    else:
        ear = 0.0
    return ear

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

@app.route('/')
def index():
    return "Backend server running. Connect with frontend client on port 8000 (usually)."

@socketio.on('connect')
def handle_connect():
    print(f"Client {request.sid} connected.")
    # Reset state for a new connection to ensure a clean start
    user_session_state["ear_distracted_consecutive_frames"] = 0
    user_session_state["no_face_consecutive_frames"] = 0
    user_session_state["alert_is_active_on_client"] = False
    user_session_state["alert_activation_time"] = 0.0
    user_session_state["alert_deactivation_time"] = 0.0

@socketio.on('disconnect')
def handle_disconnect():
    print(f"Client {request.sid} disconnected.")

@socketio.on('video_frame')
def handle_video_frame(data):
    current_time = time.time()

    try:
        encoded_image = data.split(',')[1] 
        decoded_image = base64.b64decode(encoded_image)
        np_arr = np.frombuffer(decoded_image, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is None:
            print("WARNING: Received empty frame. Skipping processing.")
            process_focus_logic(current_time, is_face_detected=False, head_pose_ok=False, ear_ok=False)
            return

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)
    except Exception as e:
        print(f"ERROR decoding or processing frame: {e}. Treating as no face detected.")
        process_focus_logic(current_time, is_face_detected=False, head_pose_ok=False, ear_ok=False)
        return

    # Flags for the current frame
    is_face_detected_in_frame = False
    head_pose_ok_in_frame = True # Assume OK unless deviated or error
    ear_ok_in_frame = True       # Assume OK unless closed eyes detected

    distraction_reasons = [] # Reasons for potential distraction in this specific frame
    
    current_yaw = 0
    current_pitch = 0
    current_avg_ear = 0

    if results.multi_face_landmarks:
        is_face_detected_in_frame = True
        user_session_state["no_face_consecutive_frames"] = 0 # Reset no-face counter

        for face_landmarks in results.multi_face_landmarks:
            image_h, image_w, _ = frame.shape

            # --- Head Pose Estimation ---
            img_coords = []
            mp_indices = [1, 152, 33, 263, 61, 291]
            for idx in mp_indices:
                l_mp = face_landmarks.landmark[idx]
                img_coords.append([l_mp.x * image_w, l_mp.y * image_h])
            image_points = np.array(img_coords, dtype="double")

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
                    
                    if sy < 1e-6:
                        current_pitch = np.degrees(np.arctan2(-R_matrix[2,0], sy))
                        current_yaw = 0
                        # current_roll = np.degrees(np.arctan2(-R_matrix[1,2], R_matrix[1,1]))
                    else:
                        current_pitch = np.degrees(np.arctan2(-R_matrix[2,0], sy))
                        current_yaw = np.degrees(np.arctan2(R_matrix[1,0], R_matrix[0,0]))
                        # current_roll = np.degrees(np.arctan2(R_matrix[2,1], R_matrix[2,2]))

                    # Check for head pose distraction
                    if abs(current_yaw) > YAW_THRESHOLD:
                        head_pose_ok_in_frame = False
                        distraction_reasons.append(f"Yaw({current_yaw:.1f}deg)")
                    if abs(current_pitch) > PITCH_THRESHOLD:
                        head_pose_ok_in_frame = False
                        distraction_reasons.append(f"Pitch({current_pitch:.1f}deg)")
                    # print(f"DEBUG: Yaw: {current_yaw:.1f}deg, Pitch: {current_pitch:.1f}deg")
                else:
                    head_pose_ok_in_frame = False
                    distraction_reasons.append("PoseUndetected")
            except cv2.error as e:
                # print(f"DEBUG: SolvePnP error for this frame: {e}")
                head_pose_ok_in_frame = False
                distraction_reasons.append("PoseError")
            
            # --- Eye Aspect Ratio (EAR) Check ---
            left_eye_coords = []
            for idx in LEFT_EYE_PTS:
                l_mp = face_landmarks.landmark[idx]
                left_eye_coords.append((l_mp.x * image_w, l_mp.y * image_h))

            right_eye_coords = []
            for idx in RIGHT_EYE_PTS:
                r_mp = face_landmarks.landmark[idx]
                right_eye_coords.append((r_mp.x * image_w, r_mp.y * image_h))
            
            if len(left_eye_coords) == 6 and len(right_eye_coords) == 6:
                left_ear = calculate_ear(left_eye_coords)
                right_ear = calculate_ear(right_eye_coords)
                current_avg_ear = (left_ear + right_ear) / 2.0
                
                # print(f"DEBUG: Avg EAR: {current_avg_ear:.2f}, Consec Frames: {user_session_state['ear_distracted_consecutive_frames']}")

                if current_avg_ear < EAR_THRESHOLD:
                    user_session_state["ear_distracted_consecutive_frames"] += 1
                    if user_session_state["ear_distracted_consecutive_frames"] >= EAR_CONSEC_FRAMES:
                        ear_ok_in_frame = False
                        distraction_reasons.append(f"Eyes Closed({current_avg_ear:.2f})")
                else:
                    user_session_state["ear_distracted_consecutive_frames"] = 0
            else:
                ear_ok_in_frame = False # Not enough eye points detected for EAR
                distraction_reasons.append("EyesUndetected")
                # Even if eye landmarks aren't perfectly found, if it's consistent, increment to trigger distraction.
                # Only increment if not caused by 'no face' already, otherwise it duplicates count purpose.
                user_session_state["ear_distracted_consecutive_frames"] += 1


    else: # No face detected in this frame
        user_session_state["no_face_consecutive_frames"] += 1
        is_face_detected_in_frame = False
        head_pose_ok_in_frame = True # Still "OK" until NO_FACE_CONSEC_FRAMES met
        ear_ok_in_frame = True # Still "OK" until NO_FACE_CONSEC_FRAMES met

        user_session_state["ear_distracted_consecutive_frames"] = 0 # Clear EAR if no face, primary 'no face' will handle
        distraction_reasons.clear() # Clear specific reasons, "no face" will be primary

    
    # --- Unified Focus Logic (decides 'is_currently_focused') ---
    # Determine the current state based on consolidated flags and counters
    is_currently_focused_this_frame = (is_face_detected_in_frame and head_pose_ok_in_frame and ear_ok_in_frame) and \
                                       (user_session_state["ear_distracted_consecutive_frames"] < EAR_CONSEC_FRAMES) and \
                                       (user_session_state["no_face_consecutive_frames"] < NO_FACE_CONSEC_FRAMES)


    # If no face is detected consistently, that's a distraction too. Add its reason late here for priority.
    if not is_face_detected_in_frame and user_session_state["no_face_consecutive_frames"] >= NO_FACE_CONSEC_FRAMES:
        if "NoFaceDetected" not in distraction_reasons: # Add if not already present
            distraction_reasons.append("NoFaceDetected")


    # --- Master Alert Control State Machine ---
    if not is_currently_focused_this_frame: # Student is NOT focused based on current frame analysis
        if not user_session_state["alert_is_active_on_client"]:
            # Alert is NOT currently active on the client. Check if cooldown period is over.
            if (current_time - user_session_state["alert_deactivation_time"]) >= ALERT_COOLDOWN_PERIOD:
                emit('alert', {'action': 'show_alert', 'reason': ", ".join(distraction_reasons) or "Distracted"})
                user_session_state["alert_is_active_on_client"] = True
                user_session_state["alert_activation_time"] = current_time
                print(f"[{request.sid}] --> ALERT TRIGGERED! Reasons: {', '.join(distraction_reasons) or 'Generic distraction'}")
            # else:
            #     print(f"[{request.sid}] Distraction detected, but in cooldown. Reason: {', '.join(distraction_reasons)}")
        else:
            # Alert IS already active on client. Check if it's exceeded its max display duration.
            if (current_time - user_session_state["alert_activation_time"]) > ALERT_DISPLAY_MAX_DURATION:
                emit('alert', {'action': 'hide_alert', 'message': 'Alert max duration exceeded while distracted.'})
                user_session_state["alert_is_active_on_client"] = False
                user_session_state["alert_deactivation_time"] = current_time # Reset for next alert trigger
                print(f"[{request.sid}] --> ALERT Auto-HIDDEN: Max duration exceeded while still distracted.")
            else:
                # Still within max duration for an active alert, just ensure client knows it's active.
                emit('alert', {'action': 'maintain_alert', 'reason': ", ".join(distraction_reasons) or "Distracted"})
                # print(f"[{request.sid}] ALERT ACTIVE (maintained). Reasons: {', '.join(distraction_reasons)}")

    else: # Student IS currently focused in this frame
        # Reset consecutive distraction counters if student is focused
        user_session_state["ear_distracted_consecutive_frames"] = 0
        user_session_state["no_face_consecutive_frames"] = 0
        
        if user_session_state["alert_is_active_on_client"]:
            # An alert IS currently active on the client, but student has refocused. Hide IMMEDIATELY.
            emit('alert', {'action': 'hide_alert', 'message': 'Student refocused!'})
            user_session_state["alert_is_active_on_client"] = False
            user_session_state["alert_deactivation_time"] = current_time # Mark when alert cleared for cooldown period
            print(f"[{request.sid}] --> ALERT CLEARED: Student refocused.")
        # else:
        #     # All clear, no alert showing. Can send 'focused' to client for UI status updates.
        #     emit('alert', {'action': 'focused', 'message': 'Currently focused.'})
        # print(f"[{request.sid}] Status: Focused. Yaw: {current_yaw:.1f}, Pitch: {current_pitch:.1f}, EAR: {current_avg_ear:.2f}")


if __name__ == '__main__':
    print("Starting Flask-SocketIO server...")
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)