import cv2
import mediapipe as mp
import numpy as np
import json
import os

# --- CONFIGURATION ---
VIDEO_PATH = 'input.mp4'  # REPLACE with your video filename
OUTPUT_VIDEO_PATH = 'output_overlay.mp4'
KEYPOINTS_PATH = 'keypoints.json'

# --- 1. SETUP MEDIAPIPE ---
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=2,        # Higher accuracy
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# --- HELPER: CALCULATE ANGLE ---
def calculate_angle(a, b, c):
    """Calculates angle ABC (in degrees) where b is the vertex."""
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle > 180.0:
        angle = 360-angle
        
    return int(angle)

# --- MAIN PROCESSING ---
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f"Error: Could not open video {VIDEO_PATH}")
    exit()

# Video Writer Setup
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = int(cap.get(cv2.CAP_PROP_FPS))
out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

keypoint_data = []
frame_count = 0

print("Processing video... this may take a moment.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    
    # Recolor to RGB for MediaPipe
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    
    # Make Detection
    results = pose.process(image)
    
    # Recolor back to BGR for OpenCV
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    frame_metrics = {"frame": frame_count}

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        
        # --- EXTRACT COORDINATES ---
        # Get coordinates for specific joints (assuming Right-handed batsman/bowler for simplicity)
        # You might need to swap 'LEFT' and 'RIGHT' depending on the player's side
        
        # Hip, Knee, Ankle (For Leg Stability)
        hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
        knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
        ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
        
        # Shoulder, Elbow, Wrist (For Arm Extension/Throwing)
        shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
        elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
        wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

        # Shoulder to Hip (Verticality/Balance)
        # We compare the torso vector to a vertical line
        
        # --- CALCULATE METRICS (Step 3 of Assignment) ---
        
        # Metric 1: Knee Flexion (Stability)
        knee_angle = calculate_angle(hip, knee, ankle)
        
        # Metric 2: Elbow Extension (Technique/Throwing)
        elbow_angle = calculate_angle(shoulder, elbow, wrist)
        
        # Metric 3: Body Lean (Forward/Backward lean)
        # Calculate angle of torso relative to vertical axis
        torso_angle = calculate_angle([shoulder[0], 0], shoulder, hip) # Angle vs Vertical

        # Store metrics
        frame_metrics["knee_angle"] = knee_angle
        frame_metrics["elbow_angle"] = elbow_angle
        frame_metrics["body_lean"] = torso_angle
        
        # --- VISUALIZATION ---
        # Draw Skeleton
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
        # Display Metrics on Screen
        cv2.putText(image, f"Knee Angle: {knee_angle}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(image, f"Elbow Angle: {elbow_angle}", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(image, f"Body Lean: {torso_angle}", (10, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    keypoint_data.append(frame_metrics)
    out.write(image)

cap.release()
out.release()

# Save Keypoints to JSON
with open(KEYPOINTS_PATH, 'w') as f:
    json.dump(keypoint_data, f, indent=4)

print(f"Done! Saved video to {OUTPUT_VIDEO_PATH} and data to {KEYPOINTS_PATH}")