import cv2
import time
import os
import math
import mediapipe as mp
from datetime import datetime

# --- CONFIGURATION ---
SAVE_FOLDER = "captured_faces"
REQUIRED_STILL_TIME = 5      # Seconds of stillness required
MOVEMENT_THRESHOLD = 60      # Pixel drift allowed
SUCCESS_LOCK_TIME = 2.0      # How long to wait after capture (Seconds)
PADDING = 80                 # <--- NOW THE BOX WILL SHOW THIS PADDING

# --- MEDIAPIPE SETUP ---
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.6)

if not os.path.exists(SAVE_FOLDER):
    os.makedirs(SAVE_FOLDER)

video_capture = cv2.VideoCapture(0)

# --- STATE VARIABLES ---
anchor_center = None
still_start_time = None
display_success_until = 0

print(f"System Active. Padding: {PADDING}px. Box shows actual capture area.")

def get_center(x, y, w, h):
    return (int(x + w / 2), int(y + h / 2))

def get_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

while True:
    ret, frame = video_capture.read()
    if not ret: break

    current_time = time.time()
    h_img, w_img, _ = frame.shape 

    # =======================================================
    # 1. LOCK SYSTEM IF "CAPTURED" IS SHOWING
    # =======================================================
    if current_time < display_success_until:
        cv2.putText(frame, "CAPTURED! Processing...", (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
        cv2.imshow('MediaPipe Face Cam', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
        continue 

    # =======================================================
    # 2. NORMAL DETECTION LOGIC
    # =======================================================
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(rgb_frame)

    if not results.detections: 
        anchor_center = None
        still_start_time = None
        cv2.putText(frame, "Waiting for subject...", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    else:
        for detection in results.detections:
            # 1. Get Original Face Coordinates (The tight fit)
            bboxC = detection.location_data.relative_bounding_box
            x = int(bboxC.xmin * w_img)
            y = int(bboxC.ymin * h_img)
            w_box = int(bboxC.width * w_img)
            h_box = int(bboxC.height * h_img)
            
            # 2. Calculate PADDED Coordinates (The capture area)
            # We calculate this EARLY so we can draw it
            x1 = max(0, x - PADDING)
            y1 = max(0, y - PADDING)
            x2 = min(w_img, x + w_box + PADDING)
            y2 = min(h_img, y + h_box + PADDING)

            # 3. Calculate Center based on original face (More stable for movement check)
            current_center = get_center(x, y, w_box, h_box)

            # Initialize Anchor
            if anchor_center is None:
                anchor_center = current_center
                still_start_time = current_time

            drift = get_distance(current_center, anchor_center)
            
            # --- DRAW THE BOX (NOW USING PADDED COORDINATES) ---
            # This box now represents exactly what will be saved
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)

            # MOVEMENT CHECK
            if drift > MOVEMENT_THRESHOLD:
                anchor_center = current_center
                still_start_time = current_time
                cv2.putText(frame, "MOVEMENT - RESET", (x1, y1-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            else:
                time_still = current_time - still_start_time
                
                # --- SNAP PHOTO ---
                if time_still >= REQUIRED_STILL_TIME:
                    
                    # CROP using the exact same variables we drew with
                    face_image = frame[y1:y2, x1:x2]

                    if face_image.size > 0:
                        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                        filename = f"{SAVE_FOLDER}/face_{timestamp}.jpg"
                        cv2.imwrite(filename, face_image)
                        print(f"[{timestamp}] SNAP: Saved.")
                        
                        display_success_until = current_time + SUCCESS_LOCK_TIME
                        anchor_center = None
                        still_start_time = None
                        break 
                else:
                    # COUNTDOWN
                    remaining = int(REQUIRED_STILL_TIME - time_still) + 1
                    cv2.putText(frame, f"Hold Still: {remaining}s", (x1, y1-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)

    cv2.imshow('MediaPipe Face Cam', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()