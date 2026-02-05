import cv2
import boto3
import threading
import time
import math
import mediapipe as mp
import datetime
import os
from dotenv import load_dotenv

# CONFIGURATION
load_dotenv() # Load secrets from .env file
REGION = os.getenv('AWS_REGION')
AWS_ACCESS_KEY = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
COLLECTION_ID = os.getenv('COLLECTION_ID', 'office_personnel')

# Tuning Parameters
REQUIRED_STILL_TIME = 5.0
MOVEMENT_THRESHOLD = 60
SUCCESS_LOCK_TIME = 2.0
PADDING = 50          

# SETUP AWS
# Safety check to prevent crashing if keys are missing
if AWS_ACCESS_KEY and AWS_SECRET_KEY:
    try:
        rekognition = boto3.client('rekognition', 
                                   region_name=REGION,
                                   aws_access_key_id=AWS_ACCESS_KEY, 
                                   aws_secret_access_key=AWS_SECRET_KEY)
    except Exception as e:
        print(f"AWS Init Error: {e}")
        rekognition = None
else:
    print("WARNING: AWS Keys missing in .env file.")
    rekognition = None

# SETUP MEDIAPIPE
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.6)

# Global variables
scan_result_message = ""       # Stores the text result from AWS
status_color = (255, 255, 255) # White

def check_face_identity(image_bytes):
    """
    Runs in background thread: Sends image to AWS and updates global status.
    """
    global scan_result_message, status_color
    
    if not rekognition:
        scan_result_message = "AWS NOT CONFIGURED"
        status_color = (0, 165, 255)
        return

    try:
        response = rekognition.search_faces_by_image(
            CollectionId=COLLECTION_ID,
            Image={'Bytes': image_bytes},
            FaceMatchThreshold=80,
            MaxFaces=1
        )
        
        face_matches = response['FaceMatches']
        
        if not face_matches:
            # GUEST
            scan_result_message = "ALERT: UNKNOWN GUEST"
            status_color = (0, 0, 255) # Red
            print(f"[{datetime.datetime.now()}] AWS Result: Guest Detected")
        else:
            # EMPLOYEE
            name = face_matches[0]['Face']['ExternalImageId']
            confidence = face_matches[0]['Similarity']
            scan_result_message = f"ACCESS GRANTED: {name}"
            status_color = (0, 255, 0) # Green
            print(f"[{datetime.datetime.now()}] AWS Result: {name}")

    except Exception as e:
        print(f"AWS Error: {e}")
        scan_result_message = "API ERROR"
        status_color = (0, 165, 255) # Orange

# --- HELPER FUNCTIONS ---
def get_center(x, y, w, h):
    return (int(x + w / 2), int(y + h / 2))

def get_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

# --- MAIN LOOP ---
video_capture = cv2.VideoCapture(0)

anchor_center = None
still_start_time = None
system_lock_until = 0  # Controls the freeze after capture

print(f"System Active. Padding: {PADDING}px. Box shows capture area.")

while True:
    ret, frame = video_capture.read()
    if not ret: break

    current_time = time.time()
    h_img, w_img, _ = frame.shape 

    # =======================================================
    # 1. LOCK SYSTEM IF "CAPTURED" IS SHOWING
    # =======================================================
    if current_time < system_lock_until:
        # Show the result text (or "Processing..." if AWS is slow)
        display_text = scan_result_message if scan_result_message else "Processing..."
        
        # Draw a banner background for readability
        cv2.rectangle(frame, (0, 0), (w_img, 80), (0, 0, 0), -1)
        cv2.putText(frame, display_text, (20, 55), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 3)
        
        cv2.imshow('Smart Security Feed', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
        continue  # SKIP EVERYTHING ELSE

    # =======================================================
    # 2. NORMAL DETECTION LOGIC
    # =======================================================
    
    # Reset message for new scan
    scan_result_message = ""
    
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(rgb_frame)

    # A. NO FACES
    if not results.detections:
        anchor_center = None
        still_start_time = None
        cv2.putText(frame, "Waiting for subject...", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # B. FACES FOUND
    else:
        for detection in results.detections:
            # 1. Get TIGHT Coordinates (from MediaPipe)
            bboxC = detection.location_data.relative_bounding_box
            x = int(bboxC.xmin * w_img)
            y = int(bboxC.ymin * h_img)
            w_box = int(bboxC.width * w_img)
            h_box = int(bboxC.height * h_img)
            
            # 2. Calculate PADDED Coordinates (The capture area)
            x1 = max(0, x - PADDING)
            y1 = max(0, y - PADDING)
            x2 = min(w_img, x + w_box + PADDING)
            y2 = min(h_img, y + h_box + PADDING)

            # 3. Calculate Center (using tight box for accuracy)
            current_center = get_center(x, y, w_box, h_box)

            # Initialize Anchor
            if anchor_center is None:
                anchor_center = current_center
                still_start_time = current_time

            drift = get_distance(current_center, anchor_center)
            
            # --- DRAW THE PADDED BOX ---
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)

            # --- MOTION CHECK ---
            if drift > MOVEMENT_THRESHOLD:
                anchor_center = current_center
                still_start_time = current_time
                cv2.putText(frame, "MOVEMENT - RESET", (x1, y1-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            else:
                time_still = current_time - still_start_time
                
                # --- SNAP PHOTO TRIGGER ---
                if time_still >= REQUIRED_STILL_TIME:
                    
                    # CROP using the PADDED variables
                    face_image = frame[y1:y2, x1:x2]

                    if face_image.size > 0:
                        # 1. Send to AWS
                        _, img_encoded = cv2.imencode('.jpg', face_image)
                        image_bytes = img_encoded.tobytes()
                        threading.Thread(target=check_face_identity, args=(image_bytes,)).start()
                        
                        print(f"[{datetime.datetime.now()}] SNAP: Sending to AWS...")
                        
                        # 2. LOCK THE SYSTEM
                        system_lock_until = current_time + SUCCESS_LOCK_TIME
                        
                        # 3. FORCE RESET VARIABLES
                        anchor_center = None
                        still_start_time = None
                        
                        # Break loop to apply lock immediately
                        break 
                else:
                    # COUNTDOWN
                    remaining = int(REQUIRED_STILL_TIME - time_still) + 1
                    cv2.putText(frame, f"Hold Still: {remaining}s", (x1, y1-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)

    cv2.imshow('Smart Security Feed', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()