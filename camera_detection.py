import cv2
import boto3
import threading
import time
import math
import mediapipe as mp
import datetime

# --- CONFIGURATION ---
COLLECTION_ID = 'office_personnel'
REGION = 'us-east-1'
AWS_ACCESS_KEY = 'test'
AWS_SECRET_KEY = 'test'

# Motion & Timing Config
REQUIRED_STILL_TIME = 5     # Seconds of stillness required before sending to AWS
MOVEMENT_THRESHOLD = 50     # Pixel drift allowed before resetting timer

# --- SETUP AWS ---
rekognition = boto3.client('rekognition', 
                           region_name=REGION,
                           aws_access_key_id=AWS_ACCESS_KEY, 
                           aws_secret_access_key=AWS_SECRET_KEY)

# --- SETUP MEDIAPIPE ---
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.6)

# Global variables for thread communication
current_status = "Waiting for subject..."
status_color = (255, 255, 255) # White
is_processing_aws = False      # Lock to prevent double-sending

def check_face_identity(image_bytes):
    """
    Runs in a background thread to identify the face via AWS.
    """
    global current_status, status_color, is_processing_aws
    
    try:
        response = rekognition.search_faces_by_image(
            CollectionId=COLLECTION_ID,
            Image={'Bytes': image_bytes},
            FaceMatchThreshold=80,
            MaxFaces=1
        )
        
        face_matches = response['FaceMatches']
        
        if not face_matches:
            # GUEST DETECTED
            current_status = "ALERT: GUEST DETECTED"
            status_color = (0, 0, 255) # Red
            print(f"[{datetime.datetime.now()}] Unknown person detected!")
        else:
            # EMPLOYEE DETECTED
            name = face_matches[0]['Face']['ExternalImageId']
            confidence = face_matches[0]['Similarity']
            current_status = f"Access Granted: {name} ({int(confidence)}%)"
            status_color = (0, 255, 0) # Green
            print(f"[{datetime.datetime.now()}] Recognized: {name}")

    except Exception as e:
        print(f"AWS Error: {e}")
        current_status = "API Error"
        status_color = (0, 165, 255) # Orange
    
    finally:
        # Unlock the system so it can scan again if needed
        time.sleep(2) # Keep the result on screen for 2 seconds
        is_processing_aws = False

# --- HELPER FUNCTIONS ---
def get_center(x, y, w, h):
    return (int(x + w / 2), int(y + h / 2))

def get_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

# --- MAIN LOOP ---
video_capture = cv2.VideoCapture(0)

# State Variables
anchor_center = None        # The (x,y) point where they started standing still
still_start_time = None     # When they started standing still

print("System Armed. Hold still for 5 seconds to scan.")

while True:
    ret, frame = video_capture.read()
    if not ret: break

    # 1. PREPARE FRAME (MediaPipe needs RGB)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(rgb_frame)

    # UI Header
    cv2.putText(frame, f"System: {current_status}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

    # 2. LOGIC: NO FACES FOUND
    if not results.detections:
        if not is_processing_aws:
            anchor_center = None
            still_start_time = None
            current_status = "Waiting for subject..."
            status_color = (255, 255, 255)

    # 3. LOGIC: FACES FOUND
    else:
        for detection in results.detections:
            # Convert Relative Coords -> Pixel Coords
            h_img, w_img, _ = frame.shape
            bboxC = detection.location_data.relative_bounding_box
            x = int(bboxC.xmin * w_img)
            y = int(bboxC.ymin * h_img)
            w = int(bboxC.width * w_img)
            h = int(bboxC.height * h_img)
            
            # Safety Clamp (Prevent crashes if face is half off-screen)
            x, y = max(0, x), max(0, y)

            # Draw Box
            cv2.rectangle(frame, (x, y), (x+w, y+h), status_color, 2)

            # Skip logic if we are currently talking to AWS
            if is_processing_aws:
                continue

            current_center = get_center(x, y, w, h)
            current_time = time.time()

            # Initialize Anchor
            if anchor_center is None:
                anchor_center = current_center
                still_start_time = current_time

            # Calculate Drift
            drift = get_distance(current_center, anchor_center)

            # BRANCH A: MOVED TOO MUCH (RESET)
            if drift > MOVEMENT_THRESHOLD:
                anchor_center = current_center
                still_start_time = current_time
                current_status = "MOVEMENT - RESETTING..."
                status_color = (0, 165, 255) # Orange

            # BRANCH B: HOLDING STILL
            else:
                time_still = current_time - still_start_time
                
                # BRANCH B1: 5 SECONDS REACHED -> SEND TO AWS
                if time_still >= REQUIRED_STILL_TIME:
                    is_processing_aws = True # Lock system
                    current_status = "Identifying..."
                    status_color = (255, 255, 0) # Yellow
                    
                    # Crop Face
                    face_roi = frame[y:y+h, x:x+w]
                    
                    # Encode and Send
                    if face_roi.size > 0:
                        _, img_encoded = cv2.imencode('.jpg', face_roi)
                        image_bytes = img_encoded.tobytes()
                        threading.Thread(target=check_face_identity, args=(image_bytes,)).start()
                    
                    # Reset Anchor for next time
                    anchor_center = None 
                    still_start_time = None

                # BRANCH B2: COUNTDOWN
                else:
                    remaining = int(REQUIRED_STILL_TIME - time_still) + 1
                    cv2.putText(frame, f"Hold Still: {remaining}s", (x, y-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
                    current_status = "Verifying Stillness..."
                    status_color = (255, 255, 255)

    cv2.imshow('Smart Security Feed', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()