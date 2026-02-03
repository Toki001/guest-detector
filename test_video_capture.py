import cv2
import time
import os
import math
from datetime import datetime

# --- CONFIGURATION ---
SAVE_FOLDER = "captured_faces"
REQUIRED_STILL_TIME = 3.5   # Seconds of stillness required
MOVEMENT_THRESHOLD = 50     # How many pixels they can drift before we reset the timer (Higher = More lenient)

if not os.path.exists(SAVE_FOLDER):
    os.makedirs(SAVE_FOLDER)

video_capture = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# --- STATE VARIABLES ---
anchor_center = None        # The (x,y) point where they started standing still
still_start_time = None     # When they started standing still
display_success_until = 0   # For the green flash

print("System Active. Hold still for 5 seconds to capture.")

def get_center(x, y, w, h):
    return (x + w // 2, y + h // 2)

def get_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

while True:
    ret, frame = video_capture.read()
    if not ret: break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))

    # --- LOGIC 1: NO FACE DETECTED ---
    if len(faces) == 0:
        # Reset everything if they leave
        anchor_center = None
        still_start_time = None
        
        cv2.putText(frame, "Waiting for subject...", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # --- LOGIC 2: FACE DETECTED ---
    for (x, y, w, h) in faces:
        current_center = get_center(x, y, w, h)
        current_time = time.time()
        
        # Draw the face box
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 0), 2)

        # Initialize the "Anchor" if this is the first frame we see them
        if anchor_center is None:
            anchor_center = current_center
            still_start_time = current_time

        # Calculate how far they have moved from the anchor
        drift = get_distance(current_center, anchor_center)

        # --- BRANCH A: MOVED TOO MUCH (RESET) ---
        if drift > MOVEMENT_THRESHOLD:
            # User moved! Reset the anchor to their NEW position
            anchor_center = current_center
            still_start_time = current_time # Restart the 5s timer
            
            # Visual Feedback: Resetting
            cv2.putText(frame, "MOVEMENT DETECTED - RESET", (x, y-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # --- BRANCH B: HOLDING STILL ---
        else:
            time_still = current_time - still_start_time
            
            # Check if we reached 5 seconds
            if time_still >= REQUIRED_STILL_TIME:
                
                # --- SNAP PHOTO ---
                face_image = frame[y:y+h, x:x+w]
                timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                filename = f"{SAVE_FOLDER}/face_{timestamp}.jpg"
                cv2.imwrite(filename, face_image)
                print(f"[{timestamp}] SNAP: Stillness Verified")
                
                # Success Visuals
                display_success_until = current_time + 1
                
                # IMPORTANT: Reset logic to allow another snap?
                # The user said "Remove cooldown", so we essentially restart the process immediately.
                # We set anchor to current position to start the next 5s check.
                anchor_center = current_center 
                still_start_time = current_time 

            else:
                # Countdown Visuals
                remaining = int(REQUIRED_STILL_TIME - time_still) + 1
                
                if current_time < display_success_until:
                    # Still showing the previous success message
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)
                    cv2.putText(frame, "CAPTURED!", (x, y-25), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                else:
                    # Showing the Countdown
                    cv2.putText(frame, f"Hold Still: {remaining}s", (x, y-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
                    
                    # Draw a small circle showing the "Anchor" point vs Current point (Optional Debugging)
                    cv2.circle(frame, anchor_center, 3, (0, 255, 0), -1) # Green dot = Anchor
                    cv2.line(frame, anchor_center, current_center, (0, 255, 255), 1) # Line showing drift

    cv2.imshow('Motion Detection Camera', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()