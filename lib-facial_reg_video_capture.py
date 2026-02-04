import face_recognition
import cv2
import os
import numpy as np

# --- CONFIGURATION ---
KNOWN_FACES_DIR = 'known_faces'
TOLERANCE = 0.6  # Lower number = stricter matching (0.6 is default)
FRAME_THICKNESS = 3
FONT_THICKNESS = 2
MODEL = 'hog'  # Use 'hog' for CPU, 'cnn' for GPU (if you have CUDA)

# --- SETUP: LOAD KNOWN FACES ---
print("Loading known faces...")
known_face_encodings = []
known_face_names = []

# Loop over the images in the folder
if not os.path.exists(KNOWN_FACES_DIR):
    os.makedirs(KNOWN_FACES_DIR)
    print(f"Created folder '{KNOWN_FACES_DIR}'. Please add photos there and restart!")

for filename in os.listdir(KNOWN_FACES_DIR):
    if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".jpeg"):
        # 1. Load the image
        path = os.path.join(KNOWN_FACES_DIR, filename)
        image = face_recognition.load_image_file(path)
        
        # 2. Get the "embedding" (encoding)
        # We assume there is only 1 face per photo in the database
        encodings = face_recognition.face_encodings(image)
        
        if encodings:
            known_face_encodings.append(encodings[0])
            # Use filename without extension as the name
            name = os.path.splitext(filename)[0]
            known_face_names.append(name)
            print(f"Loaded: {name}")

print(f"Database loaded. {len(known_face_names)} identities found.")

# --- MAIN LOOP ---
video_capture = cv2.VideoCapture(0) # 0 = Default Webcam

while True:
    ret, frame = video_capture.read()
    if not ret: break

    # Optimization: Resize frame to 1/4 size for faster processing
    # (We will scale the detection coordinates back up x4 later)
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # OpenCV uses BGR, face_recognition uses RGB
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # 1. FIND FACES
    face_locations = face_recognition.face_locations(rgb_small_frame, model=MODEL)
    
    # 2. CALCULATE EMBEDDINGS
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    for face_encoding, face_location in zip(face_encodings, face_locations):
        
        # 3. COMPARE WITH DATABASE
        # returns a list of True/False
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=TOLERANCE)
        name = "Unknown"

        # Calculate geometric distance to find the *best* match
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        
        if len(face_distances) > 0:
            best_match_index = np.argmin(face_distances)
            
            # If the best match is within tolerance, use that name
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

        # 4. DRAW RESULT (Scale back up by 4)
        top, right, bottom, left = face_location
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Choose color: Green for known, Red for unknown
        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)

        # Draw box
        cv2.rectangle(frame, (left, top), (right, bottom), color, FRAME_THICKNESS)
        
        # Draw label
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
        cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)

    cv2.imshow('Local Security', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()