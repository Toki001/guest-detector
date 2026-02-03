import cv2
import boto3
import threading
import time
import datetime

COLLECTION_ID = ''
REGION = ''
AWS_ACCESS_KEY = ''
AWS_SECRET_KEY = ''
COOLDOWN_SECONDS = 5

rekognition = boto3.client('rekognition',
                           region_name=REGION,
                           aws_access_key_id=AWS_ACCESS_KEY,
                           aws_secret_access_key=AWS_SECRET_KEY)

last_scan_time = 0
current_status = "Scanning..."
status_color = (255, 255, 255)

def check_face_identity(image_bytes):
    
    global current_status, status_color
    
    try:
        response = rekognition.search_faces_by_image(
            CollectionId=COLLECTION_ID,
            Image={'Bytes': image_bytes},
            FaceMatchThreshold=80,
            MaxFaces=1
        )
        face_matches = response['FaceMatches']

        if not face_matches:
            current_status = "ALERT: GUEST DETECTED"
            status_color = (0, 0, 255)
            print(f"[{datetime.datetime.now()}] GUEST DETECTED!")
        else:
            name = face_matches[0]['Face']['ExternalImageId']
            confidence = face_matches[0]['Similarity']
            current_status = f"Access Granted: {name} ({int(confidence)}%)"
            status_color = (0, 255, 0)
            print(f"[{datetime.datetime.now()}] Recognized: {name}")
            
    except Exception as e:
        print(f"AWS Error: {e}")
        current_status = "API Error"
        status_color = (0, 165, 255)
        
video_capture = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

print ("System Armed. Press 'q' to quit.")

while True:
    ret, frame = video_capture.read()
    if not ret: break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    cv2.putText(frame, f"Status: {current_status}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
    
    if len(faces) > 0:
        (x, y, w, h) = faces[0]
        cv2.rectangle(frame, (x,y), (x+w, y+h), status_color, 2)
        
        if time.time() - last_scan_time > COOLDOWN_SECONDS:
            last_scan_time = time.time()
            current_status = "Identifying..."
            status_color = (255, 255, 0)
            
            face_roi = frame[y:y+h, x:x+w]
            _, img_encoded = cv2.imencode('.jpg', face_roi)
            image_bytes = img_encoded.tobytes()
            
            threading.Thread(target=check_face_identity, args=(image_bytes,)).start()
            
    cv2.imshow('Security Feed', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


video_capture.release()
cv2.destroyAllWindows()