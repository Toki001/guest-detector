import boto3
import os
import re
from dotenv import load_dotenv

# 1. Load environment variables
load_dotenv()

def add_employee_to_database(image_path, employee_name):
    # 2. Get Config from .env
    REGION = os.getenv('AWS_REGION', 'us-east-1')
    ACCESS_KEY = os.getenv('AWS_ACCESS_KEY_ID')
    SECRET_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
    COLLECTION = os.getenv('COLLECTION_ID', 'office_personnel')

    # 3. Clean the name (AWS allows only a-z, 0-9, _, -)
    # This turns "John Doe" into "John_Doe" automatically
    safe_name = re.sub(r'[^a-zA-Z0-9_.\-]', '_', employee_name)

    # 4. Initialize Client
    client = boto3.client('rekognition',
                          region_name=REGION,
                          aws_access_key_id=ACCESS_KEY,
                          aws_secret_access_key=SECRET_KEY) # Fixed typo here

    # 5. Send to AWS
    print(f"Uploading {safe_name} to collection '{COLLECTION}'...")
    
    try:
        with open(image_path, 'rb') as image:
            response = client.index_faces(
                CollectionId=COLLECTION,
                Image={'Bytes': image.read()},
                ExternalImageId=safe_name, # The 'Name' tag for this face
                MaxFaces=1,
                QualityFilter='AUTO',
                DetectionAttributes=['ALL']
            )
            
            # Check if a face was actually found in the photo
            if not response['FaceRecords']:
                print("Error: No face detected in the image.")
            else:
                print(f"Successfully added: {safe_name}")
                print(f"Face ID: {response['FaceRecords'][0]['Face']['FaceId']}")

    except FileNotFoundError:
        print(f"Error: The file '{image_path}' was not found.")
    except Exception as e:
        print(f"AWS Error: {e}")


add_employee_to_database('my_photo.jpg', 'James Kier')