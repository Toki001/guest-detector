import boto3
import os
from dotenv import load_dotenv

# 1. Load the .env file
load_dotenv()

# 2. Get variables from environment
# We use 'os.getenv' to read the hidden file
REGION = os.getenv('AWS_REGION', 'us-east-1')
ACCESS_KEY = os.getenv('AWS_ACCESS_KEY_ID')
SECRET_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
COLLECTION_NAME = os.getenv('COLLECTION_ID', 'office_personnel')

# 3. Initialize the AWS Client
client = boto3.client('rekognition',
                      region_name=REGION,
                      aws_access_key_id=ACCESS_KEY,
                      aws_secret_access_key=SECRET_KEY)

# 4. Create the Collection
print(f"Attempting to create collection: {COLLECTION_NAME} in {REGION}...")

try:
    response = client.create_collection(CollectionId=COLLECTION_NAME)
    print("Collection created successfully!")
    print("Collection ARN:", response['CollectionArn'])
    print(f"StatusCode: {response['StatusCode']}")
    
except client.exceptions.ResourceAlreadyExistsException:
    print(f"⚠️ Collection '{COLLECTION_NAME}' already exists! (You don't need to do anything).")

except Exception as e:
    print(f"Error: {e}")
    print("Check your .env file to ensure keys are correct.")