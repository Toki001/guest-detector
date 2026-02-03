import boto3 

def add_employee_to_database(image_path, employee_name):
    client = boto3.client('rekognition', region_name='',
                          aws_access_key_id='',
                          aws_secret_access_key_id='')
    
    with open(image_path, 'rb') as image:
        try:
            response = client.index_faces(
                CollectionId='',
                Image={'Bytes': image.read()},
                ExternalImageId=employee_name,
                MaxFaces=1,
                QualityFilter='AUTO',
                DetectionAttributes=['ALL']
            )
            print(f'Successfully added: {employee_name}')
        except Exception as e:
            print(f'Error: {e}')

add_employee_to_database('image_path', 'employee_name')