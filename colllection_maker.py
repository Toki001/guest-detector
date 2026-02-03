import boto3

client = boto3.client('rekognition',
                      region_name='as-singapore-1',
                      aws_access_key_id = '',
                      aws_secret_access_key='')

try:
    response = client.create_collection(CollectionId='office_personnel')
    print("Collection created successfully!")
    print("Collection ARN:", response['CollectionArn'])
except client.exceptions.ResourceAlreadyExistsException:
    print("Collection already exists!")