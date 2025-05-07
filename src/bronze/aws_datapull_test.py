import boto3
import json
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# AWS Configuration
AWS_ACCESS_KEY = os.getenv('AWS_ACCESS_KEY')
AWS_SECRET_KEY = os.getenv('AWS_SECRET_KEY')
S3_BUCKET = os.getenv('AWS_S3_BUCKET_NAME')

# Initialize S3 client
s3_client = boto3.client(
    's3',
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY
)

# List objects in the OpenWeatherMap folder
response = s3_client.list_objects_v2(
    Bucket=S3_BUCKET,
    Prefix='bronze/openweathermap/'
)

# Get the most recent file
if 'Contents' in response:
    files = sorted(response['Contents'], key=lambda x: x['LastModified'], reverse=True)
    if files:
        latest_file = files[0]['Key']
        print(f"Most recent file: {latest_file}")
        
        # Get the file content
        obj = s3_client.get_object(Bucket=S3_BUCKET, Key=latest_file)
        file_content = obj['Body'].read().decode('utf-8')
        
        # Parse JSON
        data = json.loads(file_content)

        # Print contents
        print([x.keys() for x in data])
        
    else:
        print("No files found")
else:
    print("No files found")