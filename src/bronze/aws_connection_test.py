import os
import logging

from dotenv import load_dotenv
import boto3
from botocore.exceptions import ClientError

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

def test_aws_connection():
    """
    Test AWS S3 connection and perform basic operations.
    """
    try:
        # Get AWS credentials from environment variables
        aws_access_key = os.getenv('AWS_ACCESS_KEY')
        aws_secret_key = os.getenv('AWS_SECRET_KEY')
        aws_region = os.getenv('AWS_DEFAULT_REGION')
        bucket_name = os.getenv('AWS_S3_BUCKET_NAME')
        
        if not all([aws_access_key, aws_secret_key, aws_region, bucket_name]):
            logger.error("Missing AWS credentials in .env file")
            return False
            
        # Create S3 client
        s3_client = boto3.client(
            's3',
            region_name=aws_region,
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_key
        )
        
        # List buckets to verify connection
        response = s3_client.list_buckets()
        buckets = [bucket['Name'] for bucket in response['Buckets']]
        
        if bucket_name in buckets:
            logger.info(f"Successfully connected to AWS. Your bucket '{bucket_name}' exists.")
        else:
            logger.warning(f"Bucket '{bucket_name}' not found. Available buckets: {buckets}")
            return False
            
        # Create a simple test file
        test_content = "This is a test file to verify S3 connection."
        s3_client.put_object(
            Bucket=bucket_name,
            Key="test/connection_test.txt",
            Body=test_content
        )
        logger.info("Successfully created test file in S3 bucket")
        
        # List objects in the bucket
        response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix="test/")
        objects = [obj['Key'] for obj in response.get('Contents', [])]
        logger.info(f"Objects in test directory: {objects}")
        
        return True
    
    except ClientError as e:
        logger.error(f"AWS Error: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return False

if __name__ == "__main__":
    success = test_aws_connection()
    if success:
        logger.info("✅ AWS connection test passed successfully!")
    else:
        logger.error("❌ AWS connection test failed!")