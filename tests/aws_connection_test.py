import os
import logging

from dotenv import load_dotenv
import boto3
from botocore.exceptions import ClientError
import pytest

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

def test_aws_connection():
    """Test AWS connection using environment variables"""
    # Initialize AWS client
    s3_client = boto3.client(
        's3',
        aws_access_key_id=os.environ.get('AWS_ACCESS_KEY'),
        aws_secret_access_key=os.environ.get('AWS_SECRET_KEY')
    )
    
    try:
        # Try to list buckets
        response = s3_client.list_buckets()
        assert 'Buckets' in response
        assert isinstance(response['Buckets'], list)
    except Exception as e:
        pytest.fail(f"Failed to connect to AWS: {str(e)}")

if __name__ == "__main__":
    test_aws_connection()
    logger.info("✅ AWS connection test passed successfully!")