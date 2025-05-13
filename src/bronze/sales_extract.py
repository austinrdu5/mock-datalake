import boto3
import requests
import io
import zipfile
import json
from datetime import datetime
import logging
import os
from typing import Optional
from tenacity import retry, stop_after_attempt, wait_exponential

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/s3_ingest.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('s3_ingest')

# Initialize S3 client as a global variable
s3_client = boto3.client('s3')

def validate_json_content(content: bytes) -> bool:
    """Validate if the content is valid JSON array by wrapping in brackets and fixing commas."""
    try:
        content_str = content.decode('utf-8').strip()
        # Remove any trailing commas and wrap in brackets
        if content_str.endswith(','):
            content_str = content_str[:-1]
        # If not already wrapped, wrap in brackets
        if not (content_str.startswith('[') and content_str.endswith(']')):
            content_str = f'[{content_str}]'
        # Try to parse as JSON array
        json.loads(content_str)
        return True
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        logger.error(f"JSON validation error: {str(e)}")
        # Log the first 100 characters of the content for debugging
        try:
            preview = content[:100].decode('utf-8', errors='replace')
            logger.error(f"Content preview: {preview}")
        except Exception as e:
            logger.error(f"Could not preview content: {str(e)}")
        return False

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def upload_to_s3(bucket: str, key: str, content: bytes, metadata: dict) -> bool:
    """Upload content to S3 with retry logic"""
    try:
        s3_client.put_object(
            Bucket=bucket,
            Key=key,
            Body=content,
            Metadata=metadata
        )
        return True
    except Exception as e:
        logger.error(f"Failed to upload to S3: {str(e)}")
        raise

def stream_zip_to_s3(url: str, s3_bucket: str, prefix: str = "bronze/sales_data") -> bool:
    """Stream ZIP file contents directly to S3 without local storage"""
    try:
        # Validate inputs
        if not url or not s3_bucket:
            logger.error("URL and S3 bucket name are required")
            return False

        # Get the ZIP file content
        logger.info(f"Downloading ZIP file from {url}")
        response = requests.get(url)
        
        if response.status_code != 200:
            logger.error(f"Failed to download ZIP: {response.status_code}")
            return False
            
        # Current date for organizing data
        date_prefix = datetime.utcnow().strftime("%Y/%m/%d")
        
        # Process the ZIP file in memory
        with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
            file_count = 0
            
            # Process each file in the ZIP
            for filename in zip_ref.namelist():
                if filename.endswith('.json'):
                    logger.info(f"Processing {filename} from ZIP")
                    
                    # Read the file content
                    with zip_ref.open(filename) as file:
                        file_content = file.read()
                    
                    # Log file size for debugging
                    logger.info(f"File size: {len(file_content)} bytes")
                    
                    # Validate JSON content
                    if not validate_json_content(file_content):
                        logger.error(f"Invalid JSON content in {filename}")
                        continue
                    
                    # Create S3 key with date partitioning
                    s3_key = f"{prefix}/{date_prefix}/{filename}"
                    
                    # Metadata including ingestion timestamp
                    metadata = {
                        'ingestion_timestamp': datetime.utcnow().isoformat(),
                        'source': 'sample_sales_data',
                        'content_type': 'application/json',
                        'origin_url': url
                    }
                    
                    logger.info(f"Uploading to s3://{s3_bucket}/{s3_key}")
                    if upload_to_s3(s3_bucket, s3_key, file_content, metadata):
                        file_count += 1
                    
            logger.info(f"Successfully uploaded {file_count} files to S3")
            return True
            
    except Exception as e:
        logger.error(f"Error processing ZIP file: {str(e)}")
        return False

if __name__ == "__main__":
    # Get configuration from environment variables
    S3_BUCKET = os.getenv('S3_BUCKET', 'mock-datalake1')
    ZIP_URL = os.getenv('ZIP_URL', 'https://github.com/cloudacademy/ca-sample-sales-data/raw/master/sales_data.zip')
    
    stream_zip_to_s3(ZIP_URL, S3_BUCKET)