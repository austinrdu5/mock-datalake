import os
import boto3
import logging
from datetime import datetime
from dotenv import load_dotenv
from botocore.exceptions import ClientError

# Configure logging
log_filename = f"s3_copy_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
try:
    load_dotenv()
    logger.info("Environment variables loaded successfully")
except Exception as e:
    logger.error(f"Failed to load environment variables: {str(e)}")
    raise

# Source bucket configuration
source_bucket = os.getenv('AWS_S3_BUCKET_NAME')
source_region = os.getenv('AWS_DEFAULT_REGION')
source_access_key = os.getenv('AWS_ACCESS_KEY')
source_secret_key = os.getenv('AWS_SECRET_KEY')

# Destination bucket configuration
dest_bucket = os.getenv('TEST_AWS_S3_BUCKET_NAME')
dest_region = os.getenv('AWS_DEFAULT_REGION')
dest_access_key = os.getenv('TEST_AWS_ACCESS_KEY')
dest_secret_key = os.getenv('TEST_AWS_SECRET_KEY')

# Validate environment variables
required_vars = {
    'AWS_S3_BUCKET_NAME': source_bucket,
    'AWS_DEFAULT_REGION': source_region,
    'AWS_ACCESS_KEY': source_access_key,
    'AWS_SECRET_KEY': source_secret_key,
    'TEST_AWS_S3_BUCKET_NAME': dest_bucket,
    'TEST_AWS_ACCESS_KEY': dest_access_key,
    'TEST_AWS_SECRET_KEY': dest_secret_key
}

missing_vars = [var for var, value in required_vars.items() if not value]
if missing_vars:
    logger.error(f"Missing required environment variables: {', '.join(missing_vars)}")
    raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")

logger.info("Environment variables validated successfully")

# Initialize S3 clients
try:
    source_s3 = boto3.client(
        's3',
        region_name=source_region,
        aws_access_key_id=source_access_key,
        aws_secret_access_key=source_secret_key
    )
    logger.info("Source S3 client initialized successfully")

    dest_s3 = boto3.client(
        's3',
        region_name=dest_region,
        aws_access_key_id=dest_access_key,
        aws_secret_access_key=dest_secret_key
    )
    logger.info("Destination S3 client initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize S3 clients: {str(e)}")
    raise

# List of prefixes to copy
prefixes_to_copy = [
    'bronze/alphavantage/AAPL/',
    'bronze/alphavantage/AMZN/',
    'bronze/alphavantage/GOOG/',
    'bronze/alphavantage/META/',
    'bronze/ecommerce/',
    'bronze/openweathermap/city=new_york/',
    'bronze/openweathermap/city=london/',
    'bronze/sales-data/'
]

def copy_objects(prefix):
    """Copy all objects with the given prefix from source to destination bucket."""
    logger.info(f"Starting copy operation for prefix: {prefix}")
    objects_copied = 0
    objects_failed = 0
    
    try:
        # List all objects with the given prefix
        paginator = source_s3.get_paginator('list_objects_v2')
        for page_num, page in enumerate(paginator.paginate(Bucket=source_bucket, Prefix=prefix), 1):
            if 'Contents' not in page:
                logger.warning(f"No objects found in page {page_num} for prefix {prefix}")
                continue
                
            logger.info(f"Processing page {page_num} for prefix {prefix}")
            
            for obj in page['Contents']:
                source_key = obj['Key']
                try:
                    # Copy object to destination bucket
                    copy_source = {
                        'Bucket': source_bucket,
                        'Key': source_key
                    }
                    dest_s3.copy_object(
                        CopySource=copy_source,
                        Bucket=dest_bucket,
                        Key=source_key
                    )
                    objects_copied += 1
                    logger.info(f"Successfully copied: {source_key}")
                    
                except ClientError as e:
                    objects_failed += 1
                    logger.error(f"Failed to copy {source_key}: {str(e)}")
                    continue
                    
    except ClientError as e:
        logger.error(f"Error listing objects with prefix {prefix}: {str(e)}")
        raise
    
    logger.info(f"Completed copy operation for prefix {prefix}")
    logger.info(f"Summary for {prefix}: {objects_copied} objects copied, {objects_failed} objects failed")
    return objects_copied, objects_failed

def main():
    logger.info(f"Starting copy operation from {source_bucket} to {dest_bucket}")
    total_objects_copied = 0
    total_objects_failed = 0
    
    for prefix in prefixes_to_copy:
        try:
            copied, failed = copy_objects(prefix)
            total_objects_copied += copied
            total_objects_failed += failed
        except Exception as e:
            logger.error(f"Failed to process prefix {prefix}: {str(e)}")
            continue
    
    logger.info("\nCopy operation completed!")
    logger.info(f"Total summary: {total_objects_copied} objects copied, {total_objects_failed} objects failed")
    
    if total_objects_failed > 0:
        logger.warning(f"There were {total_objects_failed} failed copy operations. Check the log file for details.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.critical(f"Script failed with error: {str(e)}")
        raise
