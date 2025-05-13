import boto3
import re
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

bucket = 'mock-datalake1'
base_prefix = 'bronze/alphavantage/'
s3 = boto3.client('s3')

logger.info(f"Starting S3 file renaming process for all stock folders in {bucket}/{base_prefix}")

# List all stock symbol folders
logger.info("Listing all stock symbol folders...")
response = s3.list_objects_v2(Bucket=bucket, Prefix=base_prefix, Delimiter='/')
folders = [cp['Prefix'] for cp in response.get('CommonPrefixes', [])]
logger.info(f"Found {len(folders)} stock folders: {folders}")

total_processed = 0
total_renamed = 0
total_skipped = 0

for prefix in folders:
    logger.info(f"\n{'='*80}\nProcessing folder: {prefix}\n{'='*80}")
    logger.info(f"Listing files in {prefix}...")
    response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
    files = response.get('Contents', [])
    logger.info(f"Found {len(files)} files to process in {prefix}")
    processed = 0
    renamed = 0
    skipped = 0
    for obj in files:
        key = obj['Key']
        processed += 1
        logger.info(f"Processing file {processed}/{len(files)}: {key}")
        
        # Get just the filename without path
        filename = key.split('/')[-1]
        
        # Check if filename matches pattern YYYYMMDDHHMMSS.csv
        match = re.match(r'(\d{8})(\d{6})\.csv$', filename)
        if match:
            date_part = match.group(1)
            time_part = match.group(2)
            # Construct new filename with underscore
            new_filename = f"{date_part}_{time_part}.csv"
            new_key = f"{prefix}{new_filename}"
            
            logger.info(f"Renaming {key} -> {new_key}")
            try:
                # Copy to new key
                s3.copy_object(Bucket=bucket, CopySource={'Bucket': bucket, 'Key': key}, Key=new_key)
                # Delete old key
                s3.delete_object(Bucket=bucket, Key=key)
                logger.info(f"Successfully renamed {key} -> {new_key}")
                renamed += 1
            except Exception as e:
                logger.error(f"Failed to rename {key}: {str(e)}")
        else:
            logger.info(f"Skipping {key} - doesn't match expected pattern")
            skipped += 1
            
    logger.info(f"\nFolder complete: {prefix}")
    logger.info(f"Processed: {processed}, Renamed: {renamed}, Skipped: {skipped}")
    total_processed += processed
    total_renamed += renamed
    total_skipped += skipped

logger.info(f"\n{'='*80}\nAll folders complete.")
logger.info(f"Total Processed: {total_processed}")
logger.info(f"Total Renamed: {total_renamed}")
logger.info(f"Total Skipped: {total_skipped}\n{'='*80}")
