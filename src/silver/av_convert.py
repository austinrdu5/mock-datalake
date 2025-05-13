from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, lit, to_date, when, expr
)
from pyspark.sql.types import (
    TimestampType, DateType
)
import uuid
from datetime import datetime
import boto3
import os
import sys
import logging
from typing import Optional, Dict, Any

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    force=True  # This ensures we can reconfigure logging
)
logger = logging.getLogger(__name__)

# AWS credentials - in production, use environment variables or AWS profiles
# DO NOT hardcode credentials in your script
aws_access_key = os.environ.get('AWS_ACCESS_KEY_ID')
aws_secret_key = os.environ.get('AWS_SECRET_ACCESS_KEY')
aws_region = os.environ.get('AWS_REGION', 'us-east-1')

# Validate AWS credentials
if not all([aws_access_key, aws_secret_key]):
    logger.error("AWS credentials not found in environment variables")
    logger.error("Please set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY")
    sys.exit(1)

# Bucket name
bucket_name = "mock-datalake1"

# Initialize Spark session with S3 configuration
def init_spark():
    """Initialize the Spark session with proper S3 configurations"""
    spark = SparkSession.builder \
        .appName("StockDataProcessor") \
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem") \
        .config("spark.hadoop.fs.s3a.access.key", aws_access_key) \
        .config("spark.hadoop.fs.s3a.secret.key", aws_secret_key) \
        .config("spark.hadoop.fs.s3a.endpoint", f"s3.{aws_region}.amazonaws.com") \
        .config("spark.hadoop.fs.s3a.path.style.access", "true") \
        .config("spark.sql.sources.partitionOverwriteMode", "dynamic") \
        .getOrCreate()
    
    # Set log level to reduce verbosity
    spark.sparkContext.setLogLevel("WARN")
    
    return spark

# Function to get the most recent file for a ticker
def get_latest_file(ticker):
    try:
        # Create boto3 S3 client
        s3 = boto3.client(
            's3',
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_key,
            region_name=aws_region
        )
        
        # List objects in the ticker's directory
        prefix = f"bronze/alphavantage/{ticker}/"
        response = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
        
        if 'Contents' not in response:
            logger.warning(f"No files found for {ticker}")
            return None
        
        # Extract just the filenames and sort them
        files = [obj['Key'] for obj in response['Contents'] if obj['Key'].endswith('.csv')]
        
        if not files:
            logger.warning(f"No CSV files found for {ticker}")
            return None
        
        # Sort by filename (which contains timestamp)
        files.sort()
        
        # Get the latest file (last in the sorted list)
        latest_file = files[-1]
        
        return f"s3a://{bucket_name}/{latest_file}"
    except Exception as e:
        logger.error(f"Error accessing S3 for {ticker}: {e}")
        return None

# Function to process one stock ticker's data
def process_stock_ticker(spark, ticker):
    logger.info(f"Processing {ticker}...")
    
    # Get the path to the latest file
    latest_file_path = get_latest_file(ticker)
    
    if not latest_file_path:
        logger.warning(f"No CSV files found for {ticker}")
        return None
        
    logger.info(f"Latest file for {ticker}: {latest_file_path}")
    
    # Generate a batch ID for this processing run
    batch_id = str(uuid.uuid4())
    current_timestamp = datetime.now()
    
    try:
        # Read the bronze data from the latest file
        bronze_df = spark.read.csv(latest_file_path, header=True, inferSchema=True)
        
        # Validate input data
        if bronze_df.count() == 0:
            logger.warning(f"Empty dataset for {ticker}")
            return None
            
        # Check for required columns
        required_columns = ["timestamp", "open", "high", "low", "close", "volume"]
        missing_columns = [col for col in required_columns if col not in bronze_df.columns]
        if missing_columns:
            logger.error(f"Missing required columns for {ticker}: {missing_columns}")
            return None
        
        # Transform to silver schema
        silver_df = bronze_df.withColumn("ticker", lit(ticker)) \
            .withColumn("trade_date", to_date(col("timestamp"))) \
            .withColumnRenamed("open", "open_price") \
            .withColumnRenamed("high", "high_price") \
            .withColumnRenamed("low", "low_price") \
            .withColumnRenamed("close", "close_price") \
            .withColumn("processing_timestamp", lit(current_timestamp).cast(TimestampType())) \
            .withColumn("effective_from", to_date(col("timestamp"))) \
            .withColumn("effective_to", lit(None).cast(DateType())) \
            .withColumn("source_system", lit("AlphaVantage")) \
            .withColumn("source_file", lit(latest_file_path)) \
            .withColumn("batch_id", lit(batch_id)) \
            .withColumn("record_id", expr("uuid()"))
        
        # Calculate completeness score
        silver_df = silver_df.withColumn("completeness_score",
            (when(col("open_price").isNotNull(), 1).otherwise(0) +
             when(col("high_price").isNotNull(), 1).otherwise(0) +
             when(col("low_price").isNotNull(), 1).otherwise(0) +
             when(col("close_price").isNotNull(), 1).otherwise(0) +
             when(col("volume").isNotNull(), 1).otherwise(0)) / 5.0
        )
        
        # Data confidence score
        silver_df = silver_df.withColumn("data_confidence", col("completeness_score"))
        
        # Select only the columns in our silver schema
        result_df = silver_df.select([
            "ticker", "trade_date", "open_price", "high_price", "low_price", "close_price", "volume",
            "processing_timestamp", "effective_from", "effective_to",
            "source_system", "source_file", "batch_id", "record_id",
            "completeness_score", "data_confidence"
        ])
        
        # Count records
        record_count = result_df.count()
        logger.info(f"Processed {record_count} records for {ticker}")
        
        return result_df
    except Exception as e:
        logger.error(f"Error processing {ticker}: {e}")
        return None

# Main function to run the processing
def main():
    try:
        # Initialize Spark
        spark = init_spark()
        
        # Define tickers to process
        tickers = ["AAPL", "MSFT", "GOOGL"]
        logger.info(f"Will process these tickers: {', '.join(tickers)}")
        
        # Process each ticker
        for ticker in tickers:
            silver_df = process_stock_ticker(spark, ticker)
            
            if silver_df is not None:
                # Show sample of the data
                logger.info(f"Sample of Silver Layer for {ticker}:")
                silver_df.show(2, truncate=False)
                
                # Save to silver layer
                silver_output_path = f"s3a://{bucket_name}/silver/stock_data/{ticker}"
                logger.info(f"Saving to {silver_output_path}")
                
                # Save as partitioned parquet files
                silver_df.write.parquet(silver_output_path, mode="overwrite")
                logger.info(f"Successfully saved {ticker} data to silver layer")
    except Exception as e:
        logger.error(f"Error in main processing: {e}")
        sys.exit(1)
    finally:
        # Stop Spark session
        spark.stop()

if __name__ == "__main__":
    main()