from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, lit, to_date, when, expr, sum
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

# Initialize Spark session with S3 configuration
def init_spark(aws_config: Optional[Dict[str, str]] = None):
    """Initialize the Spark session with proper S3 configurations"""
    if aws_config is None:
        aws_config = {
            'access_key': os.environ.get('AWS_ACCESS_KEY'),
            'secret_key': os.environ.get('AWS_SECRET_KEY'),
            'bucket_name': os.environ.get('AWS_S3_BUCKET_NAME'),
            'region': os.environ.get('AWS_DEFAULT_REGION', 'us-east-1')
        }
    
    # Validate AWS credentials
    if not all([aws_config['access_key'], aws_config['secret_key']]):
        logger.error("AWS credentials not found in environment variables")
        logger.error("Please set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY")
        sys.exit(1)
    
    spark = SparkSession.builder \
        .appName("StockDataProcessor") \
        .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:3.3.4,com.amazonaws:aws-java-sdk-bundle:1.12.262") \
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem") \
        .config("spark.hadoop.fs.s3a.access.key", aws_config['access_key']) \
        .config("spark.hadoop.fs.s3a.secret.key", aws_config['secret_key']) \
        .config("spark.hadoop.fs.s3a.endpoint", f"s3.{aws_config['region']}.amazonaws.com") \
        .config("spark.hadoop.fs.s3a.path.style.access", "true") \
        .config("spark.sql.sources.partitionOverwriteMode", "dynamic") \
        .getOrCreate()
    
    # Set log level to reduce verbosity
    spark.sparkContext.setLogLevel("WARN")
    
    return spark

# Function to get the most recent file for a ticker
def get_latest_file(ticker: str, aws_config: Optional[Dict[str, str]] = None):
    """Get the latest file for a ticker from S3"""
    if aws_config is None:
        aws_config = {
            'access_key': os.environ.get('AWS_ACCESS_KEY'),
            'secret_key': os.environ.get('AWS_SECRET_KEY'),
            'bucket_name': os.environ.get('AWS_S3_BUCKET_NAME'),
            'region': os.environ.get('AWS_DEFAULT_REGION', 'us-east-1')
        }
    
    try:
        # Create boto3 S3 client
        s3 = boto3.client(
            's3',
            aws_access_key_id=aws_config['access_key'],
            aws_secret_access_key=aws_config['secret_key'],
            region_name=aws_config['region']
        )
        
        # List objects in the ticker's directory
        prefix = f"bronze/alphavantage/{ticker}/"
        response = s3.list_objects_v2(Bucket=aws_config['bucket_name'], Prefix=prefix)
        
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
        
        return f"s3a://{aws_config['bucket_name']}/{latest_file}"
    except Exception as e:
        logger.error(f"Error accessing S3 for {ticker}: {e}")
        return None

# Function to process one stock ticker's data
def process_stock_ticker(spark: SparkSession, ticker: str, latest_file_path: str):
    """Process a stock ticker's data"""
    
    logger.info(f"Processing {ticker}...")
    
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
        
        # Validate numeric columns
        numeric_columns = ["open", "high", "low", "close", "volume"]
        for col_name in numeric_columns:
            try:
                bronze_df = bronze_df.withColumn(col_name, col(col_name).cast("double"))
            except Exception as e:
                logger.error(f"Error casting {col_name} to double for {ticker}: {e}")
                return None
        
        # Check for null values in numeric columns
        null_counts = bronze_df.select([sum(col(c).isNull().cast("int")).alias(c) for c in numeric_columns]).first()
        if any(null_counts[c] > 0 for c in numeric_columns):
            logger.error(f"Invalid numeric values found in {ticker}")
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

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Process stock data')
    parser.add_argument('--use-test-bucket', action='store_true', help='Use test S3 bucket and credentials (TEST_ prefixed environment variables)')
    
    args = parser.parse_args()
    
    try:
        # Get credentials based on test mode
        prefix = "TEST_" if args.use_test_bucket else ""
        aws_config = {
            'access_key': os.environ.get(f'{prefix}AWS_ACCESS_KEY'),
            'secret_key': os.environ.get(f'{prefix}AWS_SECRET_KEY'),
            'bucket_name': os.environ.get(f'{prefix}AWS_S3_BUCKET_NAME'),
            'region': os.environ.get(f'{prefix}AWS_DEFAULT_REGION', 'us-east-1')
        }
        
        # Initialize Spark
        spark = init_spark(aws_config)
        
        # Define tickers to process
        tickers = ["AAPL", "MSFT", "GOOGL"]
        logger.info(f"Will process these tickers: {', '.join(tickers)}")
        
        # Process each ticker
        for ticker in tickers:
            # Get the path to the latest file
            latest_file_path = get_latest_file(ticker, aws_config)
            logger.info(f"Latest file for {ticker}: {latest_file_path}")

            # Process the ticker
            silver_df = process_stock_ticker(spark, ticker, latest_file_path)
            
            if silver_df is not None:
                # Show sample of the data
                logger.info(f"Sample of Silver Layer for {ticker}:")
                silver_df.show(2, truncate=False)
                
                # Save to silver layer
                silver_output_path = f"s3a://{aws_config['bucket_name']}/silver/stock_data/{ticker}"
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