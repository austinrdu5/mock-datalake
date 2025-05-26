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
from typing import Optional, Dict, Mapping
import pandera as pa
from pandera.errors import SchemaError
from pandera import Column
import pandas as pd

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    force=True  # This ensures we can reconfigure logging
)
logger = logging.getLogger(__name__)

# Initialize Spark session with S3 configuration
def init_spark(aws_config: Mapping[str, str]) -> SparkSession:
    """Initialize the Spark session with proper S3 configurations"""
    
    spark = (SparkSession.builder  
        .appName("StockDataProcessor")  # type: ignore
        .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:3.3.4,com.amazonaws:aws-java-sdk-bundle:1.12.262")
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
        .config("spark.hadoop.fs.s3a.access.key", aws_config['access_key'])
        .config("spark.hadoop.fs.s3a.secret.key", aws_config['secret_key'])
        .config("spark.hadoop.fs.s3a.endpoint", f"s3.{aws_config['region']}.amazonaws.com")
        .config("spark.hadoop.fs.s3a.path.style.access", "true")
        .config("spark.sql.sources.partitionOverwriteMode", "dynamic")
        .getOrCreate())
    
    # Set log level to reduce verbosity
    spark.sparkContext.setLogLevel("WARN")
    
    return spark

# Function to get latest files for all tickers in bronze/alphavantage/
def get_latest_files(aws_config: Mapping[str, str], specific_tickers: Optional[list] = None) -> Dict[str, str]:
    """
    Get the latest file path for each ticker in bronze/alphavantage/
    Args:
        aws_config: Dictionary containing AWS configuration
        specific_tickers: Optional list of tickers to filter for. If None, process all tickers.
    Returns:
        Dictionary mapping ticker to its latest file path
    """
    try:
        s3 = boto3.client(
            's3',
            aws_access_key_id=aws_config['access_key'],
            aws_secret_access_key=aws_config['secret_key'],
            region_name=aws_config['region']
        )
        
        # List all tickers (folders)
        prefix = "bronze/alphavantage/"
        response = s3.list_objects_v2(Bucket=aws_config['bucket_name'], Prefix=prefix, Delimiter="/")
        
        ticker_files = {}
        for cp in response.get('CommonPrefixes', []):
            ticker_prefix = cp.get('Prefix')  # e.g., 'bronze/alphavantage/AAPL/'
            ticker = ticker_prefix.split('/')[2]  # Extract ticker name
            
            # Skip if we're filtering for specific tickers and this one isn't in the list
            if specific_tickers and ticker not in specific_tickers:
                continue
            
            # List objects in ticker's directory
            ticker_response = s3.list_objects_v2(
                Bucket=aws_config['bucket_name'],
                Prefix=ticker_prefix
            )
            
            if 'Contents' not in ticker_response:
                logger.warning(f"No files found for {ticker}")
                continue
            
            # Get all CSV files and sort them
            files = [obj['Key'] for obj in ticker_response['Contents'] if obj['Key'].endswith('.csv')]
            if not files:
                logger.warning(f"No CSV files found for {ticker}")
                continue
            
            # Sort by filename (which contains timestamp) and get the latest
            files.sort()
            latest_file = files[-1]
            ticker_files[ticker] = f"s3a://{aws_config['bucket_name']}/{latest_file}"
            
        return ticker_files
    except Exception as e:
        logger.error(f"Error getting latest files: {e}")
        return {}

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
        if null_counts is None:
            logger.error(f"No data found for {ticker}")
            return None
            
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

        # Convert to pandas DataFrame for Pandera validation
        result_pd = result_df.toPandas()

        # Ensure datetime columns are correct dtype for Pandera
        for dt_col in ["trade_date", "processing_timestamp", "effective_from", "effective_to"]:
            if dt_col in result_pd.columns:
                result_pd[dt_col] = pd.to_datetime(result_pd[dt_col], errors='coerce')

        # Define Pandera schema for output validation
        silver_schema = pa.DataFrameSchema({
            "ticker": Column(str),
            "trade_date": Column(pa.DateTime, nullable=True),
            "open_price": Column(float, nullable=True),
            "high_price": Column(float, nullable=True),
            "low_price": Column(float, nullable=True),
            "close_price": Column(float, nullable=True),
            "volume": Column(float, nullable=True),
            "processing_timestamp": Column(pa.DateTime),
            "effective_from": Column(pa.DateTime, nullable=True),
            "effective_to": Column(pa.DateTime, nullable=True),
            "source_system": Column(str),
            "source_file": Column(str),
            "batch_id": Column(str),
            "record_id": Column(str),
            "completeness_score": Column(float),
            "data_confidence": Column(float)
        })

        # Validate output
        try:
            silver_schema.validate(result_pd)
            logger.info(f"Pandera output validation passed for {ticker}")
        except SchemaError as e:
            logger.error(f"Pandera output validation failed for {ticker}: {e}")
            return None

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
    parser.add_argument('--tickers', nargs='+', help='Specific tickers to process (e.g., AAPL MSFT GOOGL)')
    
    args = parser.parse_args()
    
    try:
        # Get credentials based on test mode
        prefix = "TEST_" if args.use_test_bucket else ""
        aws_config: Dict[str, str] = {
            'access_key': os.environ.get(f'{prefix}AWS_ACCESS_KEY', ''),
            'secret_key': os.environ.get(f'{prefix}AWS_SECRET_KEY', ''),
            'bucket_name': os.environ.get(f'{prefix}AWS_S3_BUCKET_NAME', ''),
            'region': os.environ.get(f'{prefix}AWS_DEFAULT_REGION', '')
        }

        # Log AWS configuration (last 4 chars of keys for security)
        logger.info(f"AWS Access Key (last 4): ...{aws_config['access_key'][-4:] if aws_config['access_key'] else 'None'}")
        logger.info(f"AWS Secret Key (last 4): ...{aws_config['secret_key'][-4:] if aws_config['secret_key'] else 'None'}")
        logger.info(f"AWS Region: {aws_config['region']}")
        logger.info(f"AWS Bucket: {aws_config['bucket_name']}")

        # Validate AWS credentials
        if any(not aws_config[key] for key in aws_config):
            logger.error("Missing AWS credentials or bucket name")
            sys.exit(1)
        
        # Initialize Spark
        spark = init_spark(aws_config)
        
        # Get latest files for specified tickers or all tickers
        ticker_files = get_latest_files(aws_config, args.tickers)
        logger.info(f"Found {len(ticker_files)} tickers to process")
        
        if args.tickers and not ticker_files:
            logger.error(f"No files found for specified tickers: {args.tickers}")
            sys.exit(1)
        
        # Process each ticker's latest file
        for ticker, file_path in ticker_files.items():
            logger.info(f"Processing {ticker} from {file_path}")
            
            # Process the ticker
            silver_df = process_stock_ticker(spark, ticker, file_path)
            
            if silver_df is not None:
                # Save to silver layer
                silver_output_path = f"s3a://{aws_config['bucket_name']}/silver/alphavantage/{ticker}"
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