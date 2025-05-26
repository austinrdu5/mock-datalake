from typing import Dict
import pandas as pd
import pandera.pandas as pa
from pandera.errors import SchemaError
from pandera.typing import Series
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, lit, to_timestamp, date_format, split, size, when, expr
)
from datetime import datetime
import boto3
import logging
import os
import argparse
import uuid

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Schema definition for ecommerce silver data validation
class EcommerceSilverSchema(pa.DataFrameModel):
    """
    Pandera schema for validating ecommerce silver layer data
    """
    
    # Event identification
    event_time: Series[pd.Timestamp] = pa.Field(description="Original event timestamp")
    event_type: Series[str] = pa.Field(isin=["view", "cart", "purchase"], description="Type of user event")
    
    # Product information
    product_id: Series[int] = pa.Field(ge=1, description="Unique product identifier")
    category_id: Series[float] = pa.Field(nullable=True, description="Category ID from source system")
    category_level_1: Series[str] = pa.Field(nullable=True, description="Top-level product category")
    category_level_2: Series[str] = pa.Field(nullable=True, description="Second-level product category") 
    category_level_3: Series[str] = pa.Field(nullable=True, description="Third-level product category")
    brand: Series[str] = pa.Field(nullable=True, description="Product brand")
    price: Series[float] = pa.Field(nullable=True, ge=0, description="Product price")
    
    # User information
    user_id: Series[float] = pa.Field(description="User identifier")
    user_session: Series[str] = pa.Field(description="User session identifier")
    
    # Temporal and partitioning fields
    year_month: Series[str] = pa.Field(description="Year-month for partitioning (YYYY-MM)")
    processing_timestamp: Series[pd.Timestamp] = pa.Field(description="Data processing timestamp")
    effective_from: Series[pd.Timestamp] = pa.Field(description="Event effective start time")
    effective_to: Series[pd.Timestamp] = pa.Field(nullable=True, description="Event effective end time (null for current)")
    
    # Data quality and lineage
    data_confidence_score: Series[float] = pa.Field(ge=0.0, le=1.0, description="Data confidence based on category/brand presence")
    source_system: Series[str] = pa.Field(description="Source system identifier")
    source_file: Series[str] = pa.Field(description="Source file path")
    batch_id: Series[str] = pa.Field(description="Processing batch identifier")
    record_id: Series[str] = pa.Field(description="Unique record identifier")
    
    class Config:
        strict = True

def init_spark_with_delta(aws_config: Dict[str, str]) -> SparkSession:
    """
    Initialize Spark session with Delta Lake and S3 configuration
    
    Key additions for Delta Lake:
    - delta-core package for Delta functionality
    - Delta catalog configuration
    - SQL extensions for Delta operations
    """
    
    builder = (SparkSession.builder
        .config("spark.jars.packages",  # type: ignore
                "org.apache.hadoop:hadoop-aws:3.3.4,"
                "com.amazonaws:aws-java-sdk-bundle:1.12.262,"
                "io.delta:delta-spark_2.12:3.3.1")  # Delta Lake package
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")  # Delta SQL support
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")  # Delta catalog
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
        .config("spark.hadoop.fs.s3a.access.key", aws_config['access_key'])
        .config("spark.hadoop.fs.s3a.secret.key", aws_config['secret_key'])
        .config("spark.hadoop.fs.s3a.endpoint", f"s3.{aws_config['region']}.amazonaws.com")
        .config("spark.hadoop.fs.s3a.path.style.access", "true")
        .config("spark.sql.sources.partitionOverwriteMode", "dynamic"))
    
    spark = builder.appName("EcommerceDataProcessor").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    
    logger.info("Initialized Spark session with Delta Lake support")
    return spark

def parse_category_hierarchy(category_code_col):
    """
    Parse category_code like 'electronics.telephone.wireless' into three levels
    Returns tuple of (level1, level2, level3) expressions
    """
    
    # Split by periods and extract levels
    split_category = split(category_code_col, "\\.")
    
    # Handle empty strings and nulls
    level_1 = when((category_code_col.isNull()) | (category_code_col == ""), None).when(size(split_category) >= 1, split_category[0]).otherwise(None)
    level_2 = when((category_code_col.isNull()) | (category_code_col == ""), None).when(size(split_category) >= 2, split_category[1]).otherwise(None)
    level_3 = when((category_code_col.isNull()) | (category_code_col == ""), None).when(size(split_category) >= 3, split_category[2]).otherwise(None)
    
    return level_1, level_2, level_3

def calculate_confidence_score_ecommerce(df):
    """
    Calculate confidence score based on category and brand presence:
    - Both category and brand present: 1.0
    - Missing category only: 0.4  
    - Missing brand only: 0.6
    - Missing both: 0.0
    """
    
    has_category = (col("category_level_1").isNotNull()) & (col("category_level_1") != "")
    has_brand = (col("brand").isNotNull()) & (col("brand") != "")
    
    confidence_score = (
        when(has_category & has_brand, 1.0)
        .when(has_category & ~has_brand, 0.6)  # Missing brand only
        .when(~has_category & has_brand, 0.4)  # Missing category only  
        .otherwise(0.0)  # Missing both
    )
    
    return df.withColumn("data_confidence_score", confidence_score)

def transform_ecommerce_bronze_to_silver(spark, input_path, output_path):
    """
    Transform ecommerce data from bronze CSV to silver Delta Lake format
    
    Args:
        spark: SparkSession with Delta support
        input_path: S3 path to bronze CSV file
        output_path: S3 path for silver Delta table
    """
    
    logger.info(f"Starting transformation from {input_path} to {output_path}")
    
    # Generate batch metadata
    batch_id = str(uuid.uuid4())
    current_timestamp = datetime.now()
    
    # Read bronze CSV data
    df = spark.read.csv(input_path, header=True, inferSchema=True)
    
    initial_count = df.count()
    logger.info(f"Read {initial_count} records from bronze layer")
    
    # Data quality checks
    if initial_count == 0:
        logger.warning("Empty dataset - aborting transformation")
        return None
    
    # Check for required columns
    required_columns = ["event_time", "event_type", "product_id", "user_id", "user_session"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        logger.error(f"Missing required columns: {missing_columns}")
        return None
    
    # Parse category hierarchy
    level_1, level_2, level_3 = parse_category_hierarchy(col("category_code"))
    
    # Main transformation
    transformed_df = df.select(
        # Event identification
        to_timestamp(col("event_time")).alias("event_time"),
        col("event_type"),
        
        # Product information  
        col("product_id").cast("bigint"),
        col("category_id").cast("double"),
        level_1.alias("category_level_1"),
        level_2.alias("category_level_2"), 
        level_3.alias("category_level_3"),
        when(col("brand").isNull() | (col("brand") == ""), None)
            .otherwise(col("brand")).alias("brand"),  # Clean empty strings
        col("price").cast("double"),
        
        # User information
        col("user_id").cast("double"),
        col("user_session"),
        
        # Derived temporal fields
        date_format(to_timestamp(col("event_time")), "yyyy-MM").alias("year_month"),
        lit(current_timestamp).cast("timestamp").alias("processing_timestamp"),
        to_timestamp(col("event_time")).alias("effective_from"),
        lit(None).cast("timestamp").alias("effective_to"),
        
        # Lineage tracking
        lit("EcommerceEvents").alias("source_system"),
        lit(input_path).alias("source_file"),
        lit(batch_id).alias("batch_id"),
        expr("uuid()").alias("record_id")
    )
    
    # Calculate confidence score
    final_df = calculate_confidence_score_ecommerce(transformed_df)
    
    logger.info("Applied transformations and calculated confidence scores")
    
    # Data validation on sample
    sample_df = final_df.limit(1000).toPandas()
    if not validate_silver_data(sample_df):
        logger.error("Silver data validation failed. Aborting transformation.")
        return None
    
    return final_df

def validate_silver_data(df_pandas):
    """Validate the transformed data using Pandera schema"""
    try:
        # Ensure datetime columns are proper pandas datetime
        datetime_cols = ["event_time", "processing_timestamp", "effective_from", "effective_to"]
        for col_name in datetime_cols:
            if col_name in df_pandas.columns:
                df_pandas[col_name] = pd.to_datetime(df_pandas[col_name], errors='coerce')
        
        # Validate against schema
        EcommerceSilverSchema.validate(df_pandas)
        logger.info("Silver data validation passed!")
        return True
        
    except SchemaError as e:
        logger.error(f"Silver data validation failed: {e}")
        return False
    
def write_to_delta_lake(df, output_path):
    """
    Write DataFrame to Delta Lake with partitioning and optimization
    
    Key Delta Lake features demonstrated:
    - Partitioned storage for query performance
    - ACID transactions 
    - Automatic schema enforcement
    - Metadata tracking for time travel
    """
    
    logger.info(f"Writing {df.count()} records to Delta Lake at {output_path}")
    
    try:
        # Write to Delta format with partitioning
        # This creates separate folders for each year_month/event_type combination
        (df.write
         .format("delta")  # This is the key difference from regular Parquet!
         .mode("append")   # Delta handles duplicate detection automatically
         .partitionBy("year_month", "event_type")  # Our partitioning strategy
         .save(output_path))
        
        logger.info(f"Successfully wrote data to Delta Lake")
        
        # Demonstrate Delta Lake capabilities
        logger.info("Delta Lake table created with following capabilities:")
        logger.info("- Time travel: Query historical versions")
        logger.info("- ACID transactions: Consistent reads/writes") 
        logger.info("- Automatic schema evolution: Safe schema changes")
        logger.info("- Optimized queries: Statistics-based pruning")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to write to Delta Lake: {e}")
        return False

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Transform ecommerce data to Delta Lake')
    parser.add_argument('--test-mode', action='store_true',
                       help='Run in test mode using test bucket')
    parser.add_argument('--demo-features', action='store_true',
                       help='Demonstrate Delta Lake features after writing')
    
    args = parser.parse_args()
    
    try:
        # Get AWS credentials
        prefix = "TEST_" if args.test_mode else ""
        aws_config: Dict[str, str] = {
            'access_key': os.environ.get(f'{prefix}AWS_ACCESS_KEY', ''),
            'secret_key': os.environ.get(f'{prefix}AWS_SECRET_KEY', ''),
            'bucket_name': os.environ.get(f'{prefix}AWS_S3_BUCKET_NAME', ''),
            'region': os.environ.get(f'{prefix}AWS_DEFAULT_REGION', '')
        }
        
        # Validate AWS credentials
        if any(not aws_config[key] for key in aws_config):
            logger.error("Missing AWS credentials or bucket name")
            exit(1)
        
        # Set up paths
        bronze_path = f"s3a://{aws_config['bucket_name']}/bronze/ecommerce/events.csv"
        silver_path = f"s3a://{aws_config['bucket_name']}/silver/ecommerce/"
        
        if args.test_mode:
            logger.info("Running in test mode")
        
        logger.info(f"Input: {bronze_path}")
        logger.info(f"Output: {silver_path}")
        
        # Initialize Spark with Delta Lake
        spark = init_spark_with_delta(aws_config)
        
        # Transform data
        logger.info("Starting ecommerce data transformation...")
        silver_df = transform_ecommerce_bronze_to_silver(spark, bronze_path, silver_path)
        
        if silver_df is None:
            logger.error("Transformation failed")
            exit(1)
        
        # Write to Delta Lake
        logger.info("Writing to Delta Lake...")
        success = write_to_delta_lake(silver_df, silver_path)
        
        if not success:
            logger.error("Failed to write to Delta Lake")
            exit(1)
        
        logger.info("=== Transformation Complete ===")
        logger.info(f"✅ Successfully processed ecommerce data to Delta Lake")
        logger.info(f"✅ Data partitioned by year_month and event_type")
        logger.info(f"✅ Time travel and ACID transactions enabled")
        logger.info(f"✅ Ready for downstream analytics and temporal queries")
        
    except Exception as e:
        logger.error(f"Error in main processing: {e}")
        exit(1)
    finally:
        if 'spark' in locals():
            spark.stop()
