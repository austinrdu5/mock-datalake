from typing import Dict, Optional, Any
import pandas as pd
import pandera.pandas as pa
from pandera.errors import SchemaError
from pandera.typing import DataFrame, Series
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
from datetime import datetime, date
import boto3
import json
import logging
import os
import argparse

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Schema definition for silver data validation
class OpenWeatherMapSilverSchema(pa.DataFrameModel):
    # Temporal fields
    observation_date: Series[pd.Timestamp] = pa.Field(description="Weather observation date (time portion should be midnight)")
    ingestion_datetime: Series[pd.Timestamp] = pa.Field(description="Data ingestion timestamp")
    year: Series[int] = pa.Field(ge=2020, le=2030, description="Year for partitioning")
    
    # Location fields  
    city: Series[str] = pa.Field(description="City name")
    latitude: Series[float] = pa.Field(ge=-90, le=90)
    longitude: Series[float] = pa.Field(ge=-180, le=180)
    
    # Weather measurements
    temperature_celsius: Series[float] = pa.Field(nullable=True)
    pressure_hpa: Series[int] = pa.Field(nullable=True, ge=0)
    humidity_percent: Series[int] = pa.Field(nullable=True, ge=0, le=100)
    cloud_cover_percent: Series[float] = pa.Field(nullable=True, ge=0.0, le=100.0)
    precipitation_mm: Series[float] = pa.Field(nullable=True, ge=0)
    
    # Data quality and lineage
    data_confidence_score: Series[float] = pa.Field(ge=0.0, le=1.0, description="Confidence score based on field completeness")
    source_system: Series[str] = pa.Field(description="Source system identifier")
    
    class Config:
        strict = True

def init_spark(aws_config: Dict[str, str]) -> SparkSession:
    """Initialize the Spark session with Delta Lake and S3 configurations"""
    
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
    
    spark = builder.appName("WeatherDataProcessor").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    
    logger.info("Initialized Spark session with Delta Lake support")
    return spark

def calculate_confidence_score(df):
    """Calculate data confidence score based on field completeness"""
    
    # Define critical fields and their weights
    field_weights = {
        'temperature_celsius': 0.6,  # Temperature is most important
        'pressure_hpa': 0.1,         # Pressure is important for weather analysis
        'humidity_percent': 0.1,     # Humidity is important
        'cloud_cover_percent': 0.1,  # Cloud cover is important
        'precipitation_mm': 0.1      # Precipitation is nice to have
    }
    
    # Create confidence score calculation
    confidence_expr = lit(0.0)
    for field, weight in field_weights.items():
        confidence_expr = confidence_expr + when(col(field).isNotNull(), lit(weight)).otherwise(lit(0.0))
    
    # Round to 2 decimal places to avoid floating point precision issues
    return df.withColumn("data_confidence_score", round(confidence_expr, 2))

def validate_silver_data(df_pandas):
    """Validate the transformed data using Pandera. If nulls are present in integer columns, cast to pandas 'Int64' before validation."""
    try:
        # Convert observation_date to datetime64[ns] if it's not already
        if 'observation_date' in df_pandas.columns and not pd.api.types.is_datetime64_any_dtype(df_pandas['observation_date']):
            df_pandas['observation_date'] = pd.to_datetime(df_pandas['observation_date'])
            
        # If pressure_hpa or humidity_percent exists and is not Int64, cast to Int64 for Pandera
        for colname in ["pressure_hpa", "humidity_percent"]:
            if colname in df_pandas.columns and not pd.api.types.is_integer_dtype(df_pandas[colname]):
                df_pandas[colname] = df_pandas[colname].astype('Int64')
        OpenWeatherMapSilverSchema.validate(df_pandas)
        logger.info("Silver data validation passed!")
        return True
    except SchemaError as e:
        logger.error(f"Silver data validation failed: {e}")
        return False
    
def transform_owm_bronze_to_silver(spark, input_paths, output_path):
    """Transform OpenWeatherMap data from bronze to silver layer using Delta Lake"""
    
    logger.info(f"Starting transformation from {input_paths} to {output_path}")
    
    # Read bronze data - using wildcard to read all JSON files in the directory
    df = spark.read.option("multiline", "true").json(input_paths)
    
    logger.info(f"Read {df.count()} records from bronze layer")
    
    # Explode the data array to get individual weather records
    df = df.select(
        explode("data.data").alias("weather_data"),
        col("data.metadata").alias("metadata")
    )
    
    # Transform the data structure, handling nested fields
    transformed_df = df.select(
        # Convert date string to date
        to_date(col("weather_data.date")).alias("observation_date"),
        col("metadata.ingestion_timestamp").cast("timestamp").alias("ingestion_datetime"),
        col("metadata.year").alias("year"),  # Rename to year but keep as integer
        
        # Location fields
        regexp_replace(lower(col("metadata.city")), " ", "_").alias("city"),  # Lowercase and replace spaces with underscores
        col("weather_data.lat").alias("latitude"),
        col("weather_data.lon").alias("longitude"),
        
        # Weather measurements - accessing nested fields
        col("weather_data.temperature.afternoon").alias("temperature_celsius"),
        col("weather_data.pressure.afternoon").cast("bigint").alias("pressure_hpa"),  # Cast to int64
        col("weather_data.humidity.afternoon").alias("humidity_percent"),
        col("weather_data.cloud_cover.afternoon").cast("double").alias("cloud_cover_percent"),  # Cast to float64
        col("weather_data.precipitation.total").alias("precipitation_mm"),
        
        # Lineage
        col("metadata.source").alias("source_system")
    )

    # Calculate confidence score
    final_df = calculate_confidence_score(transformed_df)
        
    logger.info("Applied transformations and calculated confidence scores")
    
    # Validate the final silver data
    sample_df = final_df.limit(100).toPandas()
    if not validate_silver_data(sample_df):
        logger.error("Silver data validation failed. Aborting transformation.")
        return None
    
    # Write to Delta Lake with partitioning
    try:
        (final_df.write
         .format("delta")
         .mode("append")  # Append new data
         .partitionBy("city", "year")  # Partition by city and year for efficient querying
         .save(output_path))
        
        logger.info(f"Successfully wrote {final_df.count()} records to Delta Lake at {output_path}")
        return final_df
        
    except Exception as e:
        logger.error(f"Failed to write to Delta Lake: {e}")
        return None

if __name__ == "__main__":
    import argparse
    
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Transform OpenWeatherMap data from bronze to silver layer')
    parser.add_argument('--cities', nargs='+', default=['London'],
                      help='List of cities to process (default: London). Use "all" to process all available cities.')
    parser.add_argument('--test-mode', action='store_true',
                      help='Run in test mode using local paths')
    args = parser.parse_args()
    
    # AWS configuration (you'll need to set these)
    aws_config: Dict[str, str] = {
        'access_key': os.getenv('AWS_ACCESS_KEY', ''),
        'secret_key': os.getenv('AWS_SECRET_KEY', ''), 
        'bucket_name': os.getenv('AWS_S3_BUCKET_NAME', ''),
        'region': os.getenv('AWS_DEFAULT_REGION', 'us-east-1')
    }
    
    # Initialize Spark
    spark = init_spark(aws_config)

    # Construct city-specific paths for reading
    if "all" in args.cities:
        # If "all" is specified, use a wildcard pattern to match all cities and years
        bronze_paths = ["s3a://mock-datalake1/bronze/openweathermap/city=*/year=*/"]
        logger.info("Processing all available cities and years")
    else:
        # Process specific cities with all years
        cities = [city.lower().replace(' ', '_') for city in args.cities]
        bronze_paths = [f"s3a://mock-datalake1/bronze/openweathermap/city={city}/year=*/" for city in cities]
        logger.info(f"Processing cities {cities} for all available years")

    if args.test_mode:
        silver_path = "s3a://mock-datalake1-test/silver/openweathermap/"
        logger.info("Test mode: Output files will go to test bucket")
    else:
        silver_path = "s3a://mock-datalake1/silver/openweathermap/"

    logger.info(f"Reading data from paths: {bronze_paths}")
    
    # Transform data
    result_df = transform_owm_bronze_to_silver(spark, bronze_paths, silver_path)
    
    # Sample validation (convert small sample to pandas for validation)
    if result_df is not None:
        sample_df = result_df.limit(100).toPandas()
        validate_silver_data(sample_df)
    else:
        logger.error("Transformation failed - no data to validate")
    
    spark.stop()