import pandas as pd
import pandera.pandas as pa
from pandera.typing import DataFrame, Series
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
from datetime import datetime
import boto3
import json
import logging
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Schema definition for validation
class OpenWeatherMapSilverSchema(pa.DataFrameModel):
    # Temporal fields
    observation_datetime: Series[pd.Timestamp] = pa.Field(description="Weather observation time")
    ingestion_datetime: Series[pd.Timestamp] = pa.Field(description="Data ingestion timestamp")
    year: Series[int] = pa.Field(ge=2020, le=2030, description="Year for partitioning")
    
    # Location fields  
    city: Series[str] = pa.Field(description="City name")
    latitude: Series[float] = pa.Field(ge=-90, le=90)
    longitude: Series[float] = pa.Field(ge=-180, le=180)
    
    # Weather measurements
    temperature_celsius: Series[float] = pa.Field(nullable=True)
    feels_like_celsius: Series[float] = pa.Field(nullable=True)
    pressure_hpa: Series[int] = pa.Field(nullable=True, ge=0)
    humidity_percent: Series[int] = pa.Field(nullable=True, ge=0, le=100)
    
    # Data quality and lineage
    data_confidence_score: Series[float] = pa.Field(ge=0.0, le=1.0, description="Confidence score based on field completeness")
    source_system: Series[str] = pa.Field(description="Source system identifier")
    
    class Config:
        strict = True

def init_spark(aws_config: Dict[str, str] = None):
    """Initialize the Spark session with proper S3 configurations"""
    
    spark = SparkSession.builder \
        .appName("WeatherDataProcessor") \
        .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:3.3.4,com.amazonaws:aws-java-sdk-bundle:1.12.262") \
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem") \
        .config("spark.hadoop.fs.s3a.access.key", aws_config['access_key']) \
        .config("spark.hadoop.fs.s3a.secret.key", aws_config['secret_key']) \
        .config("spark.hadoop.fs.s3a.endpoint", f"s3.{aws_config['region']}.amazonaws.com") \
        .config("spark.hadoop.fs.s3a.path.style.access", "true") \
        .config("spark.sql.sources.partitionOverwriteMode", "dynamic") \
        .getOrCreate()
    
    spark.sparkContext.setLogLevel("WARN")
    return spark

def calculate_confidence_score(df):
    """Calculate data confidence score based on field completeness"""
    
    # Define critical fields and their weights
    field_weights = {
        'temp': 0.3,           # Temperature is most important
        'pressure': 0.2,       # Pressure is important for weather analysis
        'humidity': 0.2,       # Humidity is important
        'feels_like': 0.1,     # Nice to have
        'wind_speed': 0.1,     # Nice to have
        'visibility': 0.1      # Nice to have
    }
    
    # Create confidence score calculation
    confidence_expr = lit(0.0)
    for field, weight in field_weights.items():
        confidence_expr = confidence_expr + when(col(field).isNotNull(), lit(weight)).otherwise(lit(0.0))
    
    # Round to 2 decimal places to avoid floating point precision issues
    return df.withColumn("data_confidence_score", round(confidence_expr, 2))

def transform_owm_bronze_to_silver(spark, input_path, output_path):
    """Transform OpenWeatherMap data from bronze to silver layer"""
    
    logger.info(f"Starting transformation from {input_path} to {output_path}")
    
    # Read bronze data
    df = spark.read.option("multiline", "true").json(input_path)
    
    logger.info(f"Read {df.count()} records from bronze layer")
    
    # Transform the data structure
    transformed_df = df.select(
        # Convert Unix timestamp to proper datetime
        timestamp_seconds(col("dt")).alias("observation_datetime"),
        col("_metadata.ingestion_timestamp").cast("timestamp").alias("ingestion_datetime"),
        year(timestamp_seconds(col("dt"))).cast("long").alias("year"),  # For partitioning
        
        # Location fields
        col("_metadata.city").alias("city"),
        col("lat").alias("latitude"),
        col("lon").alias("longitude"),
        
        # Weather measurements
        col("temp").alias("temperature_celsius"),
        col("feels_like").alias("feels_like_celsius"),
        col("pressure").cast("integer").alias("pressure_hpa"),  # Explicitly cast to integer
        col("humidity").alias("humidity_percent"),
        
        # Keep original fields for confidence calculation
        col("temp"), col("pressure"), col("humidity"), 
        col("feels_like"), col("wind_speed"), col("visibility"),
        
        # Lineage
        col("_metadata.source").alias("source_system")
    )

    # Calculate confidence score
    transformed_df = calculate_confidence_score(transformed_df)
    
    # Drop the helper columns we used for confidence calculation
    final_df = transformed_df.drop("temp", "pressure", "humidity", "feels_like", "wind_speed", "visibility")
    
    logger.info("Applied transformations and calculated confidence scores")
    
    # Write to silver layer partitioned by city and year
    final_df.write \
        .mode("overwrite") \
        .partitionBy("city", "year") \
        .parquet(output_path)
    
    logger.info(f"Successfully wrote {final_df.count()} records to {output_path}")
    
    return final_df

def validate_silver_data(df_pandas):
    """Validate the transformed data using Pandera. If nulls are present in integer columns, cast to pandas 'Int64' before validation."""
    try:
        # If pressure_hpa or humidity_percent exists and is not Int64, cast to Int64 for Pandera
        for colname in ["pressure_hpa", "humidity_percent"]:
            if colname in df_pandas.columns and not pd.api.types.is_integer_dtype(df_pandas[colname]):
                df_pandas[colname] = df_pandas[colname].astype('Int64')
        OpenWeatherMapSilverSchema.validate(df_pandas)
        logger.info("Data validation passed!")
        return True
    except pa.errors.SchemaError as e:
        logger.error(f"Data validation failed: {e}")
        return False

if __name__ == "__main__":
    # AWS configuration (you'll need to set these)
    aws_config = {
        'access_key': os.getenv('TEST_AWS_ACCESS_KEY'),
        'secret_key': os.getenv('TEST_AWS_SECRET_KEY'), 
        'bucket_name': os.getenv('TEST_AWS_S3_BUCKET_NAME'),
        'region': os.getenv('TEST_AWS_DEFAULT_REGION'),
    }
    
    # Initialize Spark
    spark = init_spark(aws_config)
    
    # Define paths
    bronze_path = "s3a://mock-datalake1/bronze/openweathermap/"
    silver_path = "s3a://mock-datalake1/silver/openweathermap/"
    
    # Transform data
    result_df = transform_owm_bronze_to_silver(spark, bronze_path, silver_path)
    
    # Sample validation (convert small sample to pandas for validation)
    sample_df = result_df.limit(100).toPandas()
    validate_silver_data(sample_df)
    
    spark.stop()