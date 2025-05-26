import pytest
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
import sys
import os
import json
from datetime import date

from src.silver.owm_convert import (
    calculate_confidence_score, 
    OpenWeatherMapSilverSchema,
    validate_silver_data
)

@pytest.fixture(scope="session")
def spark():
    """Create a Spark session for testing"""
    spark = (SparkSession.builder
        .appName("OWMConvertTest")  # type: ignore
        .config("spark.sql.execution.arrow.pyspark.enabled", "false")
        .getOrCreate())
    
    spark.sparkContext.setLogLevel("ERROR")  # Reduce noise
    yield spark
    spark.stop()

@pytest.fixture(scope="session")
def bronze_df(spark):
    """Fixture providing a Spark DataFrame with bronze data"""
    schema = StructType([
        StructField("data", StructType([
            StructField("metadata", StructType([
                StructField("city", StringType(), True),
                StructField("year", IntegerType(), True),
                StructField("month", IntegerType(), True),
                StructField("source", StringType(), True),
                StructField("ingestion_timestamp", StringType(), True),
                StructField("record_count", IntegerType(), True)
            ]), True),
            StructField("data", ArrayType(StructType([
                StructField("lat", DoubleType(), True),
                StructField("lon", DoubleType(), True),
                StructField("date", StringType(), True),
                StructField("temperature", StructType([
                    StructField("min", DoubleType(), True),
                    StructField("max", DoubleType(), True),
                    StructField("afternoon", DoubleType(), True),
                    StructField("night", DoubleType(), True),
                    StructField("evening", DoubleType(), True),
                    StructField("morning", DoubleType(), True)
                ]), True),
                StructField("pressure", StructType([
                    StructField("afternoon", IntegerType(), True)
                ]), True),
                StructField("humidity", StructType([
                    StructField("afternoon", IntegerType(), True)
                ]), True),
                StructField("cloud_cover", StructType([
                    StructField("afternoon", IntegerType(), True)
                ]), True),
                StructField("precipitation", StructType([
                    StructField("total", DoubleType(), True)
                ]), True)
            ])), True)
        ]), True)
    ])

    data = [
        {
            "data": {
                "metadata": {
                    "city": "New York",
                    "year": 2024,
                    "month": 1,
                    "source": "openweathermap",
                    "ingestion_timestamp": "2025-05-08T20:56:55.884426",
                    "record_count": 1
                },
                "data": [{
                    "lat": 40.7128,
                    "lon": -74.0060,
                    "date": "2024-01-01",
                    "temperature": {
                        "min": 1.18,
                        "max": 3.74,
                        "afternoon": 3.37,
                        "night": 1.57,
                        "evening": 3.42,
                        "morning": 1.41
                    },
                    "pressure": {"afternoon": 1021},
                    "humidity": {"afternoon": 72},
                    "cloud_cover": {"afternoon": 100},
                    "precipitation": {"total": 1.14}
                }]
            }
        },
        {
            "data": {
                "metadata": {
                    "city": "New York",
                    "year": 2024,
                    "month": 1,
                    "source": "openweathermap",
                    "ingestion_timestamp": "2025-05-08T20:56:55.884426",
                    "record_count": 1
                },
                "data": [{
                    "lat": 40.7128,
                    "lon": -74.0060,
                    "date": "2024-01-02",
                    "temperature": {
                        "min": 1.18,
                        "max": 3.74,
                        "afternoon": 3.37,
                        "night": 1.57,
                        "evening": 3.42,
                        "morning": 1.41
                    },
                    "pressure": {"afternoon": None},
                    "humidity": {"afternoon": 72},
                    "cloud_cover": {"afternoon": None},
                    "precipitation": {"total": 1.14}
                }]
            }
        }
    ]
    
    return spark.createDataFrame(data, schema=schema)

class TestConfidenceScoreCalculation:
    """Test the confidence score calculation logic in isolation with minimal DataFrames"""
    
    def test_perfect_confidence_score(self, spark):
        """Test confidence score with all fields present"""
        data = [{
            "temperature_celsius": 3.37,
            "pressure_hpa": 1021,
            "humidity_percent": 72,
            "cloud_cover_percent": 100,
            "precipitation_mm": 1.14
        }]
        df = spark.createDataFrame(data)
        result = calculate_confidence_score(df)
        confidence = result.select("data_confidence_score").collect()[0][0]
        assert confidence == 1.0
    
    def test_partial_confidence_score(self, spark):
        """Test confidence score with some fields missing"""
        data = [{
            "temperature_celsius": 3.37,
            "pressure_hpa": None,
            "humidity_percent": 72,
            "cloud_cover_percent": None,
            "precipitation_mm": 1.14
        }]
        schema = StructType([
            StructField("temperature_celsius", DoubleType(), True),
            StructField("pressure_hpa", IntegerType(), True),
            StructField("humidity_percent", IntegerType(), True),
            StructField("cloud_cover_percent", IntegerType(), True),
            StructField("precipitation_mm", DoubleType(), True)
        ])
        df = spark.createDataFrame(data, schema=schema)
        result = calculate_confidence_score(df)
        confidence = result.select("data_confidence_score").collect()[0][0]
        # Present: temperature(0.6) + humidity(0.1) + precipitation(0.1) = 0.8
        assert confidence == 0.8

class TestDataTransformation:
    """Test the complete data transformation pipeline"""
    
    def test_bronze_to_silver_transformation(self, spark, bronze_df):
        """Test the complete transformation matches expected schema"""
        # Apply the same transformation as in your main function
        transformed_df = bronze_df.select(
            explode("data.data").alias("weather_data"),
            col("data.metadata").alias("metadata")
        ).select(
            # Convert date string to date
            to_date(col("weather_data.date")).alias("observation_date"),
            col("metadata.ingestion_timestamp").cast("timestamp").alias("ingestion_datetime"),
            col("metadata.year").cast("long").alias("year"),
            
            # Location fields
            col("metadata.city").alias("city"),
            col("weather_data.lat").alias("latitude"),
            col("weather_data.lon").alias("longitude"),
            
            # Weather measurements - accessing nested fields
            col("weather_data.temperature.afternoon").alias("temperature_celsius"),
            col("weather_data.pressure.afternoon").cast("integer").alias("pressure_hpa"),
            col("weather_data.humidity.afternoon").alias("humidity_percent"),
            col("weather_data.cloud_cover.afternoon").alias("cloud_cover_percent"),
            col("weather_data.precipitation.total").alias("precipitation_mm"),
            
            # Lineage
            col("metadata.source").alias("source_system")
        )
        
        # Calculate confidence score
        transformed_df = calculate_confidence_score(transformed_df)
        
        # Verify we have the expected number of records
        assert transformed_df.count() == 2  # We have 2 test records
        
        # Check that required columns exist
        expected_columns = [
            "observation_date", "ingestion_datetime", "year", "city", 
            "latitude", "longitude", "temperature_celsius",
            "pressure_hpa", "humidity_percent", "cloud_cover_percent", 
            "precipitation_mm", "data_confidence_score", "source_system"
        ]
        
        actual_columns = transformed_df.columns
        for col_name in expected_columns:
            assert col_name in actual_columns, f"Missing column: {col_name}"
        
        # Verify the values for the first record
        rows = transformed_df.collect()
        assert rows[0].temperature_celsius == 3.37
        assert rows[0].pressure_hpa == 1021
        assert rows[0].humidity_percent == 72
        assert rows[0].cloud_cover_percent == 100
        assert rows[0].precipitation_mm == 1.14
        
        # Verify the values for the second record
        assert rows[1].temperature_celsius == 3.37
        assert rows[1].pressure_hpa is None
        assert rows[1].humidity_percent == 72
        assert rows[1].cloud_cover_percent is None
        assert rows[1].precipitation_mm == 1.14
        
        # Verify confidence scores
        assert rows[0].data_confidence_score == 1.0  # All fields present
        assert rows[1].data_confidence_score == 0.8  # Some fields missing
        
        # Verify source system
        assert rows[0].source_system == "openweathermap"
        assert rows[1].source_system == "openweathermap"
    
    def test_datetime_conversion(self, spark, bronze_df):
        """Test that date strings are properly converted"""
        result = bronze_df.select(
            explode("data.data").alias("weather_data")
        ).select(
            to_date(col("weather_data.date")).alias("observation_date")
        ).collect()[0]
        # date: "2024-01-01" should convert to 2024-01-01
        assert str(result.observation_date) == "2024-01-01"
    
    def test_year_extraction_for_partitioning(self, spark, bronze_df):
        """Test year extraction for partitioning"""
        # First explode the data array
        exploded_df = bronze_df.select(explode("data.data").alias("weather_data"))
        
        # Then extract the year
        result = exploded_df.select(
            year(to_timestamp(col("weather_data.date"))).cast("long").alias("year")
        ).collect()
        
        # Both test records are from 2024
        for row in result:
            assert row.year == 2024

class TestSchemaValidation:
    """Test Pandera schema validation"""
    
    def test_schema_validation_success(self, spark, bronze_df):
        """Test that properly transformed data passes validation"""
        # Apply the same transformation as in your main function
        transformed_df = bronze_df.select(
            explode("data.data").alias("weather_data"),
            col("data.metadata").alias("metadata")
        ).select(
            to_date(col("weather_data.date")).alias("observation_date"),
            col("metadata.ingestion_timestamp").cast("timestamp").alias("ingestion_datetime"),
            col("metadata.year").cast("long").alias("year"),
            col("metadata.city").alias("city"),
            col("weather_data.lat").alias("latitude"),
            col("weather_data.lon").alias("longitude"),
            col("weather_data.temperature.afternoon").alias("temperature_celsius"),
            col("weather_data.pressure.afternoon").cast("integer").alias("pressure_hpa"),
            col("weather_data.humidity.afternoon").alias("humidity_percent"),
            col("weather_data.cloud_cover.afternoon").cast("double").alias("cloud_cover_percent"),
            col("weather_data.precipitation.total").alias("precipitation_mm"),
            col("metadata.source").alias("source_system")
        )
        
        # Calculate confidence score
        transformed_df = calculate_confidence_score(transformed_df)
        
        # Convert to pandas for validation
        pandas_df = transformed_df.toPandas()
        
        # Cast integer columns to Int64 for Pandera
        for colname in ["pressure_hpa", "humidity_percent"]:
            if colname in pandas_df.columns:
                pandas_df[colname] = pandas_df[colname].astype('Int64')
        
        # This should pass without errors
        result = validate_silver_data(pandas_df)
        assert result == True
    
    def test_schema_validation_catches_bad_data(self, spark):
        """Test that schema validation catches invalid data"""
        # Create data that violates the schema
        bad_data = [{
            "observation_date": "2024-01-01",
            "ingestion_datetime": "2025-05-08 20:56:55",
            "year": 2024,
            "city": "New York",
            "latitude": 200.0,  # Invalid - should be between -90 and 90
            "longitude": -74.0060,
            "temperature_celsius": 3.37,
            "feels_like_celsius": 3.37,
            "pressure_hpa": 1021,
            "humidity_percent": 72,
            "cloud_cover_percent": 100,
            "precipitation_mm": 1.14,
            "data_confidence_score": 1.0,
            "source_system": "openweathermap"
        }]
        
        pandas_df = pd.DataFrame(bad_data)
        pandas_df['ingestion_datetime'] = pd.to_datetime(pandas_df['ingestion_datetime'])
        
        # This should fail validation
        result = validate_silver_data(pandas_df)
        assert result == False

def test_full_pipeline_integration(spark, bronze_df):
    """Integration test that simulates the full pipeline"""
    # This test runs the complete transformation pipeline
    transformed_df = bronze_df.select(
        explode("data.data").alias("weather_data"),
        col("data.metadata").alias("metadata")
    ).select(
        to_date(col("weather_data.date")).alias("observation_date"),
        col("metadata.ingestion_timestamp").cast("timestamp").alias("ingestion_datetime"),
        col("metadata.year").cast("long").alias("year"),
        col("metadata.city").alias("city"),
        col("weather_data.lat").alias("latitude"),
        col("weather_data.lon").alias("longitude"),
        col("weather_data.temperature.afternoon").alias("temperature_celsius"),
        col("weather_data.pressure.afternoon").cast("integer").alias("pressure_hpa"),
        col("weather_data.humidity.afternoon").alias("humidity_percent"),
        col("weather_data.cloud_cover.afternoon").cast("double").alias("cloud_cover_percent"),
        col("weather_data.precipitation.total").alias("precipitation_mm"),
        col("metadata.source").alias("source_system")
    )
    # Calculate confidence score
    transformed_df = calculate_confidence_score(transformed_df)
    # Validate schema
    pandas_df = transformed_df.toPandas()
    
    # Cast integer columns to Int64 for Pandera
    for colname in ["pressure_hpa", "humidity_percent"]:
        if colname in pandas_df.columns:
            pandas_df[colname] = pandas_df[colname].astype('Int64')
    assert validate_silver_data(pandas_df) == True
    # Check partitioning columns
    partitioning_check = transformed_df.select("city", "year").distinct().collect()
    assert len(partitioning_check) == 1  # New York 2024
    print("✅ Full pipeline integration test passed!")
    print(f"✅ Processed {transformed_df.count()} records")
    print(f"✅ Schema validation successful")
    print(f"✅ Partitioning ready: {len(partitioning_check)} unique city/year combinations")

