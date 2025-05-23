import pytest
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
import sys
import os

# Add src to path so we can import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from silver.owm_convert import (
    calculate_confidence_score, 
    OpenWeatherMapSilverSchema,
    validate_silver_data
)

@pytest.fixture(scope="session")
def spark():
    """Create a Spark session for testing"""
    spark = SparkSession.builder \
        .appName("OWMConvertTest") \
        .config("spark.sql.execution.arrow.pyspark.enabled", "false") \
        .getOrCreate()
    
    spark.sparkContext.setLogLevel("ERROR")  # Reduce noise
    yield spark
    spark.stop()

@pytest.fixture(scope="session")
def bronze_schema():
    """Concrete schema for bronze data, matching the structure in owm_convert.py"""
    return StructType([
        StructField("lat", DoubleType(), True),
        StructField("lon", DoubleType(), True),
        StructField("dt", LongType(), True),
        StructField("temp", DoubleType(), True),
        StructField("feels_like", DoubleType(), True),
        StructField("pressure", IntegerType(), True),
        StructField("humidity", IntegerType(), True),
        StructField("dew_point", DoubleType(), True),
        StructField("uvi", DoubleType(), True),
        StructField("clouds", IntegerType(), True),
        StructField("visibility", IntegerType(), True),
        StructField("wind_speed", DoubleType(), True),
        StructField("wind_deg", IntegerType(), True),
        StructField("wind_gust", DoubleType(), True),
        StructField("weather", ArrayType(MapType(StringType(), StringType())), True),
        StructField("_metadata", StructType([
            StructField("source", StringType(), True),
            StructField("ingestion_timestamp", StringType(), True),
            StructField("city", StringType(), True),
            StructField("timezone", StringType(), True),
            StructField("timezone_offset", IntegerType(), True),
        ]), True),
    ])

@pytest.fixture
def sample_bronze_data():
    """Sample data that matches your S3 bronze structure exactly"""
    return [
        {
            "lat": 52.5244,
            "lon": 13.4105,
            "dt": 1704092400,  # 2024-01-01 07:00:00 UTC
            "temp": 5.71,
            "feels_like": 1.92,
            "pressure": 1005,
            "humidity": 88,
            "dew_point": 3.88,
            "uvi": None,
            "clouds": 40,
            "visibility": 10000,
            "wind_speed": 5.7,
            "wind_deg": 210,
            "wind_gust": 0.0,
            "weather": [{"id": "500", "main": "Rain", "description": "light rain", "icon": "10n"}],
            "_metadata": {
                "source": "openweathermap",
                "ingestion_timestamp": "2025-05-08T20:56:55.884426",
                "city": "Berlin",
                "timezone": "Europe/Berlin",
                "timezone_offset": 3600
            }
        },
        {
            # Test case with missing fields (lower confidence score)
            "lat": 40.7128,
            "lon": -74.0060,
            "dt": 1704096000,  # 2024-01-01 08:00:00 UTC
            "temp": 2.5,
            "feels_like": None,  # Missing
            "pressure": None,    # Missing
            "humidity": 65,
            "visibility": None,  # Missing
            "wind_speed": 3.2,
            "_metadata": {
                "source": "openweathermap",
                "ingestion_timestamp": "2025-05-08T21:00:00.000000",
                "city": "New York",
            }
        }
    ]

class TestConfidenceScoreCalculation:
    """Test the confidence score calculation logic"""
    
    def test_perfect_confidence_score(self, spark, sample_bronze_data, bronze_schema):
        """Test confidence score with all fields present"""
        df = spark.createDataFrame([sample_bronze_data[0]], schema=bronze_schema)
        result = calculate_confidence_score(df)
        confidence = result.select("data_confidence_score").collect()[0][0]
        # All fields present: temp(0.3) + pressure(0.2) + humidity(0.2) + feels_like(0.1) + wind_speed(0.1) + visibility(0.1) = 1.0
        assert confidence == 1.0
    
    def test_partial_confidence_score(self, spark, sample_bronze_data, bronze_schema):
        """Test confidence score with missing fields"""
        df = spark.createDataFrame([sample_bronze_data[1]], schema=bronze_schema)
        result = calculate_confidence_score(df)
        confidence = result.select("data_confidence_score").collect()[0][0]
        # Present: temp(0.3) + humidity(0.2) + wind_speed(0.1) = 0.6
        # Missing: pressure, feels_like, visibility
        assert confidence == 0.6

class TestDataTransformation:
    """Test the complete data transformation pipeline"""
    
    def test_bronze_to_silver_transformation(self, spark, sample_bronze_data, bronze_schema):
        """Test the complete transformation matches expected schema"""
        # Create bronze DataFrame
        df = spark.createDataFrame(sample_bronze_data, schema=bronze_schema)
        
        # Apply the same transformation as in your main function
        transformed_df = df.select(
            # Convert Unix timestamp to proper datetime
            timestamp_seconds(col("dt")).alias("observation_datetime"),
            col("_metadata.ingestion_timestamp").cast("timestamp").alias("ingestion_datetime"),
            year(timestamp_seconds(col("dt"))).cast("long").alias("year"),  # Cast to long to avoid int32 issue
            
            # Location fields
            col("_metadata.city").alias("city"),
            col("lat").alias("latitude"),
            col("lon").alias("longitude"),
            
            # Weather measurements
            col("temp").alias("temperature_celsius"),
            col("feels_like").alias("feels_like_celsius"),
            col("pressure").cast("integer").alias("pressure_hpa"),
            col("humidity").alias("humidity_percent"),
            
            # Keep original fields for confidence calculation
            col("temp"), col("pressure"), col("humidity"), 
            col("feels_like"), col("wind_speed"), col("visibility"),
            
            # Lineage
            col("_metadata.source").alias("source_system")
        )
        
        # Calculate confidence score
        transformed_df = calculate_confidence_score(transformed_df)
        
        # Drop helper columns
        final_df = transformed_df.drop("temp", "pressure", "humidity", "feels_like", "wind_speed", "visibility")
        
        # Verify we have the expected number of records
        assert final_df.count() == 2
        
        # Check that required columns exist
        expected_columns = [
            "observation_datetime", "ingestion_datetime", "year", "city", 
            "latitude", "longitude", "temperature_celsius", "feels_like_celsius",
            "pressure_hpa", "humidity_percent", "data_confidence_score", "source_system"
        ]
        
        actual_columns = final_df.columns
        for col_name in expected_columns:
            assert col_name in actual_columns, f"Missing column: {col_name}"
        
        return final_df
    
    def test_datetime_conversion(self, spark, sample_bronze_data, bronze_schema):
        """Test that Unix timestamps are properly converted"""
        df = spark.createDataFrame([sample_bronze_data[0]], schema=bronze_schema)
        result = df.select(
            timestamp_seconds(col("dt")).alias("observation_datetime")
        ).collect()[0]
        # dt: 1704092400 should convert to 2024-01-01 02:00:00 in local time (UTC-5)
        assert str(result.observation_datetime).startswith("2024-01-01 02:00:00")
    
    def test_year_extraction_for_partitioning(self, spark, sample_bronze_data, bronze_schema):
        """Test year extraction for partitioning"""
        df = spark.createDataFrame(sample_bronze_data, schema=bronze_schema)
        
        result = df.select(
            year(timestamp_seconds(col("dt"))).cast("long").alias("year")
        ).collect()
        
        # Both test records are from 2024
        for row in result:
            assert row.year == 2024

class TestSchemaValidation:
    """Test Pandera schema validation"""
    
    def test_schema_validation_success(self, spark, sample_bronze_data, bronze_schema):
        """Test that properly transformed data passes validation"""
        # Run the full transformation
        test_transform = TestDataTransformation()
        final_df = test_transform.test_bronze_to_silver_transformation(spark, sample_bronze_data, bronze_schema)
        # Convert to pandas for validation
        pandas_df = final_df.toPandas()
        # Cast pressure_hpa and humidity_percent to Int64 for Pandera
        for colname in ["pressure_hpa", "humidity_percent"]:
            if colname in pandas_df.columns:
                pandas_df[colname] = pandas_df[colname].astype('Int64')
        # This should pass without errors
        result = validate_silver_data(pandas_df)
        assert result == True
    
    def test_schema_validation_catches_bad_data(self, spark, bronze_schema):
        """Test that schema validation catches invalid data"""
        # Create data that violates the schema
        bad_data = [{
            "observation_datetime": "2024-01-01 07:00:00",
            "ingestion_datetime": "2025-05-08 20:56:55",
            "year": 2024,
            "city": "Berlin",
            "latitude": 200.0,  # Invalid - should be between -90 and 90
            "longitude": 13.4105,
            "temperature_celsius": 5.71,
            "feels_like_celsius": 1.92,
            "pressure_hpa": 1005,
            "humidity_percent": 88,
            "data_confidence_score": 1.0,
            "source_system": "openweathermap"
        }]
        
        pandas_df = pd.DataFrame(bad_data)
        pandas_df['observation_datetime'] = pd.to_datetime(pandas_df['observation_datetime'])
        pandas_df['ingestion_datetime'] = pd.to_datetime(pandas_df['ingestion_datetime'])
        
        # This should fail validation
        result = validate_silver_data(pandas_df)
        assert result == False

class TestEdgeCases:
    """Test edge cases and error conditions"""
    
    def test_handles_null_metadata(self, spark, bronze_schema):
        """Test handling of records with missing metadata"""
        data_with_null_metadata = [{
            "lat": 52.5244,
            "lon": 13.4105,
            "dt": 1704092400,
            "temp": 5.71,
            "_metadata": {
                "source": "openweathermap",
                "city": None,  # Null city
                "ingestion_timestamp": "2025-05-08T20:56:55.884426"
            }
        }]
        
        df = spark.createDataFrame(data_with_null_metadata, schema=bronze_schema)
        
        # Should handle null city gracefully
        result = df.select(col("_metadata.city").alias("city")).collect()[0]
        assert result.city is None
    
    def test_handles_missing_weather_fields(self, spark, bronze_schema):
        """Test confidence calculation with completely missing weather fields"""
        minimal_data = [{
            "lat": 52.5244,
            "lon": 13.4105,
            "dt": 1704092400,
            # No weather measurements at all
            "_metadata": {
                "source": "openweathermap",
                "city": "Berlin",
                "ingestion_timestamp": "2025-05-08T20:56:55.884426"
            }
        }]
        
        df = spark.createDataFrame(minimal_data, schema=bronze_schema)
        result = calculate_confidence_score(df)
        
        confidence = result.select("data_confidence_score").collect()[0][0]
        
        # No weather fields present, confidence should be 0.0
        assert confidence == 0.0

def test_full_pipeline_integration(spark, sample_bronze_data, bronze_schema):
    """Integration test that simulates the full pipeline"""
    # This test runs the complete transformation pipeline
    df = spark.createDataFrame(sample_bronze_data, schema=bronze_schema)
    # Apply transformations (same as main function)
    transformed_df = df.select(
        timestamp_seconds(col("dt")).alias("observation_datetime"),
        col("_metadata.ingestion_timestamp").cast("timestamp").alias("ingestion_datetime"),
        year(timestamp_seconds(col("dt"))).cast("long").alias("year"),
        col("_metadata.city").alias("city"),
        col("lat").alias("latitude"),
        col("lon").alias("longitude"),
        col("temp").alias("temperature_celsius"),
        col("feels_like").alias("feels_like_celsius"),
        col("pressure").cast("integer").alias("pressure_hpa"),
        col("humidity").alias("humidity_percent"),
        col("temp"), col("pressure"), col("humidity"), 
        col("feels_like"), col("wind_speed"), col("visibility"),
        col("_metadata.source").alias("source_system")
    )
    # Calculate confidence and clean up
    transformed_df = calculate_confidence_score(transformed_df)
    final_df = transformed_df.drop("temp", "pressure", "humidity", "feels_like", "wind_speed", "visibility")
    # Validate schema
    pandas_df = final_df.toPandas()
    for colname in ["pressure_hpa", "humidity_percent"]:
        if colname in pandas_df.columns:
            pandas_df[colname] = pandas_df[colname].astype('Int64')
    assert validate_silver_data(pandas_df) == True
    # Check partitioning columns
    partitioning_check = final_df.select("city", "year").distinct().collect()
    assert len(partitioning_check) == 2  # Berlin 2024, New York 2024
    print("✅ Full pipeline integration test passed!")
    print(f"✅ Processed {final_df.count()} records")
    print(f"✅ Schema validation successful")
    print(f"✅ Partitioning ready: {len(partitioning_check)} unique city/year combinations")