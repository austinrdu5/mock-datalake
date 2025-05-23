import pytest
from datetime import datetime, timedelta, timezone
from src.bronze import owm_extract as omw
import os
from dotenv import load_dotenv
import boto3
from botocore.exceptions import ClientError
from unittest.mock import patch, MagicMock
import json
import requests
import time
import pandas as pd
import pandera.pandas as pa
from pandera.errors import SchemaError
import uuid
from calendar import monthrange

# Load environment variables
load_dotenv()

# Test-specific prefix for S3 objects
TEST_PREFIX = f"test_{uuid.uuid4()}/"

@pytest.fixture
def sample_city_list():
    return ["London", "New York"]

@pytest.fixture
def sample_months():
    # Use recent months that are guaranteed to have data
    today = datetime.now()
    return [
        (today.year, today.month - 2),  # Two months ago
        (today.year, today.month - 1),  # Last month
        (today.year, today.month),      # Current month
    ]

@pytest.fixture
def aws_config():
    """Fixture to provide AWS configuration from environment variables"""
    return {
        'access_key': os.getenv('TEST_AWS_ACCESS_KEY'),
        'secret_key': os.getenv('TEST_AWS_SECRET_KEY'),
        'bucket': os.getenv('TEST_AWS_S3_BUCKET_NAME'),
        'region': os.getenv('TEST_AWS_DEFAULT_REGION')
    }

@pytest.fixture
def s3_client(aws_config):
    """Fixture to provide S3 client"""
    return boto3.client(
        's3',
        aws_access_key_id=aws_config['access_key'],
        aws_secret_access_key=aws_config['secret_key'],
        region_name=aws_config['region']
    )

@pytest.fixture(autouse=True)
def cleanup_s3(s3_client, aws_config):
    """Fixture to clean up test data from S3 after each test"""
    yield
    try:
        # List all objects with test prefix
        paginator = s3_client.get_paginator('list_objects_v2')
        for page in paginator.paginate(Bucket=aws_config['bucket'], Prefix=TEST_PREFIX):
            if 'Contents' in page:
                # Delete all objects in the test prefix
                objects_to_delete = [{'Key': obj['Key']} for obj in page['Contents']]
                if objects_to_delete:
                    s3_client.delete_objects(
                        Bucket=aws_config['bucket'],
                        Delete={'Objects': objects_to_delete}
                    )
    except Exception as e:
        print(f"Warning: Failed to clean up S3 test data: {str(e)}")

@pytest.fixture
def api_key():
    """Fixture to provide OpenWeatherMap API key"""
    return os.getenv('OPENWEATHERMAP_API_KEY')

@pytest.fixture
def sample_weather_data():
    """Fixture to provide sample weather data that matches Pandera schema"""
    return {
        'lat': 51.5074,
        'lon': -0.1278,
        'date': '2024-03-22',
        'cloud_cover_afternoon': 20,
        'humidity_afternoon': 65,
        'precipitation_total': 0.0,
        'temperature_min': 10.0,
        'temperature_max': 15.0,
        'temperature_afternoon': 14.0,
        'temperature_night': 11.0,
        'temperature_evening': 13.0,
        'temperature_morning': 12.0,
        'pressure_afternoon': 1015,
        'wind_max_speed': 5.5,
        'wind_max_direction': 180,
        'source': 'openweathermap',
        'ingestion_timestamp': datetime.now().isoformat(),
        'city': 'London',
        'timezone': 'Europe/London'
    }

@pytest.fixture
def sample_monthly_data(sample_weather_data):
    """Fixture to provide sample monthly weather data"""
    year, month = datetime.now().year, datetime.now().month
    _, num_days = monthrange(year, month)
    
    # Create a list of daily weather data for the month
    daily_data = []
    for day in range(1, num_days + 1):
        day_data = sample_weather_data.copy()
        # Set timestamp to 9:00 AM on each day
        dt = datetime(year, month, day, 9, 0, tzinfo=timezone.utc)
        day_data['dt'] = int(dt.timestamp())
        daily_data.append(day_data)
    
    return {
        'city': sample_weather_data['city'],
        'year': year,
        'month': month,
        'data': daily_data,
        's3_key': f"bronze/openweathermap/city={sample_weather_data['city'].lower().replace(' ', '_')}/year={year}/{month:02d}.json"
    }

def test_get_lat_lon(api_key):
    """Test that we can get lat/lon for cities"""
    lat, lon = omw.get_lat_lon("London", api_key)
    assert isinstance(lat, float)
    assert isinstance(lon, float)
    assert 51.0 <= lat <= 52.0  # London's latitude
    assert -1.0 <= lon <= 0.0   # London's longitude

def test_fetch_historical_weather(api_key):
    """Test that we can fetch historical weather data"""
    lat, lon = omw.get_lat_lon("London", api_key)
    yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
    
    data = omw.fetch_historical_weather(lat, lon, yesterday, api_key)
    
    assert isinstance(data, dict)
    assert "lat" in data
    assert "lon" in data
    assert "date" in data
    assert "temperature" in data
    assert "humidity" in data
    assert "cloud_cover" in data

def test_extract_historical_weather_for_cities(sample_city_list, sample_months, api_key, sample_weather_data, aws_config):
    """Test the full extraction process with validation"""
    with patch('src.bronze.owm_extract.fetch_historical_weather') as mock_fetch, \
         patch('src.bronze.owm_extract.get_lat_lon') as mock_get_coords:
        
        # Setup mock returns with valid data
        mock_get_coords.return_value = (51.5074, -0.1278)
        mock_fetch.return_value = {
            'lat': 51.5074,
            'lon': -0.1278,
            'date': '2024-03-22',
            'temperature': {
                'min': 10.0,
                'max': 15.0,
                'afternoon': 14.0,
                'night': 11.0,
                'evening': 13.0,
                'morning': 12.0
            },
            'humidity': {'afternoon': 65},
            'cloud_cover': {'afternoon': 20},
            'pressure': {'afternoon': 1015},
            'wind': {
                'max_speed': 5.5,
                'max_direction': 180
            },
            'precipitation': {'total': 0.0},
            'source': 'openweathermap',
            'ingestion_timestamp': datetime.now().isoformat(),
            'city': 'London',
            'timezone': 'Europe/London'
        }
        
        results, failures = omw.extract_historical_weather_for_cities(
            city_list=sample_city_list,
            api_key=api_key,
            months=sample_months,
            aws_config=aws_config
        )
        
        # Verify results
        assert len(results) > 0
        for result in results:
            assert isinstance(result, dict)
            assert 'city' in result
            assert 'year' in result
            assert 'month' in result
            assert 'data' in result
            assert isinstance(result['data'], dict)
            assert 'data' in result['data']
            assert isinstance(result['data']['data'], list)
            assert len(result['data']['data']) > 0
            assert result['city'] in sample_city_list

def test_save_to_s3_real_bucket(aws_config, sample_monthly_data, s3_client):
    """Test saving data to real S3 bucket"""
    # Save to S3
    success = omw.save_to_s3([sample_monthly_data], aws_config)
    assert success is True
    
    # Read the saved data
    response = s3_client.get_object(Bucket=aws_config['bucket'], Key=sample_monthly_data['s3_key'])
    saved_data = json.loads(response['Body'].read().decode('utf-8'))
    
    # Verify the saved data matches the original
    assert saved_data == sample_monthly_data

def test_get_existing_s3_paths_real_bucket(aws_config, sample_monthly_data, s3_client):
    """Test getting existing S3 paths from real bucket"""
    # First, save some test data
    test_data = sample_monthly_data.copy()
    test_data['city'] = f'TestCity_{uuid.uuid4().hex[:8]}'  # Unique city name
    test_data['s3_key'] = f"bronze/openweathermap/city={test_data['city'].lower().replace(' ', '_')}/year={test_data['year']}/{test_data['month']:02d}.json"
    
    # Save to S3
    omw.save_to_s3([test_data], aws_config)
    
    # Add a small delay to ensure S3 operation completes
    time.sleep(2)
    
    # Get existing paths
    cities = [test_data['city']]
    months = [(test_data['year'], test_data['month'])]
    
    existing_paths = omw.get_existing_s3_paths(cities, months, aws_config)
    
    # Verify the path exists
    assert test_data['s3_key'] in existing_paths, f"Expected key {test_data['s3_key']} not found in existing paths"

# Validation Tests
def test_pandera_validation_success(sample_weather_data):
    """Test that valid data passes JSON schema validation"""
    assert omw.validate_bronze_data(sample_weather_data) is True

def test_pandera_validation_failure_invalid_lat(sample_weather_data):
    """Test that invalid latitude fails validation"""
    invalid_data = sample_weather_data.copy()
    invalid_data['lat'] = 200.0  # Invalid latitude
    assert omw.validate_bronze_data(invalid_data) is False

def test_pandera_validation_failure_invalid_lon(sample_weather_data):
    """Test that invalid longitude fails validation"""
    invalid_data = sample_weather_data.copy()
    invalid_data['lon'] = 200.0  # Invalid longitude
    assert omw.validate_bronze_data(invalid_data) is False

def test_pandera_validation_failure_invalid_humidity(sample_weather_data):
    """Test that invalid humidity fails validation"""
    invalid_data = sample_weather_data.copy()
    invalid_data['humidity_afternoon'] = 150  # Invalid humidity (>100)
    assert omw.validate_bronze_data(invalid_data) is False

def test_pandera_validation_failure_invalid_wind_deg(sample_weather_data):
    """Test that invalid wind direction fails validation"""
    invalid_data = sample_weather_data.copy()
    invalid_data['wind_max_direction'] = 400  # Invalid wind direction (>360)
    assert omw.validate_bronze_data(invalid_data) is False

def test_pandera_validation_failure_missing_required(sample_weather_data):
    """Test that missing required fields fails validation"""
    invalid_data = sample_weather_data.copy()
    del invalid_data['lat']  # Remove required field
    assert omw.validate_bronze_data(invalid_data) is False

def test_pandera_validation_int_type(sample_weather_data):
    """Test that integer fields as Python int are accepted"""
    data = sample_weather_data.copy()
    # Use Python int instead of numpy int32
    data['cloud_cover_afternoon'] = int(data['cloud_cover_afternoon'])
    data['humidity_afternoon'] = int(data['humidity_afternoon'])
    data['pressure_afternoon'] = int(data['pressure_afternoon'])
    data['wind_max_direction'] = int(data['wind_max_direction'])
    assert omw.validate_bronze_data(data) is True
