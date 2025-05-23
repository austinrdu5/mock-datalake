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

# Load environment variables
load_dotenv()

# Test-specific prefix for S3 objects
TEST_PREFIX = f"test_{uuid.uuid4()}/"

@pytest.fixture
def sample_city_list():
    return ["London", "New York"]

@pytest.fixture
def sample_dates():
    # Use recent dates that are guaranteed to have data
    today = datetime.now()
    return [
        today - timedelta(days=3),  # Oldest first
        today - timedelta(days=2),
        today - timedelta(days=1),
    ]

@pytest.fixture
def aws_config():
    """Fixture to provide AWS configuration from environment variables"""
    return {
        'access_key': os.getenv('TEST_AWS_ACCESS_KEY'),
        'secret_key': os.getenv('TEST_AWS_SECRET_KEY'),
        'bucket': os.getenv('TEST_AWS_S3_BUCKET_NAME'),
        'region': os.getenv('TEST_AWS_DEFAULT_REGION', 'us-east-1')
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
        'dt': int(datetime.now().timestamp()),
        'temp': 20.5,
        'feels_like': 19.8,
        'pressure': 1015,  # int32
        'humidity': 65,    # int32
        'dew_point': 15.2,
        'uvi': 5.2,
        'clouds': 20,      # int32
        'visibility': 10000,  # int32
        'wind_speed': 5.5,
        'wind_deg': 180,   # int32
        'wind_gust': 7.2,
        'weather': [{
            'id': 800,
            'main': 'Clear',
            'description': 'clear sky',
            'icon': '01d'
        }],
        # Flattened metadata fields
        'source': 'openweathermap',
        'ingestion_timestamp': datetime.now().isoformat(),
        'city': 'London',
        'timezone': 'Europe/London',
        'timezone_offset': 0
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
    yesterday = int((datetime.now() - timedelta(days=1)).timestamp())
    
    data = omw.fetch_historical_weather(lat, lon, yesterday, api_key)
    
    assert isinstance(data, dict)
    assert "lat" in data
    assert "lon" in data
    assert "data" in data
    assert isinstance(data["data"], list)
    assert len(data["data"]) > 0
    assert "dt" in data["data"][0]
    assert "weather" in data["data"][0]

def test_extract_historical_weather_for_cities(sample_city_list, sample_dates, api_key, sample_weather_data, aws_config):
    """Test the full extraction process with validation"""
    with patch('src.bronze.owm_extract.fetch_with_retry') as mock_fetch, \
         patch('src.bronze.owm_extract.get_lat_lon') as mock_get_coords:
        
        # Setup mock returns with valid data
        mock_get_coords.return_value = (51.5074, -0.1278)
        mock_fetch.return_value = (sample_weather_data, None)
        
        results, failures = omw.extract_historical_weather_for_cities(
            city_list=sample_city_list,
            api_key=api_key,
            dates=sample_dates,
            aws_config=aws_config
        )
        
        # Verify results
        assert len(results) > 0
        for result in results:
            assert omw.validate_bronze_data(result) is True
            assert result['city'] in sample_city_list


def test_save_to_s3_real_bucket(aws_config, sample_weather_data, s3_client):
    """Test saving data to real S3 bucket"""
    # Ensure data is valid before saving
    assert omw.validate_bronze_data(sample_weather_data) is True
    
    # Save to S3
    success = omw.save_to_s3([sample_weather_data], aws_config)
    assert success is True
    
    # Construct the expected S3 key
    city = sample_weather_data['city'].lower().replace(' ', '_')
    dt = datetime.fromtimestamp(sample_weather_data['dt'], tz=timezone.utc)
    s3_key = f"bronze/openweathermap/city={city}/{dt.strftime('%Y-%m-%d_%H-%M')}UTC.json"
    
    # Read the saved data
    response = s3_client.get_object(Bucket=aws_config['bucket'], Key=s3_key)
    saved_data = json.loads(response['Body'].read().decode('utf-8'))
    
    # Verify the saved data matches the original
    assert saved_data == sample_weather_data

def test_get_existing_s3_paths_real_bucket(aws_config, sample_weather_data, s3_client):
    """Test getting existing S3 paths from real bucket"""
    # First, save some test data
    test_data = sample_weather_data.copy()
    test_data['city'] = f'TestCity_{uuid.uuid4().hex[:8]}'  # Unique city name
    
    # Save to S3
    omw.save_to_s3([test_data], aws_config)
    
    # Add a small delay to ensure S3 operation completes
    time.sleep(2)
    
    # Get existing paths
    cities = [test_data['city']]
    dates = [datetime.fromtimestamp(test_data['dt'])]
    
    existing_paths = omw.get_existing_s3_paths(cities, dates, aws_config)
    print(f"DEBUG: existing_paths = {existing_paths}")  # Debug print
    
    # Verify the path exists
    city = test_data['city'].lower().replace(' ', '_')
    dt = datetime.fromtimestamp(test_data['dt'], tz=timezone.utc)
    expected_key = f"bronze/openweathermap/city={city}/{dt.strftime('%Y-%m-%d_%H-%M')}UTC.json"
    print(f"DEBUG: expected_key = {expected_key}")  # Debug print
    
    assert expected_key in existing_paths, f"Expected key {expected_key} not found in existing paths"

def test_timezone_handling(api_key):
    """Test that API calls are made with correct UTC timestamps based on city timezones"""
    # Mock the API responses
    with patch('requests.get') as mock_get, \
         patch('src.bronze.owm_extract.get_existing_s3_paths') as mock_get_paths, \
         patch('time.sleep') as mock_sleep:  # Mock sleep to speed up test
        
        # Mock responses for timezone info
        def mock_get_side_effect(*args, **kwargs):
            mock_response = MagicMock()
            mock_response.raise_for_status.return_value = None
            
            url = args[0] if args else kwargs.get('url', '')
            params = kwargs.get('params', {})
            
            if 'data/2.5/weather' in url:
                city = params.get('q', '')
                if city == 'London':
                    mock_response.json.return_value = {
                        'coord': {'lat': 51.5074, 'lon': -0.1278},
                        'timezone': 0,  # UTC+0
                        'name': 'London'
                    }
                elif city == 'New York':
                    mock_response.json.return_value = {
                        'coord': {'lat': 40.7128, 'lon': -74.0060},
                        'timezone': -18000,  # UTC-5
                        'name': 'New York'
                    }
                elif city == 'Tokyo':
                    mock_response.json.return_value = {
                        'coord': {'lat': 35.6762, 'lon': 139.6503},
                        'timezone': 32400,  # UTC+9
                        'name': 'Tokyo'
                    }
            elif 'data/3.0/onecall/timemachine' in url:
                # Return a complete weather data response
                mock_response.json.return_value = {
                    'lat': params.get('lat'),
                    'lon': params.get('lon'),
                    'timezone': 'UTC',
                    'timezone_offset': 0,
                    'data': [{
                        'dt': params.get('dt'),
                        'temp': 20.0,  # Use float
                        'feels_like': 18.0,  # Use float
                        'pressure': 1015,
                        'humidity': 65,
                        'dew_point': 12.0,  # Use float
                        'uvi': 5.0,  # Use float
                        'clouds': 20,
                        'visibility': 10000,
                        'wind_speed': 5.0,  # Use float
                        'wind_deg': 180,
                        'wind_gust': 7.2,  # Add wind_gust as float
                        'weather': [{
                            'id': 800,
                            'main': 'Clear',
                            'description': 'clear sky',
                            'icon': '01d'
                        }]
                    }]
                }
            return mock_response
        
        mock_get.side_effect = mock_get_side_effect
        mock_get_paths.return_value = set()
        
        # Test cities with different timezones
        cities = ["London", "New York", "Tokyo"]
        test_date = datetime(2024, 3, 15)
        
        results, failures = omw.extract_historical_weather_for_cities(
            city_list=cities,
            api_key=api_key,
            dates=[test_date]
        )
        
        # Verify results
        assert len(results) == len(cities)
        assert len(failures) == 0
        
        # Verify timestamps correspond to 9:00 AM local time
        for result in results:
            city = result['city']
            dt = datetime.fromtimestamp(result['dt'], tz=timezone.utc)
            
            if city == "London":
                local_time = dt.astimezone(timezone.utc)
            elif city == "New York":
                local_time = dt.astimezone(timezone(timedelta(hours=-5)))
            elif city == "Tokyo":
                local_time = dt.astimezone(timezone(timedelta(hours=9)))
            
            assert local_time.hour == 9
            assert local_time.minute == 0

# Validation Tests
def test_pandera_validation_success(sample_weather_data):
    """Test that valid data passes Pandera validation"""
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
    invalid_data['humidity'] = 150  # Invalid humidity (>100)
    assert omw.validate_bronze_data(invalid_data) is False

def test_pandera_validation_failure_invalid_wind_deg(sample_weather_data):
    """Test that invalid wind direction fails validation"""
    invalid_data = sample_weather_data.copy()
    invalid_data['wind_deg'] = 400  # Invalid wind direction (>360)
    assert omw.validate_bronze_data(invalid_data) is False

def test_pandera_validation_failure_missing_required(sample_weather_data):
    """Test that missing required fields fails validation"""
    invalid_data = sample_weather_data.copy()
    del invalid_data['dt']  # Remove required field
    assert omw.validate_bronze_data(invalid_data) is False

def test_pandera_validation_int32_conversion(sample_weather_data):
    """Test that integer fields are properly converted to int32"""
    data = sample_weather_data.copy()
    # Convert to DataFrame and back to test int32 conversion
    df = pd.DataFrame([data])
    df = df.astype({
        'dt': 'int32',
        'pressure': 'int32',
        'humidity': 'int32',
        'clouds': 'int32',
        'visibility': 'int32',
        'wind_deg': 'int32',
        'timezone_offset': 'int32'
    })
    converted_data = df.iloc[0].to_dict()
    assert omw.validate_bronze_data(converted_data) is True
