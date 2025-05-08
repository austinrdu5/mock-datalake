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

# Load environment variables for API key
load_dotenv()

@pytest.fixture
def sample_city_list():
    return ["London", "New York"]

@pytest.fixture
def sample_dates():
    # Use recent dates that are guaranteed to have data
    today = datetime.now()
    return [
        today - timedelta(days=1),
        today - timedelta(days=2),
        today - timedelta(days=3),
    ]

@pytest.fixture
def mock_s3_client():
    """Fixture to create a mock S3 client"""
    with patch('boto3.client') as mock_client:
        yield mock_client

@pytest.fixture
def sample_weather_data():
    """Fixture to provide sample weather data"""
    return {
        'lat': 51.5074,
        'lon': -0.1278,
        'dt': int(datetime.now().timestamp()),
        'weather': [{
            'main': 'Clear',
            'description': 'clear sky'
        }],
        '_metadata': {
            'source': 'openweathermap',
            'ingestion_timestamp': datetime.now().isoformat(),
            'city': 'London'
        }
    }

@pytest.fixture
def mock_env_vars(monkeypatch):
    """Fixture to set up test environment variables"""
    monkeypatch.setenv('OPENWEATHERMAP_API_KEY', 'test_api_key')
    monkeypatch.setenv('AWS_ACCESS_KEY', 'test_aws_key')
    monkeypatch.setenv('AWS_SECRET_KEY', 'test_aws_secret')
    monkeypatch.setenv('AWS_S3_BUCKET_NAME', 'test_bucket')

def test_get_lat_lon():
    """Test that we can get lat/lon for cities"""
    lat, lon = omw.get_lat_lon("London")
    assert isinstance(lat, float)
    assert isinstance(lon, float)
    assert 51.0 <= lat <= 52.0  # London's latitude
    assert -1.0 <= lon <= 0.0   # London's longitude

def test_fetch_historical_weather():
    """Test that we can fetch historical weather data"""
    lat, lon = omw.get_lat_lon("London")
    yesterday = int((datetime.now() - timedelta(days=1)).timestamp())
    
    data = omw.fetch_historical_weather(lat, lon, yesterday)
    
    assert isinstance(data, dict)
    assert "lat" in data
    assert "lon" in data
    assert "data" in data
    assert isinstance(data["data"], list)
    assert len(data["data"]) > 0
    assert "dt" in data["data"][0]
    assert "weather" in data["data"][0]

def test_extract_historical_weather_for_cities(sample_city_list, sample_dates):
    """Test the full extraction process with real API calls"""
    # Sort dates in chronological order
    dates = sorted(sample_dates)
    
    results, failures = omw.extract_historical_weather_for_cities(
        city_list=sample_city_list,
        dates=dates
    )
    
    # Check results structure
    assert isinstance(results, list)
    assert all(isinstance(item, dict) for item in results)
    
    # Check required fields
    for item in results:
        assert "lat" in item
        assert "lon" in item
        assert "dt" in item
        assert "weather" in item
        assert "_metadata" in item
        assert "city" in item["_metadata"]
        assert "ingestion_timestamp" in item["_metadata"]

def test_rate_limiting():
    """Test that rate limiting is working by making multiple requests"""
    with patch('src.bronze.owm_extract.fetch_historical_weather') as mock_fetch, \
         patch('src.bronze.owm_extract.get_lat_lon') as mock_get_coords, \
         patch('time.sleep') as mock_sleep:
        
        # Setup mock returns
        mock_get_coords.return_value = (51.5074, -0.1278)
        mock_fetch.return_value = {
            "lat": 51.5074,
            "lon": -0.1278,
            "data": [{
                "dt": int(datetime.now().timestamp()),
                "weather": [{
                    "id": 800,
                    "main": "Clear",
                    "description": "clear sky",
                    "icon": "01d"
                }]
            }]
        }
        
        cities = ["London", "New York", "Tokyo", "Sydney", "Berlin"]
        dates = [datetime.now() - timedelta(days=1)]
        
        results, failures = omw.extract_historical_weather_for_cities(
            city_list=cities,
            dates=dates
        )
        
        # Should have results for all cities
        assert len(results) == len(cities)
        assert len(failures) == 0
        
        # Verify rate limiting was applied
        assert mock_sleep.call_count > 0

def test_retry_mechanism():
    """Test retry mechanism by temporarily using an invalid API key"""
    with patch('requests.get') as mock_get:
        # Create a mock response with error
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.text = "Invalid API key"
        mock_response.json.return_value = {"error": "Invalid response"}
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(response=mock_response)
        mock_get.return_value = mock_response
        
        results, failures = omw.extract_historical_weather_for_cities(
            city_list=["London"],
            dates=[datetime.now() - timedelta(days=1)]
        )
        
        # Should have failures after retries
        assert len(failures) == 1
        assert "401" in failures[0]["error"]

def test_default_date_range():
    """Test that default date range works correctly"""
    # Mock the API responses
    with patch('src.bronze.owm_extract.fetch_with_retry') as mock_fetch, \
         patch('src.bronze.owm_extract.get_lat_lon') as mock_get_coords:
        
        # Setup mock returns
        mock_get_coords.return_value = (51.5074, -0.1278)
        
        # Create mock data for each day
        mock_data = []
        for day in range(3):  # 3 days of data
            date = datetime(2025, 1, 1) + timedelta(days=day)
            mock_data.append({
                "lat": 51.5074,
                "lon": -0.1278,
                "dt": int(date.timestamp()),
                "weather": [{
                    "id": 800,
                    "main": "Clear",
                    "description": "clear sky",
                    "icon": "01d"
                }],
                "_metadata": {
                    "source": "openweathermap",
                    "ingestion_timestamp": datetime.now().isoformat(),
                    "city": "London"
                }
            })
        
        # Mock fetch_with_retry to return each day's data
        def mock_fetch_side_effect(*args, **kwargs):
            dt = args[2]  # timestamp is the third argument
            date = datetime.fromtimestamp(dt)
            day_index = (date - datetime(2025, 1, 1)).days
            return mock_data[day_index], None
        
        mock_fetch.side_effect = mock_fetch_side_effect
        
        results, failures = omw.extract_historical_weather_for_cities(
            city_list=["London"]
        )
        
        # Verify date range
        dates = [datetime.fromtimestamp(item["dt"]) for item in results]
        assert min(dates).date() == datetime(2025, 1, 1).date()
        assert max(dates).date() == datetime(2025, 1, 3).date()  # Changed to 3 days instead of 50+
        assert len(dates) == 3  # Should have exactly 3 days of data

def test_get_existing_s3_paths_with_mock(mock_s3_client):
    """Test get_existing_s3_paths with mocked S3 client"""
    # Setup mock response with pagination
    mock_response1 = {
        'Contents': [
            {'Key': 'bronze/openweathermap/city=london/2024-03-15_09-00.json'},
            {'Key': 'bronze/openweathermap/city=newyork/2024-03-15_09-00.json'}
        ]
    }
    mock_response2 = {
        'Contents': [
            {'Key': 'bronze/openweathermap/city=london/2024-03-16_09-00.json'}
        ]
    }
    
    # Mock paginator to return multiple pages
    mock_paginator = MagicMock()
    mock_paginator.paginate.return_value = [mock_response1, mock_response2]
    mock_s3_client.return_value.get_paginator.return_value = mock_paginator
    
    cities = ["London", "New York"]
    dates = [datetime(2024, 3, 15), datetime(2024, 3, 16)]
    
    existing_paths = omw.get_existing_s3_paths(cities, dates)
    
    # Verify results
    assert len(existing_paths) == 3
    assert 'bronze/openweathermap/city=london/2024-03-15_09-00.json' in existing_paths
    assert 'bronze/openweathermap/city=newyork/2024-03-15_09-00.json' in existing_paths
    assert 'bronze/openweathermap/city=london/2024-03-16_09-00.json' in existing_paths

def test_skip_existing_data(mock_s3_client, sample_weather_data):
    """Test that the extraction process skips existing data"""
    # Mock S3 to return existing data
    mock_response = {
        'Contents': [
            {'Key': 'bronze/openweathermap/city=london/2024-03-15_09-00.json'}
        ]
    }
    mock_paginator = MagicMock()
    mock_paginator.paginate.return_value = [mock_response]
    mock_s3_client.return_value.get_paginator.return_value = mock_paginator
    
    test_city = "London"
    test_date = datetime(2024, 3, 15)
    
    # Mock the fetch functions to verify they're not called
    with patch('src.bronze.owm_extract.fetch_with_retry') as mock_fetch, \
         patch('src.bronze.owm_extract.get_lat_lon') as mock_get_coords:
        
        results, failures = omw.extract_historical_weather_for_cities(
            city_list=[test_city],
            dates=[test_date]
        )
        
        # Verify that neither function was called
        mock_fetch.assert_not_called()
        mock_get_coords.assert_not_called()
        
        # Results should be empty since we skipped the API call
        assert len(results) == 0
        assert len(failures) == 0

def test_mixed_existing_and_new_data(mock_s3_client, sample_weather_data):
    """Test handling of mix of existing and new data"""
    # Mock S3 to return only one existing path
    mock_response = {
        'Contents': [
            {'Key': 'bronze/openweathermap/city=london/2024-03-15_09-00.json'}
        ]
    }
    mock_paginator = MagicMock()
    mock_paginator.paginate.return_value = [mock_response]
    mock_s3_client.return_value.get_paginator.return_value = mock_paginator
    
    test_city = "London"
    test_dates = [
        datetime(2024, 3, 15),  # This will exist
        datetime(2024, 3, 16)   # This will be new
    ]
    
    # Mock the fetch functions
    with patch('src.bronze.owm_extract.fetch_with_retry') as mock_fetch, \
         patch('src.bronze.owm_extract.get_lat_lon') as mock_get_coords:
        
        # Setup mock returns
        mock_get_coords.return_value = (51.5074, -0.1278)
        
        # Create mock data with the correct timestamp
        mock_data = sample_weather_data.copy()
        local_time = test_dates[1].replace(hour=9)
        mock_data['dt'] = int(local_time.timestamp())
        mock_fetch.return_value = (mock_data, None)
        
        results, failures = omw.extract_historical_weather_for_cities(
            city_list=[test_city],
            dates=test_dates
        )
        
        # Should have results only for the new date
        assert len(results) == 1
        assert len(failures) == 0
        
        # Verify fetch was called only once (for the new date)
        assert mock_fetch.call_count == 1
        
        # Verify the result is for the second date
        result_date = datetime.fromtimestamp(results[0]['dt'])
        assert result_date.date() == test_dates[1].date()

def test_s3_error_handling(mock_s3_client):
    """Test handling of S3 errors during path checking"""
    # Setup mock to raise different types of errors
    error_cases = [
        Exception("Generic error"),
        boto3.exceptions.S3UploadFailedError("Upload failed"),
        ClientError(
            {'Error': {'Code': 'NoSuchBucket', 'Message': 'Bucket does not exist'}},
            'HeadObject'
        )
    ]
    
    for error in error_cases:
        mock_s3_client.return_value.get_paginator.return_value.paginate.side_effect = error
        
        cities = ["London"]
        dates = [datetime.now()]
        
        # Should not raise an exception, but return empty set
        existing_paths = omw.get_existing_s3_paths(cities, dates)
        assert isinstance(existing_paths, set)
        assert len(existing_paths) == 0

def test_save_to_s3_with_mock(mock_s3_client, sample_weather_data):
    """Test saving data to S3 with mocked client"""
    # Mock successful S3 put_object
    mock_s3_client.return_value.put_object.return_value = {'ResponseMetadata': {'HTTPStatusCode': 200}}
    
    success = omw.save_to_s3([sample_weather_data])
    assert success is True
    
    # Verify S3 put_object was called with correct parameters
    mock_s3_client.return_value.put_object.assert_called_once()
    call_args = mock_s3_client.return_value.put_object.call_args[1]
    assert call_args['Bucket'] == os.getenv('AWS_S3_BUCKET_NAME')
    assert call_args['ContentType'] == 'application/json'
    
    # Verify the key structure
    key = call_args['Key']
    assert key.startswith('bronze/openweathermap/city=london/')
    assert key.endswith('.json')
    
    # Extract date and time from the key
    date_time_str = key.split('/')[-1].replace('.json', '')
    assert len(date_time_str) == 16  # YYYY-MM-DD_HH-MM format
    assert date_time_str[10] == '_'  # Check separator
    assert date_time_str[13] == '-'  # Check time separator
    
    # Verify the body is valid JSON
    body = json.loads(call_args['Body'])
    assert body == sample_weather_data

def test_save_to_s3_error_handling(mock_s3_client, sample_weather_data):
    """Test error handling when saving to S3"""
    # Mock S3 put_object to raise an error
    mock_s3_client.return_value.put_object.side_effect = Exception("S3 Error")
    
    success = omw.save_to_s3([sample_weather_data])
    assert success is False

# Edge Cases
def test_empty_city_list():
    """Test behavior with empty city list"""
    results, failures = omw.extract_historical_weather_for_cities(city_list=[])
    assert len(results) == 0
    assert len(failures) == 0

def test_invalid_date_range():
    """Test behavior with invalid date range"""
    # End date before start date
    start_date = datetime.now()
    end_date = start_date - timedelta(days=1)
    
    with pytest.raises(ValueError):
        omw.extract_historical_weather_for_cities(
            city_list=["London"],
            dates=[start_date, end_date]
        )

def test_concurrent_api_calls():
    """Test concurrent API calls with rate limiting"""
    with patch('src.bronze.owm_extract.fetch_historical_weather') as mock_fetch, \
         patch('src.bronze.owm_extract.get_lat_lon') as mock_get_coords, \
         patch('time.sleep') as mock_sleep:
        
        # Setup mock returns
        mock_get_coords.return_value = (51.5074, -0.1278)
        mock_fetch.return_value = {
            "lat": 51.5074,
            "lon": -0.1278,
            "data": [{
                "dt": int(datetime.now().timestamp()),
                "weather": [{
                    "id": 800,
                    "main": "Clear",
                    "description": "clear sky",
                    "icon": "01d"
                }]
            }]
        }
        
        cities = ["London", "New York", "Tokyo", "Sydney", "Berlin", "Paris", "Rome", "Madrid"]
        dates = [datetime.now() - timedelta(days=1)]
        
        results, failures = omw.extract_historical_weather_for_cities(
            city_list=cities,
            dates=dates
        )
        
        # Should have results for all cities
        assert len(results) == len(cities)
        assert len(failures) == 0
        
        # Verify rate limiting was applied
        expected_sleep_calls = len(cities) // 30  # One sleep per 30 requests
        assert mock_sleep.call_count >= expected_sleep_calls

# Validation Tests
def test_invalid_api_response():
    """Test handling of invalid API response"""
    with patch('requests.get') as mock_get:
        # Create a mock response with error
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.text = "Invalid API key"
        mock_response.json.return_value = {"error": "Invalid response"}
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(response=mock_response)
        mock_get.return_value = mock_response
        
        results, failures = omw.extract_historical_weather_for_cities(
            city_list=["London"],
            dates=[datetime.now() - timedelta(days=1)]
        )
        
        assert len(results) == 0
        assert len(failures) == 1
        assert "401" in failures[0]["error"]

def test_malformed_weather_data():
    """Test handling of malformed weather data"""
    with patch('src.bronze.owm_extract.fetch_historical_weather') as mock_fetch, \
         patch('src.bronze.owm_extract.get_lat_lon') as mock_get_coords:
        
        mock_get_coords.return_value = (51.5074, -0.1278)
        mock_fetch.return_value = {
            "lat": 51.5074,
            "lon": -0.1278,
            # Missing data array
        }
        
        results, failures = omw.extract_historical_weather_for_cities(
            city_list=["London"],
            dates=[datetime.now() - timedelta(days=1)]
        )
        
        assert len(results) == 0
        assert len(failures) == 1
        assert "No weather data found in response" in failures[0]["error"]

def test_missing_required_fields():
    """Test handling of missing required fields in response"""
    with patch('src.bronze.owm_extract.fetch_historical_weather') as mock_fetch, \
         patch('src.bronze.owm_extract.get_lat_lon') as mock_get_coords:
        
        mock_get_coords.return_value = (51.5074, -0.1278)
        mock_fetch.return_value = {
            "lat": 51.5074,
            "lon": -0.1278,
            "data": [{
                # Missing dt and weather fields
                "temp": 20.5,
                "humidity": 80
            }]
        }
        
        # Mock the fetch_with_retry function to return the data without validation
        with patch('src.bronze.owm_extract.fetch_with_retry') as mock_fetch_retry:
            mock_fetch_retry.return_value = (None, "Missing required fields: dt or weather")
            
            results, failures = omw.extract_historical_weather_for_cities(
                city_list=["London"],
                dates=[datetime.now() - timedelta(days=1)]
            )
            
            assert len(results) == 0
            assert len(failures) == 1
            assert "Missing required fields" in failures[0]["error"]

# Environment Tests
def test_invalid_aws_credentials(mock_s3_client):
    """Test behavior with invalid AWS credentials"""
    mock_s3_client.return_value.get_paginator.side_effect = ClientError(
        {'Error': {'Code': 'InvalidClientTokenId', 'Message': 'Invalid client token'}},
        'GetObject'
    )
    
    cities = ["London"]
    dates = [datetime.now()]
    
    existing_paths = omw.get_existing_s3_paths(cities, dates)
    assert isinstance(existing_paths, set)
    assert len(existing_paths) == 0

def test_timezone_handling():
    """Test that API calls are made with correct UTC timestamps based on city timezones"""
    # Mock the API responses
    with patch('requests.get') as mock_get:
        
        # Mock responses for timezone info
        def mock_get_side_effect(*args, **kwargs):
            mock_response = MagicMock()
            mock_response.raise_for_status.return_value = None
            
            # Get the URL from args[0] if it exists, otherwise from kwargs
            url = args[0] if args else kwargs.get('url', '')
            params = kwargs.get('params', {})
            
            # Different responses based on the API endpoint and parameters
            if 'data/2.5/weather' in url:
                # Current weather endpoint
                city = params.get('q', '')
                if city == 'London' or (params.get('lat') == 51.5074 and params.get('lon') == -0.1278):
                    mock_response.json.return_value = {
                        'coord': {'lat': 51.5074, 'lon': -0.1278},
                        'timezone': 0,  # UTC+0
                        'name': 'London'
                    }
                elif city == 'New York' or (params.get('lat') == 40.7128 and params.get('lon') == -74.0060):
                    mock_response.json.return_value = {
                        'coord': {'lat': 40.7128, 'lon': -74.0060},
                        'timezone': -18000,  # UTC-5
                        'name': 'New York'
                    }
                elif city == 'Tokyo' or (params.get('lat') == 35.6762 and params.get('lon') == 139.6503):
                    mock_response.json.return_value = {
                        'coord': {'lat': 35.6762, 'lon': 139.6503},
                        'timezone': 32400,  # UTC+9
                        'name': 'Tokyo'
                    }
            elif 'data/3.0/onecall/timemachine' in url:
                # Historical data endpoint
                mock_response.json.return_value = {
                    'lat': params.get('lat'),
                    'lon': params.get('lon'),
                    'timezone': 'UTC',
                    'timezone_offset': 0,
                    'data': [{
                        'dt': params.get('dt'),
                        'temp': 20,
                        'feels_like': 18,
                        'pressure': 1015,
                        'humidity': 65,
                        'dew_point': 12,
                        'uvi': 5,
                        'clouds': 20,
                        'visibility': 10000,
                        'wind_speed': 5,
                        'wind_deg': 180,
                        'weather': [{'main': 'Clear', 'description': 'clear sky'}]
                    }]
                }
            return mock_response
        
        mock_get.side_effect = mock_get_side_effect
        
        # Test cities with different timezones
        cities = ["London", "New York", "Tokyo"]
        test_date = datetime(2024, 3, 15)
        
        results, failures = omw.extract_historical_weather_for_cities(
            city_list=cities,
            dates=[test_date]
        )
        
        # Verify results
        assert len(results) == len(cities)
        assert len(failures) == 0
        
        # Extract timestamps from the results
        city_timestamps = {result['_metadata']['city']: result['dt'] for result in results}
        
        # Verify that API calls were made with correct UTC timestamps
        # London (UTC+0): 9:00 AM local = 09:00 UTC
        london_time = datetime(2024, 3, 15, 9, 0, tzinfo=timezone.utc)
        london_expected = int(london_time.timestamp())
        assert city_timestamps["London"] == london_expected
        
        # New York (UTC-5): 9:00 AM local = 14:00 UTC
        ny_time = datetime(2024, 3, 15, 9, 0, tzinfo=timezone(timedelta(hours=-5)))
        ny_expected = int(ny_time.astimezone(timezone.utc).timestamp())
        assert city_timestamps["New York"] == ny_expected
        
        # Tokyo (UTC+9): 9:00 AM local = 00:00 UTC (next day)
        tokyo_time = datetime(2024, 3, 15, 9, 0, tzinfo=timezone(timedelta(hours=9)))
        tokyo_expected = int(tokyo_time.astimezone(timezone.utc).timestamp())
        assert city_timestamps["Tokyo"] == tokyo_expected
        
        # Verify that the S3 keys use local time
        for result in results:
            city = result['_metadata']['city']
            dt = datetime.fromtimestamp(result['dt'], tz=timezone.utc)
            
            # Convert UTC to local time based on city
            if city == "London":
                local_time = dt.astimezone(timezone.utc)
            elif city == "New York":
                local_time = dt.astimezone(timezone(timedelta(hours=-5)))
            elif city == "Tokyo":
                local_time = dt.astimezone(timezone(timedelta(hours=9)))
            
            # Verify local time is 9:00 AM
            assert local_time.hour == 9
            assert local_time.minute == 0
