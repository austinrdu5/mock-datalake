import pytest
from datetime import datetime, timedelta
from src.bronze import openweathermap_extract as omw
import os
from dotenv import load_dotenv

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
    assert "dt" in data
    assert "weather" in data
    assert isinstance(data["weather"], list)
    assert len(data["weather"]) > 0
    assert "main" in data["weather"][0]
    assert "description" in data["weather"][0]

def test_extract_historical_weather_for_cities(sample_city_list, sample_dates):
    """Test the full extraction process with real API calls"""
    results, failures = omw.extract_historical_weather_for_cities(
        city_list=sample_city_list,
        dates=sample_dates
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
    
    # Check failures structure
    assert isinstance(failures, list)
    for fail in failures:
        assert "city" in fail
        assert "date" in fail
        assert "error" in fail

def test_rate_limiting():
    """Test that rate limiting is working by making multiple requests"""
    cities = ["London", "New York", "Tokyo", "Sydney", "Berlin"]
    dates = [datetime.now() - timedelta(days=1)]
    
    start_time = datetime.now()
    results, failures = omw.extract_historical_weather_for_cities(
        city_list=cities,
        dates=dates
    )
    end_time = datetime.now()
    
    # Should take at least 2 seconds due to rate limiting (30 requests per minute)
    assert (end_time - start_time).total_seconds() >= 2
    
    # Should have gotten data for all cities
    assert len(results) == len(cities)
    assert len(failures) == 0

def test_retry_mechanism():
    """Test retry mechanism by temporarily using an invalid API key"""
    original_key = os.environ.get('OPENWEATHERMAP_API_KEY')
    try:
        # Temporarily set invalid key
        os.environ['OPENWEATHERMAP_API_KEY'] = 'invalid_key'
        
        results, failures = omw.extract_historical_weather_for_cities(
            city_list=["London"],
            dates=[datetime.now() - timedelta(days=1)]
        )
        
        # Should have failures after retries
        assert len(failures) == 1
        assert "401" in failures[0]["error"]  # Unauthorized error
        
    finally:
        # Restore original key
        if original_key:
            os.environ['OPENWEATHERMAP_API_KEY'] = original_key
        else:
            del os.environ['OPENWEATHERMAP_API_KEY']

def test_default_date_range():
    """Test that default date range works correctly"""
    results, failures = omw.extract_historical_weather_for_cities(
        city_list=["London"]
    )
    
    # Verify date range
    dates = [datetime.fromtimestamp(item["dt"]) for item in results]
    assert min(dates).date() == datetime(2025, 1, 1).date()
    assert max(dates).date() == datetime(2025, 2, 25).date()
