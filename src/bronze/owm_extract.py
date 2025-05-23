import os
import json
import requests
import boto3
import datetime
from dotenv import load_dotenv
import logging
from typing import List, Dict, Any, Optional, Tuple, Union
import time
from ratelimit import limits, sleep_and_retry
import argparse
import concurrent.futures
from functools import partial
import pandera.pandas as pa
from pandera.typing import Series
import pandas as pd
import numpy as np
from calendar import monthrange
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import jsonschema

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/owm_extract.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Define JSON schema for OpenWeatherMap bronze data
OPENWEATHERMAP_BRONZE_JSONSCHEMA = {
    "type": "object",
    "properties": {
        "lat": {"type": "number", "minimum": -90, "maximum": 90},
        "lon": {"type": "number", "minimum": -180, "maximum": 180},
        "date": {"type": "string"},
        "cloud_cover_afternoon": {"type": "integer", "minimum": 0, "maximum": 100},
        "humidity_afternoon": {"type": "integer", "minimum": 0, "maximum": 100},
        "precipitation_total": {"type": "number"},
        "temperature_min": {"type": "number"},
        "temperature_max": {"type": "number"},
        "temperature_afternoon": {"type": "number"},
        "temperature_night": {"type": "number"},
        "temperature_evening": {"type": "number"},
        "temperature_morning": {"type": "number"},
        "pressure_afternoon": {"type": "integer"},
        "wind_max_speed": {"type": "number"},
        "wind_max_direction": {"type": "integer", "minimum": 0, "maximum": 360},
        "source": {"type": "string"},
        "ingestion_timestamp": {"type": "string"},
        "city": {"type": "string"},
        "timezone": {"type": "string"}
    },
    "required": [
        "lat", "lon", "date", "cloud_cover_afternoon", "humidity_afternoon", "precipitation_total",
        "temperature_min", "temperature_max", "temperature_afternoon", "temperature_night",
        "temperature_evening", "temperature_morning", "pressure_afternoon", "wind_max_speed",
        "wind_max_direction", "source", "ingestion_timestamp", "city", "timezone"
    ]
}

def validate_bronze_data(data: dict) -> bool:
    """
    Validate the bronze data using JSON schema
    Args:
        data: Dictionary containing weather data
    Returns:
        bool: True if validation passes, False otherwise
    """
    try:
        jsonschema.validate(instance=data, schema=OPENWEATHERMAP_BRONZE_JSONSCHEMA)
        return True
    except jsonschema.ValidationError as e:
        logger.error(f"Data validation failed: {e.message}")
        return False

# Constants
BASIC_URL = "https://api.openweathermap.org/data/2.5/weather"
HISTORY_URL = "https://api.openweathermap.org/data/3.0/onecall/day_summary"

# Rate limiting configuration
CALLS_PER_MINUTE = 30
ONE_MINUTE = 60

# Retry configuration
RETRY_DELAYS = [10, 30, 60]  # seconds

def get_lat_lon(city: str, api_key: str) -> Tuple[float, float]:
    """
    Get latitude and longitude for a city using OpenWeatherMap API
    
    Args:
        city: City name
        api_key: OpenWeatherMap API key
        
    Returns:
        Tuple of (latitude, longitude)
        
    Raises:
        ValueError: If API key is not set
        Exception: If API call fails
    """
    params = {
        'q': city,
        'appid': api_key
    }
    
    try:
        response = requests.get(BASIC_URL, params=params)
        response.raise_for_status()
        data = response.json()
        
        if 'coord' not in data:
            raise ValueError("No coordinates found in response")
            
        return data['coord']['lat'], data['coord']['lon']
    except requests.exceptions.HTTPError as e:
        error_msg = str(e)
        if e.response is not None:
            error_msg = f"{e.response.status_code}: {e.response.text}"
        raise Exception(error_msg)

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=10, max=60),
    retry=retry_if_exception_type((requests.exceptions.HTTPError, KeyError, ValueError, Exception)),
    before_sleep=lambda retry_state: logger.warning(
        f"Attempt {retry_state.attempt_number} failed. Retrying soon..."
    )
)
def fetch_historical_weather(lat: float, lon: float, date: str, api_key: str) -> Dict[str, Any]:
    """
    Fetch historical weather data for a specific location and date
    
    Args:
        lat: Latitude
        lon: Longitude
        date: Date in YYYY-MM-DD format
        api_key: OpenWeatherMap API key
        
    Returns:
        Dictionary containing weather data
        
    Raises:
        Exception: If API call fails after all retries
    """
    logger.info(f"Making API call for coordinates ({lat}, {lon}) for date {date}")
    params = {
        'lat': lat,
        'lon': lon,
        'date': date,
        'appid': api_key,
        'units': 'metric'
    }
    
    response = requests.get(HISTORY_URL, params=params)
    response.raise_for_status()
    logger.info(f"Successfully received API response for date {date}")
    return response.json()

def get_existing_s3_paths(cities: List[str], months: List[Tuple[int, int]], aws_config: Optional[Dict[str, str]] = None) -> set:
    """
    Get a set of existing S3 paths for given cities and months using a single S3 listing call
    
    Args:
        cities: List of city names
        months: List of (year, month) tuples
        aws_config: Optional dictionary containing AWS configuration
        
    Returns:
        set: Set of existing S3 paths
    """
    if aws_config is None:
        raise ValueError("AWS configuration is required")
        
    AWS_ACCESS_KEY = aws_config.get('access_key')
    AWS_SECRET_KEY = aws_config.get('secret_key')
    S3_BUCKET = aws_config.get('bucket')
    AWS_REGION = aws_config.get('region')
    
    if not AWS_ACCESS_KEY or not AWS_SECRET_KEY:
        raise ValueError("AWS credentials (AWS_ACCESS_KEY and AWS_SECRET_KEY) must be set")
    if not S3_BUCKET:
        raise ValueError("AWS_S3_BUCKET_NAME environment variable is not set")
    
    s3_client = boto3.client(
        's3',
        aws_access_key_id=AWS_ACCESS_KEY,
        aws_secret_access_key=AWS_SECRET_KEY,
        region_name=AWS_REGION
    )
    
    # Create prefix for listing (e.g., "bronze/openweathermap/city=")
    prefix = "bronze/openweathermap/city="
    existing_paths = set()
    
    try:
        # List all objects with the prefix
        paginator = s3_client.get_paginator('list_objects_v2')
        for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=prefix):
            if 'Contents' in page:
                for obj in page['Contents']:
                    # Normalize the path by replacing spaces with underscores
                    normalized_key = obj['Key'].replace(' ', '_')
                    existing_paths.add(normalized_key)
        logger.info(f"Found {len(existing_paths)} existing files in S3")
    except Exception as e:
        logger.warning(f"Error listing S3 objects: {str(e)}")
        return set()
    
    return existing_paths

def extract_historical_weather_for_cities(
    city_list: List[str],
    api_key: str,
    months: List[Tuple[int, int]],
    aws_config: Optional[Dict[str, str]] = None
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Extract historical weather data for multiple cities and months
    
    Args:
        city_list: List of city names
        api_key: OpenWeatherMap API key
        months: List of (year, month) tuples. If None, uses default range
        aws_config: Dictionary containing AWS configuration
        
    Returns:
        Tuple of (successful_results, failures)
    """
    if not city_list:
        return [], []
    
    results = []
    failures = []
    
    # Get all existing S3 paths in one call
    existing_paths = get_existing_s3_paths(city_list, months, aws_config)
    
    # Process each city and month combination
    for city in city_list:
        # Normalize city name for path checking
        normalized_city = city.lower().replace(' ', '_')
        
        # Check if data already exists in S3
        months_to_fetch = []
        for year, month in months:
            # Create S3 key for this month
            s3_key = f"bronze/openweathermap/city={normalized_city}/year={year}/{month:02d}.json"
            
            if s3_key not in existing_paths:
                months_to_fetch.append((year, month, s3_key))
                logger.info(f"Need to fetch data for {city} in {year}-{month:02d} - file {s3_key} not found in S3")
            else:
                logger.info(f"Skipping {city} for {year}-{month:02d} - file {s3_key} already exists in S3")
        
        if len(months_to_fetch) == 0:
            logger.info(f"All data exists in S3 for {city}, skipping API calls")
            continue
            
        try:
            lat, lon = get_lat_lon(city, api_key)
        except Exception as e:
            logger.error(f"Failed to get coordinates for {city}: {str(e)}")
            failures.append({
                'city': city,
                'year': None,
                'month': None,
                'error': str(e)
            })
            continue
        
        # Process each month
        for year, month, s3_key in months_to_fetch:
            daily_data = []
            # Get number of days in the month
            _, num_days = monthrange(year, month)
            
            # Fetch data for each day
            for day in range(1, num_days + 1):
                date = f"{year}-{month:02d}-{day:02d}"
                try:
                    data = fetch_historical_weather(lat, lon, date, api_key)
                    daily_data.append(data)
                except Exception as e:
                    failures.append({
                        'city': city,
                        'year': year,
                        'month': month,
                        'day': day,
                        'error': str(e)
                    })
            
            if daily_data:
                # Create monthly file with metadata
                monthly_data = {
                    'metadata': {
                        'city': city,
                        'year': year,
                        'month': month,
                        'source': 'openweathermap',
                        'ingestion_timestamp': datetime.datetime.now().isoformat(),
                        'record_count': len(daily_data)
                    },
                    'data': daily_data
                }
                # Add to results
                results.append({
                    'city': city,
                    'year': year,
                    'month': month,
                    'data': monthly_data,
                    's3_key': s3_key
                })
    
    return results, failures

def save_to_s3(data: List[Dict[str, Any]], aws_config: Dict[str, str]) -> bool:
    """
    Save the weather data to S3 bronze layer with monthly batching
    
    Args:
        data: List of dictionaries containing monthly weather data
        aws_config: Dictionary containing AWS configuration
        
    Returns:
        bool: True if save was successful, False otherwise
    """
    if not data:
        return True
    
    # Initialize S3 client
    s3_client = boto3.client(
        's3',
        aws_access_key_id=aws_config['access_key'],
        aws_secret_access_key=aws_config['secret_key'],
        region_name=aws_config['region']
    )
    
    try:
        for month_data in data:
            # Convert month data to JSON
            json_data = json.dumps(month_data)
            
            # Upload to S3
            s3_client.put_object(
                Bucket=aws_config['bucket'],
                Key=month_data['s3_key'],
                Body=json_data,
                ContentType='application/json'
            )
            
            logger.info(f"Successfully saved data to {month_data['s3_key']}")
        
        return True
    
    except Exception as e:
        logger.error(f"Error saving data to S3: {str(e)}", exc_info=False)
        return False

def main() -> int:
    """
    Main function to extract weather data and save to S3
    
    Returns:
        int: 0 for success, 1 for failure
    """
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Extract historical weather data from OpenWeatherMap API')
    parser.add_argument('--cities', nargs='+', help='List of cities to fetch weather data for')
    parser.add_argument('--start-month', type=str, help='Start month (YYYY-MM)')
    parser.add_argument('--end-month', type=str, help='End month (YYYY-MM)')
    args = parser.parse_args()

    # Get environment variables
    api_key = os.getenv('OPENWEATHERMAP_API_KEY')
    if not api_key:
        logger.error("OPENWEATHERMAP_API_KEY environment variable is not set")
        return 1
        
    aws_config = {
        'access_key': os.getenv('AWS_ACCESS_KEY'),
        'secret_key': os.getenv('AWS_SECRET_KEY'),
        'bucket': os.getenv('AWS_S3_BUCKET_NAME'),
        'region': os.getenv('AWS_REGION', 'us-east-1')
    }

    logger.info("Starting OpenWeatherMap historical data extraction")
    
    try:
        # Parse months if provided, otherwise use default range
        if args.start_month and args.end_month:
            start_date = datetime.datetime.strptime(args.start_month, '%Y-%m')
            end_date = datetime.datetime.strptime(args.end_month, '%Y-%m')
        else:
            # Default to Jan 2024 to Mar 2024
            start_date = datetime.datetime(2024, 1, 1)
            end_date = datetime.datetime(2024, 3, 1)
            logger.info(f"Using default date range: {start_date.strftime('%Y-%m')} to {end_date.strftime('%Y-%m')}")
        
        # Generate list of (year, month) tuples
        months = []
        current = start_date
        while current <= end_date:
            months.append((current.year, current.month))
            # Move to next month
            if current.month == 12:
                current = current.replace(year=current.year + 1, month=1)
            else:
                current = current.replace(month=current.month + 1)
        
        # Fetch weather data
        city_list = args.cities if args.cities else ['London', 'New York']
        weather_data, failures = extract_historical_weather_for_cities(
            city_list=city_list,
            api_key=api_key,
            months=months,
            aws_config=aws_config
        )
        
        if not weather_data:
            logger.error("No weather data retrieved. Exiting.")
            return 1
        
        # Log summary of failures
        if failures:
            logger.warning(f"Failed to retrieve data for {len(failures)} city-month combinations:")
            for fail in failures:
                logger.warning(f"City: {fail['city']}, Year: {fail['year']}, Month: {fail['month']}, Error: {fail['error']}")
        
        # Save to S3
        success = save_to_s3(weather_data, aws_config)
        
        if success:
            logger.info("Successfully completed OpenWeatherMap data extraction")
            return 0
        else:
            logger.error("Failed to complete OpenWeatherMap data extraction")
            return 1
            
    except Exception as e:
        logger.error(f"Unexpected error during data extraction: {str(e)}", exc_info=False)
        return 1

if __name__ == "__main__":
    exit(main())