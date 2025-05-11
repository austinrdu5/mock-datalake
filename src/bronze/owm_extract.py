import os
import json
import requests
import boto3
import datetime
from dotenv import load_dotenv
import logging
from typing import List, Dict, Any, Optional, Tuple
import time
from ratelimit import limits, sleep_and_retry
import argparse
import concurrent.futures
from functools import partial

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

def initialize():
    """Initialize module with environment variables"""
    global API_KEY, AWS_ACCESS_KEY, AWS_SECRET_KEY, S3_BUCKET, AWS_REGION
    
    # API Configuration
    API_KEY = os.getenv('OPENWEATHERMAP_API_KEY')
    if not API_KEY:
        raise ValueError("OPENWEATHERMAP_API_KEY environment variable is not set")

    # AWS Configuration
    AWS_ACCESS_KEY = os.getenv('AWS_ACCESS_KEY')
    AWS_SECRET_KEY = os.getenv('AWS_SECRET_KEY')
    S3_BUCKET = os.getenv('AWS_S3_BUCKET_NAME')

    if not AWS_ACCESS_KEY or not AWS_SECRET_KEY:
        raise ValueError("AWS credentials (AWS_ACCESS_KEY and AWS_SECRET_KEY) must be set")
    if not S3_BUCKET:
        raise ValueError("AWS_S3_BUCKET_NAME environment variable is not set")

    AWS_REGION = os.getenv('AWS_REGION', 'us-east-1')

# Initialize module
initialize()

# Constants
BASIC_URL = "https://api.openweathermap.org/data/2.5/weather"
HISTORY_URL = "https://api.openweathermap.org/data/3.0/onecall/timemachine"

# Rate limiting configuration
CALLS_PER_MINUTE = 30
ONE_MINUTE = 60

# Retry configuration
RETRY_DELAYS = [10, 30, 60]  # seconds

def validate_env_vars():
    """Validate that required environment variables are set"""
    initialize()  # This will raise appropriate errors if env vars are missing

def get_lat_lon(city: str) -> Tuple[float, float]:
    """
    Get latitude and longitude for a city using OpenWeatherMap API
    
    Args:
        city: City name
        
    Returns:
        Tuple of (latitude, longitude)
        
    Raises:
        ValueError: If API key is not set
        Exception: If API call fails
    """
    validate_env_vars()
    
    params = {
        'q': city,
        'appid': API_KEY
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
    except Exception as e:
        raise Exception(str(e))

@sleep_and_retry
@limits(calls=CALLS_PER_MINUTE, period=ONE_MINUTE)
def fetch_historical_weather(lat: float, lon: float, dt: int) -> Dict[str, Any]:
    """
    Fetch historical weather data for a specific location and time
    
    Args:
        lat: Latitude
        lon: Longitude
        dt: Unix timestamp
        
    Returns:
        Dictionary containing weather data
        
    Raises:
        Exception: If API call fails
    """
    logger.info(f"Making API call for coordinates ({lat}, {lon}) at timestamp {dt}")
    params = {
        'lat': lat,
        'lon': lon,
        'dt': dt,
        'appid': API_KEY,
        'units': 'metric'
    }
    
    response = requests.get(HISTORY_URL, params=params)
    response.raise_for_status()
    logger.info(f"Successfully received API response for timestamp {dt}")
    return response.json()

def fetch_with_retry(lat: float, lon: float, dt: int, city: str) -> Tuple[Dict[str, Any], Optional[str]]:
    """
    Fetch weather data with retry mechanism
    
    Args:
        lat: Latitude
        lon: Longitude
        dt: Unix timestamp
        city: City name
        
    Returns:
        Tuple of (data, error_message)
    """
    for delay in RETRY_DELAYS:
        try:
            data = fetch_historical_weather(lat, lon, dt)
            
            # Extract the first data point from the array
            if 'data' in data and len(data['data']) > 0:
                weather_data = data['data'][0]
                # Merge weather data with top-level coordinates
                processed_data = {
                    'lat': data['lat'],
                    'lon': data['lon'],
                    'dt': dt,  # Use the input timestamp instead of the response timestamp
                    'temp': weather_data.get('temp'),
                    'feels_like': weather_data.get('feels_like'),
                    'pressure': weather_data.get('pressure'),
                    'humidity': weather_data.get('humidity'),
                    'dew_point': weather_data.get('dew_point'),
                    'uvi': weather_data.get('uvi'),
                    'clouds': weather_data.get('clouds'),
                    'visibility': weather_data.get('visibility'),
                    'wind_speed': weather_data.get('wind_speed'),
                    'wind_deg': weather_data.get('wind_deg'),
                    'wind_gust': weather_data.get('wind_gust'),
                    'weather': weather_data.get('weather', []),
                    '_metadata': {
                        'source': 'openweathermap',
                        'ingestion_timestamp': datetime.datetime.now().isoformat(),
                        'city': city,
                        'timezone': data.get('timezone'),
                        'timezone_offset': data.get('timezone_offset')
                    }
                }
                
                # Validate required fields
                if not all(key in processed_data for key in ['dt', 'weather']):
                    raise KeyError("Missing required fields: dt or weather")
                    
                return processed_data, None
            else:
                raise ValueError("No weather data found in response")
            
        except requests.exceptions.HTTPError as e:
            error_msg = str(e)
            if e.response is not None:
                error_msg = f"{e.response.status_code}: {e.response.text}"
            logger.warning(f"Attempt failed for {city} at {dt}: {error_msg}")
            if delay == RETRY_DELAYS[-1]:  # Last retry
                return None, error_msg
            time.sleep(delay)
        except (KeyError, ValueError) as e:
            error_msg = str(e)
            logger.warning(f"Attempt failed for {city} at {dt}: {error_msg}")
            if delay == RETRY_DELAYS[-1]:  # Last retry
                return None, error_msg
            time.sleep(delay)
        except Exception as e:
            error_msg = str(e)
            logger.warning(f"Attempt failed for {city} at {dt}: {error_msg}")
            if delay == RETRY_DELAYS[-1]:  # Last retry
                return None, error_msg
            time.sleep(delay)
    
    return None, "All retry attempts failed"

def get_existing_s3_paths(cities: List[str], dates: List[datetime.datetime]) -> set:
    """
    Get a set of existing S3 paths for given cities and dates using a single S3 listing call
    
    Args:
        cities: List of city names
        dates: List of datetime objects
        
    Returns:
        set: Set of existing S3 paths
        
    Raises:
        ValueError: If AWS credentials or bucket name are not set
    """
    if not AWS_ACCESS_KEY or not AWS_SECRET_KEY:
        raise ValueError("AWS credentials (AWS_ACCESS_KEY and AWS_SECRET_KEY) must be set")
    if not S3_BUCKET:
        raise ValueError("AWS_S3_BUCKET_NAME environment variable is not set")
    
    s3_client = boto3.client(
        's3',
        aws_access_key_id=AWS_ACCESS_KEY,
        aws_secret_access_key=AWS_SECRET_KEY
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
        # Return empty set on error
        return set()
    
    return existing_paths

def extract_historical_weather_for_cities(
    city_list: List[str],
    dates: Optional[List[datetime.datetime]] = None
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Extract historical weather data for multiple cities and dates
    
    Args:
        city_list: List of city names
        dates: List of dates to fetch data for. If None, uses default range
        
    Returns:
        Tuple of (successful_results, failures)
        
    Raises:
        ValueError: If dates are not in chronological order or if API key is not set
    """
    validate_env_vars()
    
    if not city_list:
        return [], []
        
    if dates is None:
        # Default date range: Jan 1, 2025 to Jan 3, 2025 (3 days)
        start_date = datetime.datetime(2025, 1, 1)
        end_date = datetime.datetime(2025, 1, 3)
        dates = [start_date + datetime.timedelta(days=x) for x in range((end_date - start_date).days + 1)]
    else:
        # Validate dates are in chronological order
        sorted_dates = sorted(dates)
        if dates != sorted_dates:
            raise ValueError("Dates must be provided in chronological order")
    
    results = []
    failures = []
    
    # Get all existing S3 paths in one call
    existing_paths = get_existing_s3_paths(city_list, dates)
    
    # Process each city and date combination
    for city in city_list:
        # Normalize city name for path checking
        normalized_city = city.lower().replace(' ', '_')
        
        # Check if we need to fetch coordinates
        need_coords = False
        dates_to_fetch = []
        for date in dates:
            # First get the timezone for this city
            try:
                # Get current weather to get timezone info
                params = {
                    'q': city,
                    'appid': API_KEY
                }
                response = requests.get(BASIC_URL, params=params)
                response.raise_for_status()
                timezone_data = response.json()
                
                # Get timezone offset in seconds
                timezone_offset = timezone_data.get('timezone', 0)
                
                # Create local time (9:00 AM) in the city's timezone
                local_time = datetime.datetime.combine(date.date(), datetime.time(9, 0))
                local_tz = datetime.timezone(datetime.timedelta(seconds=timezone_offset))
                local_time = local_time.replace(tzinfo=local_tz)
                
                # Convert to UTC timestamp for API
                utc_time = local_time.astimezone(datetime.timezone.utc)
                dt = int(utc_time.timestamp())
                
                # Use UTC time for S3 key with normalized city name
                s3_key = f"bronze/openweathermap/city={normalized_city}/{utc_time.strftime('%Y-%m-%d_%H-%M')}UTC.json"
                
                if s3_key not in existing_paths:
                    need_coords = True
                    dates_to_fetch.append((date, dt, s3_key))
                    logger.info(f"Need to fetch data for {city} on {date.date()} - file {s3_key} not found in S3")
                else:
                    logger.info(f"Skipping {city} on {date.date()} - file {s3_key} already exists in S3")
            except Exception as e:
                logger.error(f"Failed to get timezone for {city}: {str(e)}")
                failures.append({
                    'city': city,
                    'date': date,
                    'error': f"Timezone error: {str(e)}"
                })
                continue
        
        if not need_coords:
            logger.info(f"All data exists in S3 for {city}, skipping API calls")
            continue
            
        try:
            lat, lon = get_lat_lon(city)
        except Exception as e:
            logger.error(f"Failed to get coordinates for {city}: {str(e)}")
            failures.append({
                'city': city,
                'date': None,
                'error': str(e)
            })
            continue
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            future_to_params = {}
            
            for date, dt, s3_key in dates_to_fetch:
                future = executor.submit(fetch_with_retry, lat, lon, dt, city)
                future_to_params[future] = (city, dt)
            
            for future in concurrent.futures.as_completed(future_to_params):
                city, dt = future_to_params[future]
                try:
                    data, error = future.result()
                    if error:
                        failures.append({
                            'city': city,
                            'date': datetime.datetime.fromtimestamp(dt),
                            'error': error
                        })
                    else:
                        try:
                            # Validate required fields
                            if not all(key in data for key in ['dt', 'weather']):
                                raise KeyError("Missing required fields: dt or weather")
                            results.append(data)
                        except KeyError as e:
                            failures.append({
                                'city': city,
                                'date': datetime.datetime.fromtimestamp(dt),
                                'error': f"KeyError: {str(e)}"
                            })
                except Exception as e:
                    failures.append({
                        'city': city,
                        'date': datetime.datetime.fromtimestamp(dt),
                        'error': str(e)
                    })
    
    return results, failures

def save_to_s3(data: List[Dict[str, Any]], batch_id: Optional[str] = None) -> bool:
    """
    Save the weather data to S3 bronze layer with new path structure
    
    Args:
        data: List of dictionaries containing weather data
        batch_id: Optional batch identifier. If None, current timestamp will be used
        
    Returns:
        bool: True if save was successful, False otherwise
        
    Raises:
        ValueError: If AWS credentials or bucket name are not set
    """
    validate_env_vars()
    
    if not data:  # Return early if no data to save
        return True
        
    if batch_id is None:
        batch_id = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Initialize S3 client
    s3_client = boto3.client(
        's3',
        aws_access_key_id=AWS_ACCESS_KEY,
        aws_secret_access_key=AWS_SECRET_KEY,
        region_name=AWS_REGION
    )
    
    try:
        # Group data by city and date
        for item in data:
            # Normalize city name: lowercase and replace spaces with underscores
            city = item['_metadata']['city'].lower().replace(' ', '_')
            
            # Convert timestamp to UTC datetime
            dt = datetime.datetime.fromtimestamp(item['dt'], tz=datetime.timezone.utc)
            
            # Create S3 path with new format using UTC timestamp
            s3_key = f"bronze/openweathermap/city={city}/{dt.strftime('%Y-%m-%d_%H-%M')}UTC.json"
            
            # Convert single item to JSON
            json_data = json.dumps(item)
            
            # Upload to S3
            s3_client.put_object(
                Bucket=S3_BUCKET,
                Key=s3_key,
                Body=json_data,
                ContentType='application/json'
            )
            
            logger.info(f"Successfully saved data to {s3_key}")
        
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
    parser.add_argument('--start-date', type=str, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, help='End date (YYYY-MM-DD)')
    args = parser.parse_args()

    # Validate API key
    if not API_KEY:
        logger.error("OPENWEATHERMAP_API_KEY environment variable is not set")
        return 1

    logger.info("Starting OpenWeatherMap historical data extraction")
    
    try:
        # Parse dates if provided
        dates = None
        if args.start_date and args.end_date:
            start_date = datetime.datetime.strptime(args.start_date, '%Y-%m-%d')
            end_date = datetime.datetime.strptime(args.end_date, '%Y-%m-%d')
            dates = [start_date + datetime.timedelta(days=x) for x in range((end_date - start_date).days + 1)]
        
        # Fetch weather data
        city_list = args.cities if args.cities else ['London', 'New York', 'Tokyo', 'Sydney', 'Berlin']
        weather_data, failures = extract_historical_weather_for_cities(city_list=city_list, dates=dates)
        
        if not weather_data:
            logger.error("No weather data retrieved. Exiting.")
            return 1
        
        # Log summary of failures
        if failures:
            logger.warning(f"Failed to retrieve data for {len(failures)} city-date combinations:")
            for fail in failures:
                logger.warning(f"City: {fail['city']}, Date: {fail['date']}, Error: {fail['error']}")
        
        # Save to S3
        success = save_to_s3(weather_data)
        
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