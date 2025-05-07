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

# API Configuration
API_KEY = os.getenv('OPENWEATHERMAP_API_KEY')
BASE_URL = "https://api.openweathermap.org/data/2.5"
HISTORY_URL = "https://history.openweathermap.org/data/3.0/history/timemachine"

# AWS Configuration
AWS_ACCESS_KEY = os.getenv('AWS_ACCESS_KEY')
AWS_SECRET_KEY = os.getenv('AWS_SECRET_KEY')
S3_BUCKET = os.getenv('AWS_S3_BUCKET_NAME')
AWS_REGION = os.getenv('AWS_REGION', 'us-east-1')

# Rate limiting configuration
CALLS_PER_MINUTE = 30
ONE_MINUTE = 60

# Retry configuration
RETRY_DELAYS = [10, 30, 60]  # seconds

def get_lat_lon(city: str) -> Tuple[float, float]:
    """
    Get latitude and longitude for a city using OpenWeatherMap API
    
    Args:
        city: City name
        
    Returns:
        Tuple of (latitude, longitude)
        
    Raises:
        Exception: If API call fails
    """
    params = {
        'q': city,
        'appid': API_KEY
    }
    
    response = requests.get(f"{BASE_URL}/weather", params=params)
    response.raise_for_status()
    data = response.json()
    
    return data['coord']['lat'], data['coord']['lon']

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
    params = {
        'lat': lat,
        'lon': lon,
        'dt': dt,
        'appid': API_KEY,
        'units': 'metric'
    }
    
    response = requests.get(HISTORY_URL, params=params)
    response.raise_for_status()
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
            
            # Add metadata
            data['_metadata'] = {
                'source': 'openweathermap',
                'ingestion_timestamp': datetime.datetime.now().isoformat(),
                'city': city
            }
            
            return data, None
            
        except Exception as e:
            logger.warning(f"Attempt failed for {city} at {dt}: {str(e)}")
            if delay == RETRY_DELAYS[-1]:  # Last retry
                return None, str(e)
            time.sleep(delay)
    
    return None, "All retry attempts failed"

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
    """
    if dates is None:
        # Default date range: Jan 1, 2025 to Feb 25, 2025
        start_date = datetime.datetime(2025, 1, 1)
        end_date = datetime.datetime(2025, 2, 25)
        dates = [start_date + datetime.timedelta(days=x) for x in range((end_date - start_date).days + 1)]
    
    # Convert dates to 9:00 EST timestamps
    est_dates = []
    for date in dates:
        est_time = datetime.datetime.combine(date.date(), datetime.time(9, 0))
        est_time = est_time.replace(tzinfo=datetime.timezone(datetime.timedelta(hours=-5)))  # EST
        est_dates.append(int(est_time.timestamp()))
    
    results = []
    failures = []
    
    # Get lat/lon for all cities first
    city_coords = {}
    for city in city_list:
        try:
            lat, lon = get_lat_lon(city)
            city_coords[city] = (lat, lon)
        except Exception as e:
            logger.error(f"Failed to get coordinates for {city}: {str(e)}")
            failures.append({
                'city': city,
                'date': None,
                'error': str(e)
            })
    
    # Process each city and date combination
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        future_to_params = {}
        
        for city, (lat, lon) in city_coords.items():
            for dt in est_dates:
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
                    results.append(data)
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
    """
    if batch_id is None:
        batch_id = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Initialize S3 client
    s3_client = boto3.client(
        's3',
        aws_access_key_id=AWS_ACCESS_KEY,
        aws_secret_access_key=AWS_SECRET_KEY
    )
    
    try:
        # Group data by city and date
        for item in data:
            city = item['_metadata']['city'].lower()
            dt = datetime.datetime.fromtimestamp(item['dt'])
            
            # Create S3 path with new format
            s3_key = f"bronze/openweathermap/city={city}/year={dt.year}/month={dt.month:02d}/day={dt.day:02d}/weather_9am.json"
            
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