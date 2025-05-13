import os
import requests
import json
import logging
import time
import boto3
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
import random
from typing import Dict, List, Optional, Union, Any
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    force=True  # This ensures we can reconfigure logging
)
logger = logging.getLogger(__name__)

# Alpha Vantage API configuration
ALPHA_VANTAGE_API_KEY = os.environ.get('ALPHA_VANTAGE_API_KEY')
if not ALPHA_VANTAGE_API_KEY:
    raise ValueError("ALPHA_VANTAGE_API_KEY environment variable not set")
BASE_URL = 'https://www.alphavantage.co/query'

# AWS S3 configuration
AWS_ACCESS_KEY = os.environ.get('AWS_ACCESS_KEY')
AWS_SECRET_KEY = os.environ.get('AWS_SECRET_KEY')
S3_BUCKET_NAME = os.environ.get('AWS_S3_BUCKET_NAME')
S3_PREFIX = 'bronze/alphavantage'

# Debug logging - show last 4 chars of keys
logger.info(f"AWS_ACCESS_KEY ends with: ...{AWS_ACCESS_KEY[-4:] if AWS_ACCESS_KEY else 'None'}")
logger.info(f"AWS_SECRET_KEY ends with: ...{AWS_SECRET_KEY[-4:] if AWS_SECRET_KEY else 'None'}")
logger.info(f"S3_BUCKET_NAME: {S3_BUCKET_NAME}")

# Create S3 client
s3_client = boto3.client(
    's3',
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY,
    region_name='us-east-1'  # Added explicit region
)

# Stock symbols to fetch (you can expand this list)
STOCK_SYMBOLS: List[str] = ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META']

# Alpha Vantage API Functions
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=10, min=10, max=60),
    retry=retry_if_exception_type((requests.exceptions.RequestException, KeyError)),
    before_sleep=lambda retry_state: logger.warning(f"Retrying in {retry_state.next_action.sleep} seconds...")
)
def fetch_time_series_data(symbol: str, function: str = 'TIME_SERIES_DAILY', outputsize: str = 'full') -> Optional[Dict[str, Any]]:
    """
    Fetch time series data for a given stock symbol in CSV format
    
    Args:
        symbol (str): Stock symbol (e.g., 'AAPL' for Apple)
        function (str): Alpha Vantage API function to use (default: TIME_SERIES_DAILY)
        outputsize (str): 'compact' for latest 100 data points, 'full' for up to 20 years
        
    Returns:
        Optional[Dict[str, Any]]: Dictionary containing CSV data and metadata
    """
    params = {
        'function': function,
        'symbol': symbol,
        'outputsize': outputsize,
        'datatype': 'csv',
        'apikey': ALPHA_VANTAGE_API_KEY
    }
    
    # Add timestamp for data lineage
    timestamp = datetime.now().isoformat()
    
    try:
        logger.info(f"Fetching {function} data for {symbol}")
        response = requests.get(BASE_URL, params=params)
        
        if response.status_code == 200:
            content = response.text
            logger.info(f"Response content: {content[:200]}...")
            
            # Check if response is an error message
            if any(error_text in content for error_text in ['Error Message', 'Information', 'Note']):
                logger.error(f"API Error: {content}")
                return None
            
            # Check if the response looks like CSV data
            lines = content.strip().split('\n')
            logger.info(f"Number of lines in response: {len(lines)}")
            if len(lines) >= 1:
                logger.info(f"Header line: {lines[0]}")
            
            if len(lines) < 2:  # Need at least header and one data row
                logger.error("Response does not contain valid CSV data")
                return None
                
            # Verify CSV format based on the function type
            header = lines[0].lower()
            if function == 'TIME_SERIES_DAILY_ADJUSTED':
                expected_columns = ['timestamp', 'open', 'high', 'low', 'close', 'adjusted_close', 'volume', 'dividend_amount', 'split_coefficient']
            else:  # TIME_SERIES_DAILY
                expected_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
                
            if not all(col in header.replace(' ', '_') for col in expected_columns):
                logger.error(f"Invalid CSV header format: {header}")
                return None
            
            # Create response dictionary with metadata
            data = {
                'csv_data': content,
                '_metadata': {
                    'source': 'alpha_vantage',
                    'function': function,
                    'symbol': symbol,
                    'extraction_time': timestamp,
                    'raw_params': params
                }
            }
            
            return data
        else:
            logger.error(f"Failed to fetch data: HTTP {response.status_code}")
            return None
            
    except Exception as e:
        logger.error(f"Exception during API call: {str(e)}")
        return None

def save_to_s3(data: Dict[str, Any], symbol: str, function: str, batch_timestamp: str) -> bool:
    """
    Save data to S3 bucket with appropriate partitioning
    
    Args:
        data (Dict[str, Any]): Data to save (contains csv_data and metadata)
        symbol (str): Stock symbol
        function (str): Alpha Vantage function used
        batch_timestamp (str): Timestamp of the batch run
    
    Returns:
        bool: Success status
    """
    if not data or 'csv_data' not in data:
        return False
        
    # Create filename using batch timestamp
    filename = f"{batch_timestamp}.csv"
    
    # Create S3 path with new structure
    s3_key = f"{S3_PREFIX}/{symbol}/{filename}"
    
    try:
        logger.info(f"Saving data to S3: {s3_key}")
        s3_client.put_object(
            Bucket=S3_BUCKET_NAME,
            Key=s3_key,
            Body=data['csv_data'],
            ContentType='text/csv'
        )
        return True
    except Exception as e:
        logger.error(f"Failed to save to S3: {str(e)}")
        return False
    
def check_data_exists_in_s3(symbol: str, function: str, days_threshold: int = 1, s3_client_override: Optional[boto3.client] = None) -> bool:
    """
    Check if data for a symbol exists in S3 and is recent enough
    
    Args:
        symbol (str): Stock symbol
        function (str): Alpha Vantage function used
        days_threshold (int): Number of days to consider data fresh
        s3_client_override (Optional[boto3.client]): Optional S3 client for testing
        
    Returns:
        bool: True if recent data exists, False otherwise
    """
    try:
        client = s3_client_override or s3_client
        # List objects in the symbol's directory
        prefix = f"{S3_PREFIX}/{symbol}/"
        response = client.list_objects_v2(
            Bucket=S3_BUCKET_NAME,
            Prefix=prefix
        )
        
        if 'Contents' not in response:
            return False
            
        # Get the most recent file based on filename timestamp
        latest_file = max(response['Contents'], key=lambda x: x['Key'].split('/')[-1].split('.')[0])
        filename = latest_file['Key'].split('/')[-1].split('.')[0]
        
        # Parse timestamp from filename (YYYYMMDD_HHMMSS)
        file_timestamp = datetime.strptime(filename, '%Y%m%d_%H%M%S')
        
        # Check if the file is recent enough
        time_threshold = datetime.now() - timedelta(days=days_threshold)
        return file_timestamp > time_threshold
        
    except Exception as e:
        logger.error(f"Error checking S3 for existing data: {str(e)}")
        return False

def process_symbol(symbol: str, function: str, batch_timestamp: str, force_refresh: bool = False, s3_client_override: Optional[boto3.client] = None) -> bool:
    """
    Process a single symbol
    
    Args:
        symbol (str): Stock symbol
        function (str): Alpha Vantage function to use
        batch_timestamp (str): Timestamp of the batch run
        force_refresh (bool): Force refresh data even if it exists in S3
        s3_client_override (Optional[boto3.client]): Optional S3 client for testing
        
    Returns:
        bool: Success status
    """
    # Check if data exists and is recent enough
    if not force_refresh and check_data_exists_in_s3(symbol, function, s3_client_override=s3_client_override):
        logger.info(f"Recent data for {symbol} already exists in S3, skipping API call")
        return True
        
    data = fetch_time_series_data(symbol, function)
    if data:
        return save_to_s3(data, symbol, function, batch_timestamp)
    return False

def batch_process(
    symbols: List[str] = STOCK_SYMBOLS,
    functions: Optional[List[str]] = None,
    concurrent: bool = False,
    force_refresh: bool = False
) -> int:
    """
    Batch process multiple symbols with rate limiting
    
    Args:
        symbols (List[str]): List of stock symbols to process
        functions (Optional[List[str]]): List of Alpha Vantage functions to use. Defaults to ['TIME_SERIES_DAILY']
        concurrent (bool): Whether to use concurrent processing
        force_refresh (bool): Force refresh data even if it exists in S3
        
    Returns:
        int: Number of successful operations
    """
    if functions is None:
        functions = ['TIME_SERIES_DAILY']  # Default to just daily time series data
    
    batch_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    logger.info(f"Starting batch process at {batch_timestamp} for {len(symbols)} symbols")
    
    success_count = 0
    
    # Process each symbol once with the first function
    tasks = [(symbol, functions[0]) for symbol in symbols]
    
    if concurrent:
        # Concurrent processing with rate limiting
        with ThreadPoolExecutor(max_workers=2) as executor:
            for i, (symbol, function) in enumerate(tasks):
                # Rate limiting: Alpha Vantage free tier allows 5 calls per minute
                if i > 0 and i % 5 == 0:
                    sleep_time = 60 + random.randint(1, 5)  # Add jitter
                    logger.info(f"Rate limiting: sleeping for {sleep_time} seconds")
                    time.sleep(sleep_time)
                
                # Submit task to thread pool
                future = executor.submit(process_symbol, symbol, function, batch_timestamp, force_refresh)
                if future.result():
                    success_count += 1
    else:
        # Sequential processing
        for i, (symbol, function) in enumerate(tasks):
            # Rate limiting
            if i > 0 and i % 5 == 0:
                sleep_time = 60 + random.randint(1, 5)
                logger.info(f"Rate limiting: sleeping for {sleep_time} seconds")
                time.sleep(sleep_time)
            
            if process_symbol(symbol, function, batch_timestamp, force_refresh):
                success_count += 1
    
    logger.info(f"Batch at {batch_timestamp} completed: {success_count}/{len(tasks)} successful")
    return success_count

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Alpha Vantage Data Extractor')
    parser.add_argument('--symbols', nargs='+', default=STOCK_SYMBOLS, 
                        help='List of stock symbols to process')
    parser.add_argument('--functions', nargs='+', 
                        default=['TIME_SERIES_DAILY', 'OVERVIEW', 'GLOBAL_QUOTE'],
                        help='Alpha Vantage functions to use')
    parser.add_argument('--concurrent', action='store_true', 
                        help='Use concurrent processing')
    parser.add_argument('--force-refresh', action='store_true',
                        help='Force refresh data even if it exists in S3')
    
    args = parser.parse_args()
    
    logger.info("Starting Alpha Vantage extraction process")
    success_count = batch_process(args.symbols, args.functions, args.concurrent, args.force_refresh)
    
    logger.info(f"Extraction complete: {success_count} successful API calls")