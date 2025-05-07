import os
import requests
import json
import logging
import time
import boto3
from datetime import datetime
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

# Add console handler to see logs during testing
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# Alpha Vantage API configuration
ALPHA_VANTAGE_API_KEY = os.environ.get('ALPHA_VANTAGE_API_KEY')
if not ALPHA_VANTAGE_API_KEY:
    raise ValueError("ALPHA_VANTAGE_API_KEY environment variable not set")
BASE_URL = 'https://www.alphavantage.co/query'

# AWS S3 configuration
AWS_ACCESS_KEY = os.environ.get('AWS_ACCESS_KEY')
AWS_SECRET_KEY = os.environ.get('AWS_SECRET_KEY')
S3_BUCKET_NAME = 'mock-datalake1'  # Hardcoded bucket name
S3_PREFIX = 'bronze/alphavantage'  # Updated prefix

# Create S3 client
s3_client = boto3.client(
    's3',
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY
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

def save_to_s3(data: Dict[str, Any], symbol: str, function: str, batch_id: Optional[str] = None) -> bool:
    """
    Save data to S3 bucket with appropriate partitioning
    
    Args:
        data (Dict[str, Any]): Data to save (contains csv_data and metadata)
        symbol (str): Stock symbol
        function (str): Alpha Vantage function used
        batch_id (Optional[str]): Optional batch identifier
    
    Returns:
        bool: Success status
    """
    if not data or 'csv_data' not in data:
        return False
        
    # Create filename with timestamp for temporal integration
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    batch_suffix = f"_batch{batch_id}" if batch_id else ""
    filename = f"{timestamp}{batch_suffix}.csv"
    
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
    
def process_symbol(symbol: str, function: str, batch_id: Optional[str] = None) -> bool:
    """Process a single symbol"""
    data = fetch_time_series_data(symbol, function)
    if data:
        return save_to_s3(data, symbol, function, batch_id)
    return False

def batch_process(
    symbols: List[str] = STOCK_SYMBOLS,
    functions: Optional[List[str]] = None,
    concurrent: bool = False
) -> int:
    """
    Batch process multiple symbols with rate limiting
    
    Args:
        symbols (List[str]): List of stock symbols to process
        functions (Optional[List[str]]): List of Alpha Vantage functions to use
        concurrent (bool): Whether to use concurrent processing
        
    Returns:
        int: Number of successful operations
    """
    if functions is None:
        functions = ['TIME_SERIES_DAILY', 'OVERVIEW', 'GLOBAL_QUOTE']
    
    batch_id = datetime.now().strftime('%Y%m%d%H%M%S')
    logger.info(f"Starting batch process {batch_id} for {len(symbols)} symbols")
    
    success_count = 0
    
    # Generate all combinations of symbols and functions
    tasks = [(symbol, function) for symbol in symbols for function in functions]
    
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
                future = executor.submit(process_symbol, symbol, function, batch_id)
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
            
            if process_symbol(symbol, function, batch_id):
                success_count += 1
    
    logger.info(f"Batch {batch_id} completed: {success_count}/{len(tasks)} successful")
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
    
    args = parser.parse_args()
    
    logger.info("Starting Alpha Vantage extraction process")
    success_count = batch_process(args.symbols, args.functions, args.concurrent)
    
    logger.info(f"Extraction complete: {success_count} successful API calls")