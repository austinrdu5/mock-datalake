import os
import requests
import json
import logging
import time
import boto3
import pandas as pd
from io import StringIO
import pandera.pandas as pa
from pandera.typing import Series
from pandera import DataFrameSchema
from pandera.errors import SchemaError
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
BASE_URL = 'https://www.alphavantage.co/query'

# S3 prefix for data storage
S3_PREFIX = 'bronze/alphavantage'

# Stock symbols to fetch (you can expand this list)
STOCK_SYMBOLS: List[str] = ['IBM']  # Demo key only works with IBM

# Define Pandera schema for Alpha Vantage time series data
AlphaVantageSchema = pa.DataFrameSchema({
    "timestamp": pa.Column(datetime, description="Date of the data point"),
    "open": pa.Column(float, checks=pa.Check.greater_than_or_equal_to(0), description="Opening price"),
    "high": pa.Column(float, checks=pa.Check.greater_than_or_equal_to(0), description="Highest price"),
    "low": pa.Column(float, checks=pa.Check.greater_than_or_equal_to(0), description="Lowest price"),
    "close": pa.Column(float, checks=pa.Check.greater_than_or_equal_to(0), description="Closing price"),
    "volume": pa.Column(int, checks=pa.Check.greater_than_or_equal_to(0), description="Trading volume")
}, strict=True)

def validate_time_series_data(data: str, function: str) -> Optional[pd.DataFrame]:
    """
    Validate time series data using Pandera schema
    
    Args:
        data (str): CSV data string
        function (str): Alpha Vantage function used
        
    Returns:
        Optional[pd.DataFrame]: Validated DataFrame if validation passes, None otherwise
    """
    try:
        # Convert CSV string to DataFrame
        df = pd.read_csv(StringIO(data))
        
        # Convert timestamp column to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Validate DataFrame
        AlphaVantageSchema.validate(df)
        logger.info(f"Data validation successful for {function}")
        return df
        
    except SchemaError as e:
        logger.error(f"Data validation failed: {str(e)}")
        return None
    except Exception as e:
        logger.error(f"Error during data validation: {str(e)}")
        return None

# Alpha Vantage API Functions
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=10, min=10, max=60),
    retry=retry_if_exception_type((requests.exceptions.RequestException, KeyError)),
    before_sleep=lambda retry_state: logger.warning(f"Retrying in {retry_state.next_action.sleep} seconds...")  # type: ignore
)
def fetch_time_series_data(symbol: str, api_key: str, function: str = 'TIME_SERIES_DAILY', outputsize: str = 'full') -> Optional[Dict[str, Any]]:
    """
    Fetch time series data for a given stock symbol in CSV format
    
    Args:
        symbol (str): Stock symbol (e.g., 'AAPL' for Apple)
        api_key (str): Alpha Vantage API key
        function (str): Alpha Vantage API function to use (default: TIME_SERIES_DAILY)
        outputsize (str): 'compact' for latest 100 data points, 'full' for up to 20 years
        
    Returns:
        Optional[Dict[str, Any]]: Dictionary containing CSV data and metadata
    """
    params = {
        'function': function,
        'symbol': symbol,
        'outputsize': outputsize,
        'apikey': api_key
    }
    
    # Add timestamp for data lineage
    timestamp = datetime.now().isoformat()
    
    try:
        logger.info(f"Fetching {function} data for {symbol}")
        response = requests.get(BASE_URL, params=params)
        
        if response.status_code == 200:
            # First try to parse as JSON (for demo key)
            try:
                json_data = response.json()
                if 'Time Series (Daily)' in json_data:
                    # Convert JSON to CSV format
                    time_series = json_data['Time Series (Daily)']
                    csv_lines = ['timestamp,open,high,low,close,volume']
                    for date, values in time_series.items():
                        row = [
                            date,
                            values['1. open'],
                            values['2. high'],
                            values['3. low'],
                            values['4. close'],
                            values['5. volume']
                        ]
                        csv_lines.append(','.join(row))
                    content = '\n'.join(csv_lines)
                else:
                    logger.error("No time series data found in JSON response")
                    return None
            except ValueError:
                # If not JSON, try as CSV
                content = response.text
                logger.info(f"Response content: {content[:200]}...")
                
                # Check if response is an error message
                if any(error_text in content for error_text in ['Error Message', 'Invalid API call', 'Invalid API Key']):
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
            expected_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
                
            if not all(col in header.replace(' ', '_') for col in expected_columns):
                logger.error(f"Invalid CSV header format: {header}")
                return None
            
            # Validate data using Pandera
            validated_df = validate_time_series_data(content, function)
            if validated_df is None:
                return None
            
            # Create response dictionary with metadata
            data = {
                'csv_data': content,
                '_metadata': {
                    'source': 'alpha_vantage',
                    'function': function,
                    'symbol': symbol,
                    'extraction_time': timestamp,
                    'raw_params': params,
                    'validation_status': 'passed',
                    'row_count': len(validated_df)
                }
            }
            
            return data
        else:
            logger.error(f"Failed to fetch data: HTTP {response.status_code}")
            return None
            
    except Exception as e:
        logger.error(f"Exception during API call: {str(e)}")
        return None

def save_to_s3(data: Dict[str, Any], symbol: str, function: str, batch_timestamp: str, aws_config: Dict[str, str]) -> bool:
    """
    Save data to S3 bucket with appropriate partitioning
    
    Args:
        data (Dict[str, Any]): Data to save (contains csv_data and metadata)
        symbol (str): Stock symbol
        function (str): Alpha Vantage function used
        batch_timestamp (str): Timestamp of the batch run
        aws_config (Dict[str, str]): AWS configuration dictionary
    
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
        s3_client = boto3.client(
            's3',
            aws_access_key_id=aws_config['access_key'],
            aws_secret_access_key=aws_config['secret_key'],
            region_name=aws_config.get('region', 'us-east-1')
        )
        s3_client.put_object(
            Bucket=aws_config['bucket'],
            Key=s3_key,
            Body=data['csv_data'],
            ContentType='text/csv'
        )
        return True
    except Exception as e:
        logger.error(f"Failed to save to S3: {str(e)}")
        return False
    
def check_data_exists_in_s3(symbol: str, function: str, aws_config: Dict[str, str], days_threshold: int = 1) -> bool:
    """
    Check if data for a symbol exists in S3 and is recent enough
    
    Args:
        symbol (str): Stock symbol
        function (str): Alpha Vantage function used
        aws_config (Dict[str, str]): AWS configuration dictionary
        days_threshold (int): Number of days to consider data fresh
        
    Returns:
        bool: True if recent data exists, False otherwise
    """
    try:
        s3_client = boto3.client(
            's3',
            aws_access_key_id=aws_config['access_key'],
            aws_secret_access_key=aws_config['secret_key'],
            region_name=aws_config.get('region', 'us-east-1')
        )
        # List objects in the symbol's directory
        prefix = f"{S3_PREFIX}/{symbol}/"
        response = s3_client.list_objects_v2(
            Bucket=aws_config['bucket'],
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

def process_symbol(symbol: str, function: str, batch_timestamp: str, aws_config: Dict[str, str], api_key: str, force_refresh: bool = False) -> bool:
    """
    Process a single symbol
    
    Args:
        symbol (str): Stock symbol
        function (str): Alpha Vantage function to use
        batch_timestamp (str): Timestamp of the batch run
        aws_config (Dict[str, str]): AWS configuration dictionary
        api_key (str): Alpha Vantage API key
        force_refresh (bool): Force refresh data even if it exists in S3
        
    Returns:
        bool: Success status
    """
    # Check if data exists and is recent enough
    if not force_refresh and check_data_exists_in_s3(symbol, function, aws_config):
        logger.info(f"Recent data for {symbol} already exists in S3, skipping API call")
        return True
        
    data = fetch_time_series_data(symbol, api_key, function)
    if data:
        return save_to_s3(data, symbol, function, batch_timestamp, aws_config)
    return False

def batch_process(
    symbols: List[str] = STOCK_SYMBOLS,
    functions: Optional[List[str]] = None,
    aws_config: Optional[Dict[str, str]] = None,
    api_key: Optional[str] = None,
    concurrent: bool = False,
    force_refresh: bool = False
) -> int:
    """
    Batch process multiple symbols with rate limiting
    
    Args:
        symbols (List[str]): List of stock symbols to process
        functions (Optional[List[str]]): List of Alpha Vantage functions to use. Defaults to ['TIME_SERIES_DAILY']
        aws_config (Optional[Dict[str, str]]): AWS configuration dictionary
        api_key (Optional[str]): Alpha Vantage API key
        concurrent (bool): Whether to use concurrent processing
        force_refresh (bool): Force refresh data even if it exists in S3
        
    Returns:
        int: Number of successful operations
    """
    if functions is None:
        functions = ['TIME_SERIES_DAILY']  # Default to just daily time series data
    
    if aws_config is None:
        raise ValueError("AWS configuration is required")
    
    if api_key is None:
        raise ValueError("Alpha Vantage API key is required")
    
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
                future = executor.submit(process_symbol, symbol, function, batch_timestamp, aws_config, api_key, force_refresh)
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
            
            if process_symbol(symbol, function, batch_timestamp, aws_config, api_key, force_refresh):
                success_count += 1
    
    logger.info(f"Batch at {batch_timestamp} completed: {success_count}/{len(tasks)} successful")
    return success_count

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Alpha Vantage Data Extractor')
    parser.add_argument('--symbols', nargs='+', default=STOCK_SYMBOLS, 
                        help='List of stock symbols to process')
    parser.add_argument('--functions', nargs='+', 
                        default=['TIME_SERIES_DAILY'],
                        help='Alpha Vantage functions to use')
    parser.add_argument('--concurrent', action='store_true', 
                        help='Use concurrent processing')
    parser.add_argument('--force-refresh', action='store_true',
                        help='Force refresh data even if it exists in S3')
    
    args = parser.parse_args()
    
    # Get AWS configuration
    aws_config = {
        'access_key': os.getenv('AWS_ACCESS_KEY'),
        'secret_key': os.getenv('AWS_SECRET_KEY'),
        'bucket': os.getenv('AWS_S3_BUCKET_NAME'),
        'region': os.getenv('AWS_REGION', 'us-east-1')
    }
    
    # Get Alpha Vantage API key
    api_key = os.getenv('ALPHA_VANTAGE_API_KEY')
    
    # Validate configurations
    if not aws_config['access_key'] or not aws_config['secret_key']:
        logger.error("AWS credentials (AWS_ACCESS_KEY and AWS_SECRET_KEY) must be set")
        exit(1)
    if not aws_config['bucket']:
        logger.error("AWS_S3_BUCKET_NAME environment variable is not set")
        exit(1)
    if not api_key:
        logger.error("ALPHA_VANTAGE_API_KEY environment variable is not set")
        exit(1)
    
    logger.info("Starting Alpha Vantage extraction process")
    success_count = batch_process(args.symbols, args.functions, aws_config, api_key, args.concurrent, args.force_refresh)
    
    logger.info(f"Extraction complete: {success_count} successful API calls")