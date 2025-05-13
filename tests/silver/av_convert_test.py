import pytest
import os
import boto3
from datetime import datetime
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from src.silver.av_convert import init_spark, process_stock_ticker, get_latest_files

def validate_test_credentials():
    """Validate that test credentials are properly configured"""
    required_vars = {
        'TEST_AWS_S3_BUCKET_NAME': os.getenv('TEST_AWS_S3_BUCKET_NAME'),
        'TEST_AWS_ACCESS_KEY': os.getenv('TEST_AWS_ACCESS_KEY'),
        'TEST_AWS_SECRET_KEY': os.getenv('TEST_AWS_SECRET_KEY')
    }
    
    missing_vars = [var for var, value in required_vars.items() if not value]
    if missing_vars:
        raise ValueError(
            f"Missing required test environment variables: {', '.join(missing_vars)}\n"
            "Please ensure these are set in your .env file or environment."
        )
    
    # Ensure test credentials are different from production
    prod_vars = {
        'AWS_S3_BUCKET_NAME': os.getenv('AWS_S3_BUCKET_NAME'),
        'AWS_ACCESS_KEY': os.getenv('AWS_ACCESS_KEY'),
        'AWS_SECRET_KEY': os.getenv('AWS_SECRET_KEY')
    }
    
    for test_var, prod_var in zip(required_vars.keys(), prod_vars.keys()):
        if required_vars[test_var] == prod_vars[prod_var]:
            raise ValueError(
                f"Test credentials ({test_var}) should be different from production credentials ({prod_var})\n"
                "Please ensure you're using separate credentials for testing."
            )

# Validate test credentials before running any tests
validate_test_credentials()

# Test configuration
TEST_BUCKET = os.getenv('TEST_AWS_S3_BUCKET_NAME')
TEST_ACCESS_KEY = os.getenv('TEST_AWS_ACCESS_KEY')
TEST_SECRET_KEY = os.getenv('TEST_AWS_SECRET_KEY')
TEST_REGION = os.getenv('TEST_AWS_DEFAULT_REGION', 'us-east-2')

# AWS config for test environment
TEST_AWS_CONFIG = {
    'access_key': TEST_ACCESS_KEY,
    'secret_key': TEST_SECRET_KEY,
    'bucket_name': TEST_BUCKET,
    'region': TEST_REGION
}

@pytest.fixture(scope="session")
def spark_session():
    """Create a Spark session with test configuration"""
    spark = init_spark(TEST_AWS_CONFIG)
    yield spark
    spark.stop()

@pytest.fixture(scope="session")
def s3_client():
    """Create an S3 client with test credentials"""
    return boto3.client(
        's3',
        aws_access_key_id=TEST_ACCESS_KEY,
        aws_secret_access_key=TEST_SECRET_KEY,
        region_name=TEST_REGION
    )

@pytest.fixture(scope="function")
def test_data():
    """Create sample test data for a stock ticker"""
    return {
        "timestamp": "2024-01-01",
        "open": "100.0",
        "high": "105.0",
        "low": "95.0",
        "close": "102.0",
        "volume": "1000"
    }

@pytest.fixture(scope="function")
def setup_test_data(s3_client, test_data):
    """Set up test data in S3 and clean up after"""
    test_ticker = "TEST"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create CSV content
    csv_content = "timestamp,open,high,low,close,volume\n"
    csv_content += f"{test_data['timestamp']},{test_data['open']},{test_data['high']},{test_data['low']},{test_data['close']},{test_data['volume']}"
    
    # Upload to bronze layer
    bronze_key = f"bronze/alphavantage/{test_ticker}/{timestamp}.csv"
    s3_client.put_object(
        Bucket=TEST_BUCKET,
        Key=bronze_key,
        Body=csv_content
    )
    
    yield test_ticker, f"s3a://{TEST_BUCKET}/{bronze_key}"
    
    # Cleanup
    try:
        # Clean up bronze layer
        s3_client.delete_object(
            Bucket=TEST_BUCKET,
            Key=bronze_key
        )
        # Clean up silver layer
        s3_client.delete_object(
            Bucket=TEST_BUCKET,
            Key=f"silver/stock_data/{test_ticker}"
        )
    except Exception as e:
        print(f"Cleanup failed: {e}")

def test_init_spark():
    """Test Spark session initialization with test credentials"""
    spark = init_spark(TEST_AWS_CONFIG)
    try:
        assert spark is not None
        assert spark.sparkContext.appName == "StockDataProcessor"
    finally:
        spark.stop()

def test_get_latest_files_single_ticker(s3_client, setup_test_data):
    """Test getting latest files with a single ticker"""
    test_ticker, expected_path = setup_test_data
    
    # Get latest files
    ticker_files = get_latest_files(TEST_AWS_CONFIG)
    
    # Verify results
    assert test_ticker in ticker_files
    assert ticker_files[test_ticker] == expected_path

def test_get_latest_files_multiple_tickers(s3_client, test_data):
    """Test getting latest files with multiple tickers"""
    # First, clean up any existing test data
    try:
        response = s3_client.list_objects_v2(
            Bucket=TEST_BUCKET,
            Prefix="bronze/alphavantage/"
        )
        for obj in response.get('Contents', []):
            s3_client.delete_object(
                Bucket=TEST_BUCKET,
                Key=obj['Key']
            )
    except Exception as e:
        print(f"Initial cleanup failed: {e}")
    
    tickers = ["AAPL", "MSFT", "GOOGL"]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create and upload test data for each ticker
    for ticker in tickers:
        csv_content = "timestamp,open,high,low,close,volume\n"
        csv_content += f"{test_data['timestamp']},{test_data['open']},{test_data['high']},{test_data['low']},{test_data['close']},{test_data['volume']}"
        
        bronze_key = f"bronze/alphavantage/{ticker}/{timestamp}.csv"
        s3_client.put_object(
            Bucket=TEST_BUCKET,
            Key=bronze_key,
            Body=csv_content
        )
    
    try:
        # Get latest files
        ticker_files = get_latest_files(TEST_AWS_CONFIG)
        
        # Verify results
        assert len(ticker_files) == len(tickers)
        for ticker in tickers:
            assert ticker in ticker_files
            assert ticker_files[ticker].endswith(f"{ticker}/{timestamp}.csv")
    finally:
        # Cleanup
        for ticker in tickers:
            try:
                s3_client.delete_object(
                    Bucket=TEST_BUCKET,
                    Key=f"bronze/alphavantage/{ticker}/{timestamp}.csv"
                )
            except Exception as e:
                print(f"Cleanup failed for {ticker}: {e}")

def test_get_latest_files_empty_bucket(s3_client):
    """Test getting latest files when bucket is empty"""
    # Ensure the test prefix is empty
    try:
        response = s3_client.list_objects_v2(
            Bucket=TEST_BUCKET,
            Prefix="bronze/alphavantage/"
        )
        for obj in response.get('Contents', []):
            s3_client.delete_object(
                Bucket=TEST_BUCKET,
                Key=obj['Key']
            )
    except Exception as e:
        print(f"Cleanup failed: {e}")
    
    # Get latest files
    ticker_files = get_latest_files(TEST_AWS_CONFIG)
    
    # Verify results
    assert len(ticker_files) == 0

def test_get_latest_files_invalid_credentials():
    """Test getting latest files with invalid credentials"""
    invalid_config = {
        'access_key': 'invalid',
        'secret_key': 'invalid',
        'bucket_name': TEST_BUCKET,
        'region': TEST_REGION
    }
    
    # Get latest files
    ticker_files = get_latest_files(invalid_config)
    
    # Verify results
    assert len(ticker_files) == 0

def test_process_stock_ticker(spark_session, setup_test_data):
    """Test processing a stock ticker's data"""
    test_ticker, file_path = setup_test_data
    
    # Process the data
    result_df = process_stock_ticker(spark_session, test_ticker, file_path)
    
    # Verify the result
    assert result_df is not None
    assert result_df.count() > 0
    
    # Check required columns
    required_columns = [
        "ticker", "trade_date", "open_price", "high_price", "low_price",
        "close_price", "volume", "processing_timestamp", "effective_from",
        "effective_to", "source_system", "source_file", "batch_id",
        "record_id", "completeness_score", "data_confidence"
    ]
    for col_name in required_columns:
        assert col_name in result_df.columns
    
    # Verify data values
    row = result_df.first()
    assert row["ticker"] == test_ticker
    assert row["open_price"] == 100.0
    assert row["high_price"] == 105.0
    assert row["low_price"] == 95.0
    assert row["close_price"] == 102.0
    assert row["volume"] == 1000
    assert row["completeness_score"] == 1.0
    assert row["data_confidence"] == 1.0

def test_process_stock_ticker_missing_data(spark_session):
    """Test processing a non-existent ticker"""
    test_ticker = "MISSING"
    non_existent_path = f"s3a://{TEST_BUCKET}/bronze/alphavantage/{test_ticker}/nonexistent.csv"
    
    # Process non-existent data
    result_df = process_stock_ticker(spark_session, test_ticker, non_existent_path)
    
    # Verify the result is None
    assert result_df is None

def test_process_stock_ticker_invalid_data(spark_session, s3_client):
    """Test processing a ticker with invalid data"""
    test_ticker = "INVALID"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create invalid CSV content
    invalid_csv = "timestamp,open,high,low,close,volume\ninvalid,invalid,invalid,invalid,invalid,invalid"
    
    # Upload to bronze layer
    bronze_key = f"bronze/alphavantage/{test_ticker}/{timestamp}.csv"
    s3_client.put_object(
        Bucket=TEST_BUCKET,
        Key=bronze_key,
        Body=invalid_csv
    )
    
    try:
        # Process the invalid data
        file_path = f"s3a://{TEST_BUCKET}/{bronze_key}"
        result_df = process_stock_ticker(spark_session, test_ticker, file_path)
        
        # Verify the result is None
        assert result_df is None
    finally:
        # Cleanup
        try:
            s3_client.delete_object(
                Bucket=TEST_BUCKET,
                Key=bronze_key
            )
        except Exception as e:
            print(f"Cleanup failed: {e}")