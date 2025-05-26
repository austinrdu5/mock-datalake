import os
import pytest
import requests
from datetime import datetime, timedelta
import boto3
from moto import mock_s3
import responses
from unittest.mock import MagicMock, patch
import pandas as pd
from src.bronze.av_extract import (
    fetch_time_series_data,
    BASE_URL,
    S3_PREFIX,
    save_to_s3,
    process_symbol,
    validate_time_series_data,
    check_data_exists_in_s3
)
import re

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
    'bucket': TEST_BUCKET,
    'region': TEST_REGION
}

# Global variable to store the full test data
_full_test_data = None

@pytest.mark.dependency(name="fetch_full_data")
def test_fetch_full_data():
    """Test that fetches full data once and validates it"""
    global _full_test_data
    print("\nFetching full test data...")
    
    # Make direct API call to verify demo key works
    response = requests.get(
        BASE_URL,
        params={
            'function': 'TIME_SERIES_DAILY',
            'symbol': 'IBM',
            'outputsize': 'full',
            'apikey': 'demo'
        }
    )
    assert response.status_code == 200, "API call failed"
    data = response.json()
    assert 'Time Series (Daily)' in data, "Response should contain time series data"
    
    # Now try with our function
    data = fetch_time_series_data('IBM', 'demo', outputsize='full')
    assert data is not None, "Failed to fetch full test data"
    assert 'csv_data' in data, "Response should contain csv_data"
    assert '_metadata' in data, "Response should contain metadata"
    
    # Store the data globally
    _full_test_data = data
    
    # Verify data format
    csv_lines = data['csv_data'].strip().split('\n')
    assert len(csv_lines) > 100, "Full data should have more than 100 data points"
    
    # Verify data spans a longer time period
    first_date = datetime.strptime(csv_lines[1].split(',')[0], '%Y-%m-%d')
    last_date = datetime.strptime(csv_lines[-1].split(',')[0], '%Y-%m-%d')
    days_difference = (first_date - last_date).days
    assert days_difference > 100, "Full data should span more than 100 days"
    
    assert data is not None, "Test data should be returned"

@pytest.fixture
def test_data_full():
    """Fixture that provides the full test data"""
    if _full_test_data is None:
        pytest.skip("Full test data not available - run test_fetch_full_data first")
    return _full_test_data

@pytest.fixture
def test_timestamp():
    """Fixture to provide a consistent timestamp for testing"""
    return datetime.now().strftime('%Y%m%d_%H%M%S')

def verify_data_format(data, expected_outputsize):
    """Helper function to verify data format and content"""
    assert data is not None, f"Should return data for valid symbol with {expected_outputsize} output"
    assert 'csv_data' in data, "Response should contain csv_data"
    assert '_metadata' in data, "Response should contain metadata"
    
    # Verify CSV format
    csv_lines = data['csv_data'].strip().split('\n')
    assert len(csv_lines) > 1, "CSV should have header and data rows"
    
    # Verify header format
    header = csv_lines[0]
    expected_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    for col in expected_columns:
        assert col in header.lower(), f"CSV header should contain {col}"
    
    # Verify metadata
    metadata = data['_metadata']
    assert metadata['symbol'] == 'IBM'
    assert metadata['function'] == 'TIME_SERIES_DAILY'
    assert metadata['raw_params']['outputsize'] == expected_outputsize
    assert 'extraction_time' in metadata
    
    # Verify data format
    data_row = csv_lines[1]
    columns = data_row.split(',')
    assert len(columns) == 6, "Data row should have 6 columns"
    
    # Verify data types
    assert len(columns[0].split('-')) == 3, "Timestamp should be in YYYY-MM-DD format"
    for col in columns[1:]:
        try:
            float(col)
        except ValueError:
            pytest.fail(f"Column {col} should be a valid number")
    
    return csv_lines

@pytest.mark.dependency(depends=["fetch_full_data"])
def test_fetch_time_series_data_full(test_data_full):
    """Test successful data fetch with full output size"""
    csv_lines = verify_data_format(test_data_full, 'full')
    assert len(csv_lines) > 100, "Full data should have more than 100 data points"

@pytest.mark.dependency(depends=["fetch_full_data"])
def test_fetch_time_series_data_compact(test_data_full):
    """Test successful data fetch with compact output size by using a subset of full data"""
    # Create a compact version by taking the first 100 lines
    compact_data = test_data_full.copy()
    csv_lines = compact_data['csv_data'].strip().split('\n')
    compact_data['csv_data'] = '\n'.join(csv_lines[:101])  # Header + 100 data points
    compact_data['_metadata']['raw_params']['outputsize'] = 'compact'
    
    csv_lines = verify_data_format(compact_data, 'compact')
    assert len(csv_lines) <= 101, "Compact data should have at most 100 data points plus header"

@pytest.mark.dependency(depends=["fetch_full_data"])
def test_fetch_time_series_data_invalid_symbol():
    """Test API response for invalid symbol"""
    data = fetch_time_series_data('INVALID_SYMBOL_123', 'demo')
    assert data is None, "Should return None for invalid symbol"

@pytest.mark.dependency(depends=["fetch_full_data"])
@patch('boto3.client')
def test_check_data_exists_in_s3(mock_boto3_client):
    """Test S3 data existence check functionality"""
    # Create mock S3 client
    mock_s3_client = MagicMock()
    mock_boto3_client.return_value = mock_s3_client
    
    # Test case 1: No data exists
    mock_s3_client.list_objects_v2.return_value = {}
    assert not check_data_exists_in_s3('TEST', 'TIME_SERIES_DAILY', TEST_AWS_CONFIG), "Should return False when no data exists"
    
    # Test case 2: Data exists but is old (2 days ago)
    old_date = (datetime.now() - timedelta(days=2)).strftime('%Y%m%d_%H%M%S')
    mock_s3_client.list_objects_v2.return_value = {
        'Contents': [{'Key': f'{S3_PREFIX}/TEST/{old_date}.csv'}]
    }
    assert not check_data_exists_in_s3('TEST', 'TIME_SERIES_DAILY', TEST_AWS_CONFIG), "Should return False for old data"
    
    # Test case 3: Recent data exists (12 hours ago)
    recent_date = (datetime.now() - timedelta(hours=12)).strftime('%Y%m%d_%H%M%S')
    mock_s3_client.list_objects_v2.return_value = {
        'Contents': [{'Key': f'{S3_PREFIX}/TEST/{recent_date}.csv'}]
    }
    assert check_data_exists_in_s3('TEST', 'TIME_SERIES_DAILY', TEST_AWS_CONFIG), "Should return True for recent data"

@pytest.mark.dependency(depends=["fetch_full_data"])
@mock_s3
@responses.activate
def test_process_symbol_with_existing_data(test_timestamp):
    """Test process_symbol behavior with existing data"""
    # Create mock S3 client and bucket
    mock_s3_client = boto3.client('s3')
    mock_s3_client.create_bucket(
        Bucket=TEST_BUCKET,
        CreateBucketConfiguration={'LocationConstraint': TEST_REGION}
    )
    
    # Add existing file to mock S3
    recent_timestamp = datetime.now() - timedelta(hours=12)
    formatted_timestamp = recent_timestamp.strftime('%Y%m%d_%H%M%S')
    mock_s3_client.put_object(
        Bucket=TEST_BUCKET,
        Key=f'{S3_PREFIX}/TEST/{formatted_timestamp}.csv',
        Body='test data'
    )
    
    # Mock Alpha Vantage API response for force refresh
    mock_csv_data = "timestamp,open,high,low,close,volume\n2024-01-01,100.0,101.0,99.0,100.5,1000"
    responses.add(
        responses.GET,
        BASE_URL,
        body=mock_csv_data,
        status=200,
        content_type='text/csv'
    )
    
    # Test normal processing (should skip API call)
    assert process_symbol('TEST', 'TIME_SERIES_DAILY', test_timestamp, TEST_AWS_CONFIG, 'demo'), "Should return True when data exists"
    
    # Test force refresh
    assert process_symbol('TEST', 'TIME_SERIES_DAILY', test_timestamp, TEST_AWS_CONFIG, 'demo', force_refresh=True), "Should return True with force refresh"

@pytest.mark.dependency(depends=["fetch_full_data"])
def test_real_s3_integration(test_timestamp, test_data_full):
    """Integration test using real S3 bucket and Alpha Vantage demo key"""
    # Format timestamp to match new naming scheme
    formatted_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save to S3
    assert save_to_s3(test_data_full, 'IBM', 'TIME_SERIES_DAILY', formatted_timestamp, TEST_AWS_CONFIG), "Failed to save data to S3"
    
    # Check if data exists in S3
    assert check_data_exists_in_s3('IBM', 'TIME_SERIES_DAILY', TEST_AWS_CONFIG), "Data should exist in S3"
    
    # Check with force refresh
    assert not check_data_exists_in_s3('IBM', 'TIME_SERIES_DAILY', TEST_AWS_CONFIG, days_threshold=0), "Data should be considered old with 0 day threshold"
    
    # Clean up - delete the test file
    s3_client = boto3.client(
        's3',
        aws_access_key_id=TEST_ACCESS_KEY,
        aws_secret_access_key=TEST_SECRET_KEY,
        region_name=TEST_REGION
    )
    try:
        # List objects to find the file we just created
        response = s3_client.list_objects_v2(
            Bucket=TEST_BUCKET,
            Prefix=f"{S3_PREFIX}/IBM/"
        )
        if 'Contents' in response:
            for obj in response['Contents']:
                s3_client.delete_object(
                    Bucket=TEST_BUCKET,
                    Key=obj['Key']
                )
    except Exception as e:
        print(f"Warning: Failed to clean up S3: {str(e)}")

@pytest.mark.dependency(depends=["fetch_full_data"])
def test_validation_with_demo_key(test_data_full):
    """Test that data from Alpha Vantage demo key passes Pandera validation"""
    # Validate the data
    validated_df = validate_time_series_data(test_data_full['csv_data'], 'TIME_SERIES_DAILY')
    assert validated_df is not None, "Data validation should pass"
    
    # Verify DataFrame structure
    assert 'timestamp' in validated_df.columns, "DataFrame should have timestamp column"
    assert 'open' in validated_df.columns, "DataFrame should have open column"
    assert 'high' in validated_df.columns, "DataFrame should have high column"
    assert 'low' in validated_df.columns, "DataFrame should have low column"
    assert 'close' in validated_df.columns, "DataFrame should have close column"
    assert 'volume' in validated_df.columns, "DataFrame should have volume column"
    
    # Verify data types
    assert pd.api.types.is_datetime64_any_dtype(validated_df['timestamp']), "timestamp should be datetime"
    assert pd.api.types.is_float_dtype(validated_df['open']), "open should be float"
    assert pd.api.types.is_float_dtype(validated_df['high']), "high should be float"
    assert pd.api.types.is_float_dtype(validated_df['low']), "low should be float"
    assert pd.api.types.is_float_dtype(validated_df['close']), "close should be float"
    assert pd.api.types.is_integer_dtype(validated_df['volume']), "volume should be integer"
    
    # Verify value constraints
    assert (validated_df['open'] >= 0).all(), "open prices should be non-negative"
    assert (validated_df['high'] >= 0).all(), "high prices should be non-negative"
    assert (validated_df['low'] >= 0).all(), "low prices should be non-negative"
    assert (validated_df['close'] >= 0).all(), "close prices should be non-negative"
    assert (validated_df['volume'] >= 0).all(), "volume should be non-negative"
    
    # Verify price relationships
    assert (validated_df['high'] >= validated_df['low']).all(), "high should be >= low"
    assert (validated_df['high'] >= validated_df['open']).all(), "high should be >= open"
    assert (validated_df['high'] >= validated_df['close']).all(), "high should be >= close"
    assert (validated_df['low'] <= validated_df['open']).all(), "low should be <= open"
    assert (validated_df['low'] <= validated_df['close']).all(), "low should be <= close"
    
    # Verify timestamps are in order
    assert validated_df['timestamp'].is_monotonic_decreasing, "timestamps should be in descending order"
    
    # Verify we have the expected number of rows for full data
    assert len(validated_df) > 100, "full data should have more than 100 rows"

if __name__ == '__main__':
    pytest.main([__file__])
