import os
import pytest
import requests
from datetime import datetime, timedelta
import boto3
from moto import mock_s3
import responses
from unittest.mock import MagicMock, patch
from src.bronze.av_extract import (
    fetch_time_series_data,
    ALPHA_VANTAGE_API_KEY,
    BASE_URL,
    check_data_exists_in_s3,
    S3_BUCKET_NAME,
    S3_PREFIX,
    save_to_s3,
    process_symbol
)

# Set up test environment
os.environ['ALPHA_VANTAGE_API_KEY'] = 'demo'  # Using Alpha Vantage's demo API key

@pytest.fixture(scope="module")
def test_data_compact():
    """Fixture to fetch compact data once for all tests"""
    print("\nFetching compact test data...")
    data = fetch_time_series_data('IBM', outputsize='compact')
    assert data is not None, "Failed to fetch compact test data"
    return data

@pytest.fixture(scope="module")
def test_data_full():
    """Fixture to fetch full data once for all tests"""
    print("\nFetching full test data...")
    data = fetch_time_series_data('IBM', outputsize='full')
    assert data is not None, "Failed to fetch full test data"
    return data

@pytest.fixture
def test_timestamp():
    """Fixture to provide a consistent timestamp for testing"""
    return datetime.now().strftime('%Y%m%d_%H%M%S')

def test_api_key_validation():
    """Test that API key is properly set"""
    assert ALPHA_VANTAGE_API_KEY is not None, "API key should not be None"
    assert len(ALPHA_VANTAGE_API_KEY) > 0, "API key should not be empty"
    assert ALPHA_VANTAGE_API_KEY != 'YOUR_API_KEY_HERE', "API key should be properly set"

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

def test_fetch_time_series_data_compact(test_data_compact):
    """Test successful data fetch with compact output size"""
    csv_lines = verify_data_format(test_data_compact, 'compact')
    assert len(csv_lines) <= 101, "Compact data should have at most 100 data points plus header"

def test_fetch_time_series_data_full(test_data_full):
    """Test successful data fetch with full output size"""
    csv_lines = verify_data_format(test_data_full, 'full')
    assert len(csv_lines) > 100, "Full data should have more than 100 data points"
    
    # Verify data spans a longer time period
    first_date = datetime.strptime(csv_lines[1].split(',')[0], '%Y-%m-%d')
    last_date = datetime.strptime(csv_lines[-1].split(',')[0], '%Y-%m-%d')
    days_difference = (first_date - last_date).days
    assert days_difference > 100, "Full data should span more than 100 days"

def test_fetch_time_series_data_invalid_symbol():
    """Test API response for invalid symbol"""
    data = fetch_time_series_data('INVALID_SYMBOL_123')
    assert data is None, "Should return None for invalid symbol"

def test_check_data_exists_in_s3():
    """Test S3 data existence check functionality"""
    # Create mock S3 client
    mock_s3_client = MagicMock()
    
    # Test case 1: No data exists
    mock_s3_client.list_objects_v2.return_value = {}
    assert not check_data_exists_in_s3('TEST', 'TIME_SERIES_DAILY', s3_client_override=mock_s3_client), "Should return False when no data exists"
    
    # Test case 2: Data exists but is old
    old_timestamp = datetime.now() - timedelta(days=2)
    mock_s3_client.list_objects_v2.return_value = {
        'Contents': [{'Key': f'{S3_PREFIX}/TEST/20240101_000000.csv', 'LastModified': old_timestamp}]
    }
    assert not check_data_exists_in_s3('TEST', 'TIME_SERIES_DAILY', s3_client_override=mock_s3_client), "Should return False for old data"
    
    # Test case 3: Recent data exists
    recent_timestamp = datetime.now() - timedelta(hours=12)
    mock_s3_client.list_objects_v2.return_value = {
        'Contents': [{'Key': f'{S3_PREFIX}/TEST/20240315_123456.csv', 'LastModified': recent_timestamp}]
    }
    assert check_data_exists_in_s3('TEST', 'TIME_SERIES_DAILY', s3_client_override=mock_s3_client), "Should return True for recent data"

@mock_s3
@responses.activate
def test_process_symbol_with_existing_data(test_timestamp):
    """Test process_symbol behavior with existing data"""
    # Create mock S3 client and bucket
    mock_s3_client = boto3.client('s3')
    mock_s3_client.create_bucket(Bucket=S3_BUCKET_NAME)
    
    # Add existing file to mock S3
    recent_timestamp = datetime.now() - timedelta(hours=12)
    mock_s3_client.put_object(
        Bucket=S3_BUCKET_NAME,
        Key=f'{S3_PREFIX}/TEST/{test_timestamp}.csv',
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
    assert process_symbol('TEST', 'TIME_SERIES_DAILY', test_timestamp, s3_client_override=mock_s3_client), "Should return True when data exists"
    
    # Test force refresh
    assert process_symbol('TEST', 'TIME_SERIES_DAILY', test_timestamp, force_refresh=True, s3_client_override=mock_s3_client), "Should return True with force refresh"

@pytest.mark.integration
def test_real_s3_integration(test_timestamp):
    """Integration test using real S3 bucket and Alpha Vantage demo key"""
    # Set up demo API key
    os.environ['ALPHA_VANTAGE_API_KEY'] = 'demo'
    
    # Fetch IBM data using demo key
    data = fetch_time_series_data('IBM', outputsize='compact')
    assert data is not None, "Failed to fetch IBM data with demo key"
    assert 'csv_data' in data, "Response should contain csv_data"
    
    # Save to S3
    assert save_to_s3(data, 'IBM', 'TIME_SERIES_DAILY', test_timestamp), "Failed to save data to S3"
    
    # Check if data exists in S3
    assert check_data_exists_in_s3('IBM', 'TIME_SERIES_DAILY'), "Data should exist in S3"
    
    # Check with force refresh
    assert check_data_exists_in_s3('IBM', 'TIME_SERIES_DAILY', days_threshold=0), "Data should be considered old with 0 day threshold"
    
    # Clean up - delete the test file
    s3_client = boto3.client('s3')
    try:
        # List objects to find the file we just created
        response = s3_client.list_objects_v2(
            Bucket=S3_BUCKET_NAME,
            Prefix=f"{S3_PREFIX}/IBM/"
        )
        if 'Contents' in response:
            for obj in response['Contents']:
                s3_client.delete_object(
                    Bucket=S3_BUCKET_NAME,
                    Key=obj['Key']
                )
    except Exception as e:
        print(f"Warning: Failed to clean up S3: {str(e)}")

if __name__ == '__main__':
    pytest.main([__file__])
