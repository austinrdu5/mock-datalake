import os
import pytest
import requests
from datetime import datetime
from src.bronze.av_extract import (
    fetch_time_series_data,
    ALPHA_VANTAGE_API_KEY,
    BASE_URL
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

if __name__ == '__main__':
    pytest.main([__file__])
