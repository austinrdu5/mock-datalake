import pytest
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit
import boto3
import os
import tempfile
import shutil
from datetime import datetime
import uuid
import logging
from unittest.mock import patch, MagicMock

from src.silver.ecommerce_convert import (
    init_spark_with_delta, 
    parse_category_hierarchy,
    calculate_confidence_score_ecommerce,
    transform_ecommerce_bronze_to_silver,   
    write_to_delta_lake,
    validate_silver_data,
    EcommerceSilverSchema
)

# Set up logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestEcommerceConverter:
    """Test suite for ecommerce Delta Lake converter"""
    
    @classmethod
    def setup_class(cls):
        """Set up test environment once for all tests"""
        cls.aws_config = {
            'access_key': os.environ.get('TEST_AWS_ACCESS_KEY', ''),
            'secret_key': os.environ.get('TEST_AWS_SECRET_KEY', ''),
            'bucket_name': os.environ.get('TEST_AWS_S3_BUCKET_NAME', ''),
            'region': os.environ.get('TEST_AWS_DEFAULT_REGION', 'us-east-1')
        }
        
        # Validate test credentials
        missing_creds = [k for k, v in cls.aws_config.items() if not v]
        if missing_creds:
            pytest.skip(f"Missing test AWS credentials: {missing_creds}")
        
        # Initialize Spark session for tests
        cls.spark = init_spark_with_delta(cls.aws_config)
        
        # Set up S3 client for test data management
        cls.s3_client = boto3.client(
            's3',
            aws_access_key_id=cls.aws_config['access_key'],
            aws_secret_access_key=cls.aws_config['secret_key'],
            region_name=cls.aws_config['region']
        )
        
        # Test data paths
        cls.test_bronze_path = f"s3a://{cls.aws_config['bucket_name']}/test/bronze/ecommerce/events.csv"
        cls.test_silver_path = f"s3a://{cls.aws_config['bucket_name']}/test/silver/ecommerce/"
        
    @classmethod
    def teardown_class(cls):
        """Clean up test environment"""
        cls.cleanup_test_data()
        if hasattr(cls, 'spark'):
            cls.spark.stop()
    
    @classmethod
    def cleanup_test_data(cls):
        """Clean up test data from S3"""
        try:
            # List and delete test objects
            response = cls.s3_client.list_objects_v2(
                Bucket=cls.aws_config['bucket_name'],
                Prefix='test/'
            )
            
            if 'Contents' in response:
                objects_to_delete = [{'Key': obj['Key']} for obj in response['Contents']]
                cls.s3_client.delete_objects(
                    Bucket=cls.aws_config['bucket_name'],
                    Delete={'Objects': objects_to_delete}
                )
                logger.info(f"Cleaned up {len(objects_to_delete)} test objects")
        except Exception as e:
            logger.warning(f"Error cleaning up test data: {e}")
    
    def create_test_csv_data(self):
        """Create sample ecommerce CSV data for testing"""
        test_data = [
            "event_time,event_type,product_id,category_id,category_code,brand,price,user_id,user_session",
            "2020-09-24 11:57:06 UTC,view,1996170,2144415922528452715,electronics.telephone.wireless,apple,31.90,1515915625519388267,LJuJVLEjPT",
            "2020-09-24 11:57:26 UTC,cart,139905,2144415926932472027,computers.components.cooler,zalman,17.16,1515915625519380411,tdicluNnRY", 
            "2020-09-24 11:57:27 UTC,purchase,215454,2144415927158964449,,,9.81,1515915625513238515,4TMArHtXQy",  # Missing category and brand
            "2020-09-24 11:57:33 UTC,view,635807,2144415923107266682,computers.peripherals.printer,,113.81,1515915625519014356,aGFYrNgC08",  # Missing brand
            "2020-10-01 12:00:00 UTC,view,3658723,2144415921169498184,home.kitchen,cameronsino,15.87,1515915625510743344,aa4mmk0kwQ"  # Different month
        ]
        return "\n".join(test_data)
    
    def upload_test_csv(self):
        """Upload test CSV data to S3"""
        csv_content = self.create_test_csv_data()
        
        self.s3_client.put_object(
            Bucket=self.aws_config['bucket_name'],
            Key='test/bronze/ecommerce/events.csv',
            Body=csv_content.encode('utf-8')
        )
        logger.info("Uploaded test CSV data to S3")
    
    def test_spark_delta_initialization(self):
        """Test that Spark session initializes with Delta Lake support"""
        assert self.spark is not None
        packages = self.spark.sparkContext.getConf().get("spark.jars.packages")
        assert packages is not None, "spark.jars.packages should be configured"
        assert "delta" in packages, "Delta Lake package should be configured"
        logger.info("✅ Spark with Delta Lake initialized successfully")
    
    def test_category_hierarchy_parsing(self):
        """Test category code parsing into hierarchy levels"""
        # Create test DataFrame
        test_data = [
            ("electronics.telephone.wireless",),
            ("computers.desktop",),
            ("home",),
            ("",),
            (None,)
        ]
        
        df = self.spark.createDataFrame(test_data, ["category_code"])
        
        # Apply parsing function
        level_1, level_2, level_3 = parse_category_hierarchy(col("category_code"))
        result_df = df.select(
            col("category_code"),
            level_1.alias("level_1"),
            level_2.alias("level_2"), 
            level_3.alias("level_3")
        ).collect()
        
        # Verify parsing results
        expected = [
            ("electronics.telephone.wireless", "electronics", "telephone", "wireless"),
            ("computers.desktop", "computers", "desktop", None),
            ("home", "home", None, None),
            ("", None, None, None),
            (None, None, None, None)
        ]
        
        for i, (orig, exp_l1, exp_l2, exp_l3) in enumerate(expected):
            row = result_df[i]
            assert row.level_1 == exp_l1, f"Level 1 mismatch for {orig}: expected {exp_l1}, got {row.level_1}"
            assert row.level_2 == exp_l2, f"Level 2 mismatch for {orig}: expected {exp_l2}, got {row.level_2}"
            assert row.level_3 == exp_l3, f"Level 3 mismatch for {orig}: expected {exp_l3}, got {row.level_3}"
        
        logger.info("✅ Category hierarchy parsing works correctly")
    
    def test_confidence_score_calculation(self):
        """Test confidence score calculation logic"""
        # Create test data with different combinations
        test_data = [
            ("electronics", "apple"),      # Both present: 1.0
            ("electronics", ""),           # Missing brand: 0.6  
            ("electronics", None),         # Missing brand: 0.6
            ("", "apple"),                 # Missing category: 0.4
            (None, "apple"),               # Missing category: 0.4
            ("", ""),                      # Missing both: 0.0
            (None, None)                   # Missing both: 0.0
        ]
        
        test_cases = [
            "Both category and brand present",
            "Category present, empty brand",
            "Category present, null brand",
            "Empty category, brand present",
            "Null category, brand present",
            "Empty category and brand",
            "Null category and brand"
        ]
        
        df = self.spark.createDataFrame(test_data, ["category_level_1", "brand"])
        
        # Apply confidence score calculation
        result_df = calculate_confidence_score_ecommerce(df)
        scores = [row.data_confidence_score for row in result_df.collect()]
        
        expected_scores = [1.0, 0.6, 0.6, 0.4, 0.4, 0.0, 0.0]
        
        for i, (expected, actual, case) in enumerate(zip(expected_scores, scores, test_cases)):
            assert abs(actual - expected) < 0.01, f"Score mismatch for case '{case}': expected {expected}, got {actual}"
        
        logger.info("✅ Confidence score calculation works correctly")
    
    def test_schema_validation(self):
        """Test Pandera schema validation"""
        # Create valid test data
        valid_data = pd.DataFrame({
            'event_time': [pd.Timestamp('2020-09-24 11:57:06')],
            'event_type': ['view'],
            'product_id': [1996170],
            'category_id': [2144415922528452715.0],
            'category_level_1': ['electronics'],
            'category_level_2': ['telephone'],
            'category_level_3': ['wireless'],
            'brand': ['apple'],
            'price': [31.90],
            'user_id': [1515915625519388267.0],
            'user_session': ['LJuJVLEjPT'],
            'year_month': ['2020-09'],
            'processing_timestamp': [pd.Timestamp.now()],
            'effective_from': [pd.Timestamp('2020-09-24 11:57:06')],
            'effective_to': [None],
            'data_confidence_score': [1.0],
            'source_system': ['EcommerceEvents'],
            'source_file': ['test.csv'],
            'batch_id': [str(uuid.uuid4())],
            'record_id': [str(uuid.uuid4())]
        })
        
        # Should pass validation
        assert validate_silver_data(valid_data) == True
        
        # Test invalid data (invalid event_type)
        invalid_data = valid_data.copy()
        invalid_data['event_type'] = ['invalid_event']
        
        assert validate_silver_data(invalid_data) == False
        
        logger.info("✅ Schema validation works correctly")
    
    def test_end_to_end_transformation(self):
        """Test complete transformation pipeline"""
        # Upload test data
        self.upload_test_csv()
        
        try:
            # Run transformation
            silver_df = transform_ecommerce_bronze_to_silver(
                self.spark, 
                self.test_bronze_path, 
                self.test_silver_path
            )
            
            assert silver_df is not None, "Transformation should not return None"
            
            # Verify record count
            record_count = silver_df.count()
            assert record_count == 5, f"Expected 5 records, got {record_count}"
            
            # Verify required columns exist
            expected_columns = [
                # Event identification
                'event_time', 'event_type',
                
                # Product information
                'product_id', 'category_id', 'category_level_1', 'category_level_2', 
                'category_level_3', 'brand', 'price',
                
                # User information
                'user_id', 'user_session',
                
                # Temporal and partitioning fields
                'year_month', 'processing_timestamp', 'effective_from', 'effective_to',
                
                # Data quality and lineage
                'data_confidence_score', 'source_system', 'source_file', 'batch_id', 'record_id'
            ]
            actual_columns = silver_df.columns
            for col_name in expected_columns:
                assert col_name in actual_columns, f"Missing column: {col_name}"
            
            # Convert to pandas for schema validation
            pandas_df = silver_df.toPandas()
            assert validate_silver_data(pandas_df), "Data failed schema validation"
            
            # Verify confidence scores
            scores = silver_df.select("data_confidence_score").collect()
            score_values = [row.data_confidence_score for row in scores]
            
            # Should have mix of scores based on our test data
            assert 1.0 in score_values, "Should have perfect confidence score"
            assert 0.6 in score_values, "Should have missing brand score"  
            assert 0.0 in score_values, "Should have missing both score"
            
            # Verify partitioning fields
            months = silver_df.select("year_month").distinct().collect()
            month_values = [row.year_month for row in months]
            assert "2020-09" in month_values
            assert "2020-10" in month_values
            
            logger.info("✅ End-to-end transformation works correctly")
            
        finally:
            # Clean up test data
            self.cleanup_test_data()
    
    def test_delta_lake_write_and_read(self):
        """Test Delta Lake write and read operations"""
        # Upload test data
        self.upload_test_csv()
        
        try:
            # Transform data
            silver_df = transform_ecommerce_bronze_to_silver(
                self.spark, 
                self.test_bronze_path, 
                self.test_silver_path
            )
            
            assert silver_df is not None, "Transformation should not return None"
            
            # Write to Delta Lake
            success = write_to_delta_lake(silver_df, self.test_silver_path)
            assert success == True, "Delta Lake write should succeed"
            
            # Read back from Delta Lake
            read_df = self.spark.read.format("delta").load(self.test_silver_path)
            
            # Verify we can read the data back
            read_count = read_df.count()
            original_count = silver_df.count()
            assert read_count == original_count, f"Read count {read_count} != original {original_count}"
            
            # Verify partitioning worked
            partitions = read_df.select("year_month", "event_type").distinct().collect()
            assert len(partitions) > 1, "Should have multiple partitions"
            
            # Test Delta-specific features
            # Check if we can query with Delta syntax
            read_df.createOrReplaceTempView("ecommerce_events")
            result = self.spark.sql("SELECT COUNT(*) as count FROM ecommerce_events").collect()
            assert result[0]["count"] == original_count
            
            logger.info("✅ Delta Lake write and read operations work correctly")
            
        finally:
            # Clean up test data
            self.cleanup_test_data()
    
    def test_error_handling(self):
        """Test error handling for edge cases"""
        # Test with non-existent file in our test bucket
        non_existent_path = f"s3a://{self.aws_config['bucket_name']}/test/nonexistent/file.csv"
        
        # Test with non-existent file
        with pytest.raises(Exception) as exc_info:
            transform_ecommerce_bronze_to_silver(
                self.spark,
                non_existent_path,
                self.test_silver_path
            )
        
        # Verify the error message contains the expected text
        assert "Path does not exist" in str(exc_info.value)
        
        logger.info("✅ Error handling test completed")


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])