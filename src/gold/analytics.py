"""
Gold Layer Analytics Pipeline
Creates daily aggregated business analysis table from silver layer data.
Combines ecommerce, stock market, and weather data for business intelligence.
"""

import os
import logging
from datetime import datetime, timedelta
from typing import Dict
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import (
    col, to_date, count, sum as spark_sum, avg, countDistinct, 
    stddev, dayofweek, when, lit, date_add, sequence, posexplode,
    coalesce
)
from delta import configure_spark_with_delta_pip
import great_expectations as ge
from great_expectations.core import ExpectationConfiguration
from great_expectations.dataset import SparkDFDataset

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
BUSINESS_START = "2020-09-01"
BUSINESS_END = "2021-02-28"
TARGET_STOCKS = ["AAPL", "TSLA", "AMZN", "GOOGL"]
TARGET_CITY = "new_york"

def init_spark(aws_config: Dict[str, str]) -> SparkSession:
    """Initialize the Spark session with Delta Lake and S3 configurations"""
    
    builder = (SparkSession.builder
        .config("spark.jars.packages",  # type: ignore
                "org.apache.hadoop:hadoop-aws:3.3.4,"
                "com.amazonaws:aws-java-sdk-bundle:1.12.262,"
                "io.delta:delta-spark_2.12:3.3.1")  # Delta Lake package
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")  # Delta SQL support
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")  # Delta catalog
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
        .config("spark.hadoop.fs.s3a.access.key", aws_config['access_key'])
        .config("spark.hadoop.fs.s3a.secret.key", aws_config['secret_key'])
        .config("spark.hadoop.fs.s3a.endpoint", f"s3.{aws_config['region']}.amazonaws.com")
        .config("spark.hadoop.fs.s3a.path.style.access", "true")
        .config("spark.sql.sources.partitionOverwriteMode", "dynamic"))
    
    spark = builder.appName("DailyBusinessAnalytics").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    
    logger.info("Initialized Spark session with Delta Lake support")
    return spark

def create_date_spine(spark: SparkSession) -> DataFrame:
    """
    Create continuous date range for analysis period.
    Ensures no missing dates in final gold table.
    """
    logger.info(f"Creating date spine from {BUSINESS_START} to {BUSINESS_END}")
    
    # Calculate number of days between start and end
    start_date = datetime.strptime(BUSINESS_START, "%Y-%m-%d")
    end_date = datetime.strptime(BUSINESS_END, "%Y-%m-%d")
    days_diff = (end_date - start_date).days
    
    date_spine = spark.sql(f"""
        SELECT date_add('{BUSINESS_START}', pos) as date
        FROM (SELECT posexplode(sequence(0, {days_diff})))
    """)
    
    logger.info(f"Created date spine with {date_spine.count()} days")
    return date_spine

def aggregate_sales_data(spark: SparkSession, bucket_name: str) -> DataFrame:
    """
    Aggregate ecommerce data to daily level.
    Metrics: event count, revenue, avg price, unique users
    """
    logger.info("Aggregating daily sales data...")
    
    # Read ecommerce silver data
    ecommerce_path = f"s3a://{bucket_name}/silver/ecommerce/"
    ecommerce = spark.read.format("delta").load(ecommerce_path)
    
    # Filter to business period and aggregate by date
    daily_sales = ecommerce \
        .filter((col("event_time") >= BUSINESS_START) & 
               (col("event_time") <= BUSINESS_END)) \
        .withColumn("date", to_date("event_time")) \
        .groupBy("date").agg(
            count("*").alias("sales_event_count"),
            spark_sum("price").alias("sales_total_revenue"),
            avg("price").alias("sales_avg_price"),
            countDistinct("user_id").alias("sales_unique_users"),
            countDistinct("product_id").alias("sales_unique_products")
        ) \
        .withColumn("sales_revenue_per_user", 
                   col("sales_total_revenue") / col("sales_unique_users"))
    
    logger.info(f"Sales data aggregated for {daily_sales.count()} days")
    return daily_sales

def aggregate_market_data(spark: SparkSession, bucket_name: str) -> DataFrame:
    """
    Create market index from target stocks (AAPL, TSLA, AMZN, GOOGL).
    Metrics: avg daily change %, volatility, positive stock ratio
    """
    logger.info("Aggregating market index data...")
    
    # Read alphavantage silver data
    av_path = f"s3a://{bucket_name}/silver/alphavantage/"
    stocks = spark.read.format("delta").load(av_path)
    
    # Filter to target stocks and business period
    market_stocks = stocks \
        .filter(col("ticker").isin(TARGET_STOCKS)) \
        .filter((col("trade_date") >= BUSINESS_START) & 
               (col("trade_date") <= BUSINESS_END)) \
        .withColumn("date", to_date("trade_date")) \
        .withColumn("daily_change_pct", 
                   ((col("close_price") - col("open_price")) / col("open_price")) * 100) \
        .withColumn("is_positive", when(col("daily_change_pct") > 0, 1).otherwise(0))
    
    # Aggregate by date
    daily_market = market_stocks.groupBy("date").agg(
        avg("daily_change_pct").alias("market_avg_change_pct"),
        stddev("daily_change_pct").alias("market_volatility"),
        avg("is_positive").alias("market_positive_ratio"),
        count("ticker").alias("market_stocks_traded"),
        avg("volume").alias("market_avg_volume")
    )
    
    logger.info(f"Market data aggregated for {daily_market.count()} days")
    return daily_market

def aggregate_weather_data(spark: SparkSession, bucket_name: str) -> DataFrame:
    """
    Aggregate weather data for New York.
    Metrics: avg temperature, humidity, precipitation
    """
    logger.info("Aggregating weather data for New York...")
    
    # Read weather silver data
    weather_path = f"s3a://{bucket_name}/silver/openweathermap/"
    weather = spark.read.format("delta").load(weather_path)
    
    # Filter to NYC and business period
    daily_weather = weather \
        .filter(col("city") == TARGET_CITY) \
        .filter((col("observation_date") >= BUSINESS_START) & 
               (col("observation_date") <= BUSINESS_END)) \
        .withColumn("date", to_date("observation_date")) \
        .groupBy("date").agg(
            avg("temperature_celsius").alias("weather_avg_temp_c"),
            avg("humidity_percent").alias("weather_avg_humidity"),
            avg("pressure_hpa").alias("weather_avg_pressure"),
            avg("cloud_cover_percent").alias("weather_avg_cloud_cover"),
            spark_sum("precipitation_mm").alias("weather_total_precipitation")
        ) \
        .withColumn("weather_temp_f", (col("weather_avg_temp_c") * 9/5) + 32) \
        .withColumn("weather_is_cold", when(col("weather_avg_temp_c") < 10, 1).otherwise(0)) \
        .withColumn("weather_is_rainy", when(col("weather_total_precipitation") > 0, 1).otherwise(0))
    
    logger.info(f"Weather data aggregated for {daily_weather.count()} days")
    return daily_weather

def create_time_features(df: DataFrame) -> DataFrame:
    """Add time-based features for analysis."""
    return df \
        .withColumn("day_of_week", dayofweek("date")) \
        .withColumn("is_weekend", col("day_of_week").isin([1, 7]).cast("int")) \
        .withColumn("is_monday", when(col("day_of_week") == 2, 1).otherwise(0)) \
        .withColumn("is_friday", when(col("day_of_week") == 6, 1).otherwise(0)) \
        .withColumn("month", col("date").substr(6, 2).cast("int")) \
        .withColumn("is_holiday_season", 
                   when(col("month").isin([11, 12, 1]), 1).otherwise(0))

def validate_gold_table(spark: SparkSession, gold_table: DataFrame) -> bool:
    """
    Validate the gold table using Great Expectations.
    Ensures data quality and consistency of the analytics output.
    """
    logger.info("Starting gold table validation...")
    
    # Convert to Great Expectations dataset
    ge_df = SparkDFDataset(gold_table)
    
    # Define expectations
    expectations = [
        # Date range validation
        ExpectationConfiguration(
            expectation_type="expect_column_values_to_be_between",
            kwargs={
                "column": "date",
                "min_value": BUSINESS_START,
                "max_value": BUSINESS_END,
                "parse_strings_as_datetimes": True
            }
        ),
        
        # Sales metrics validation
        ExpectationConfiguration(
            expectation_type="expect_column_values_to_not_be_null",
            kwargs={"column": "sales_event_count"}
        ),
        ExpectationConfiguration(
            expectation_type="expect_column_values_to_be_between",
            kwargs={
                "column": "sales_event_count",
                "min_value": 0
            }
        ),
        ExpectationConfiguration(
            expectation_type="expect_column_values_to_be_between",
            kwargs={
                "column": "sales_total_revenue",
                "min_value": 0
            }
        ),
        
        # Market metrics validation
        ExpectationConfiguration(
            expectation_type="expect_column_values_to_be_between",
            kwargs={
                "column": "market_avg_change_pct",
                "min_value": -100,
                "max_value": 100
            }
        ),
        ExpectationConfiguration(
            expectation_type="expect_column_values_to_be_between",
            kwargs={
                "column": "market_positive_ratio",
                "min_value": 0,
                "max_value": 1
            }
        ),
        
        # Weather metrics validation
        ExpectationConfiguration(
            expectation_type="expect_column_values_to_be_between",
            kwargs={
                "column": "weather_avg_temp_c",
                "min_value": -50,
                "max_value": 50
            }
        ),
        ExpectationConfiguration(
            expectation_type="expect_column_values_to_be_between",
            kwargs={
                "column": "weather_avg_humidity",
                "min_value": 0,
                "max_value": 100
            }
        ),
        
        # Time features validation
        ExpectationConfiguration(
            expectation_type="expect_column_values_to_be_between",
            kwargs={
                "column": "day_of_week",
                "min_value": 1,
                "max_value": 7
            }
        ),
        ExpectationConfiguration(
            expectation_type="expect_column_values_to_be_in_set",
            kwargs={
                "column": "is_weekend",
                "value_set": [0, 1]
            }
        ),
        
        # Metadata validation
        ExpectationConfiguration(
            expectation_type="expect_column_values_to_not_be_null",
            kwargs={"column": "processing_timestamp"}
        ),
        ExpectationConfiguration(
            expectation_type="expect_column_values_to_be_in_set",
            kwargs={
                "column": "data_source",
                "value_set": ["daily_business_analytics"]
            }
        )
    ]
    
    # Run validations
    validation_results = []
    for expectation in expectations:
        result = ge_df.validate(expectation)
        validation_results.append(result)
        
        if not result.success:
            logger.error(f"Validation failed for {expectation.expectation_type}: {result.result}")
        else:
            logger.info(f"Validation passed for {expectation.expectation_type}")
    
    # Check if all validations passed
    all_passed = all(result.success for result in validation_results)
    
    if all_passed:
        logger.info("All validations passed successfully!")
    else:
        logger.error("Some validations failed. Check the logs for details.")
    
    return all_passed

def create_gold_table(spark: SparkSession, bucket_name: str) -> DataFrame:
    """
    Main function to create the daily business analytics gold table.
    Combines all data sources into analysis-ready format.
    """
    logger.info("Starting gold layer analytics table creation...")
    
    try:
        # Create base date spine
        date_spine = create_date_spine(spark)
        
        # Get daily aggregations from each source
        daily_sales = aggregate_sales_data(spark, bucket_name)
        daily_market = aggregate_market_data(spark, bucket_name)
        daily_weather = aggregate_weather_data(spark, bucket_name)
        
        # Join all data sources
        gold_table = date_spine \
            .join(daily_sales, "date", "left") \
            .join(daily_market, "date", "left") \
            .join(daily_weather, "date", "left")
        
        # Add time-based features
        gold_table = create_time_features(gold_table)
        
        # Fill null values with appropriate defaults
        gold_table = gold_table.fillna({
            "sales_event_count": 0,
            "sales_total_revenue": 0.0,
            "sales_unique_users": 0,
            "market_avg_change_pct": 0.0,
            "market_volatility": 0.0,
            "weather_avg_temp_c": 0.0,
            "weather_total_precipitation": 0.0
        })
        
        # Add metadata
        gold_table = gold_table \
            .withColumn("processing_timestamp", lit(datetime.now())) \
            .withColumn("data_source", lit("daily_business_analytics")) \
            .withColumn("version", lit("1.0"))
        
        # Validate the gold table
        if not validate_gold_table(spark, gold_table):
            raise ValueError("Gold table validation failed")
        
        # Write to gold layer
        output_path = f"s3a://{bucket_name}/gold/daily_business_analytics/"
        
        logger.info(f"Writing gold table to {output_path}")
        gold_table.write \
            .format("delta") \
            .mode("overwrite") \
            .option("overwriteSchema", "true") \
            .save(output_path)
        
        # Log summary statistics
        record_count = gold_table.count()
        date_range = gold_table.select("date").orderBy("date").collect()
        min_date = date_range[0]["date"]
        max_date = date_range[-1]["date"]
        
        logger.info(f"Gold table created successfully!")
        logger.info(f"Records: {record_count}")
        logger.info(f"Date range: {min_date} to {max_date}")
        
        # Show sample data
        logger.info("Sample data preview:")
        gold_table.select(
            "date", "sales_event_count", "sales_total_revenue", 
            "market_avg_change_pct", "weather_avg_temp_c"
        ).orderBy("date").show(10)
        
        return gold_table
        
    except Exception as e:
        logger.error(f"Error creating gold table: {str(e)}")
        raise

def main():
    """
    Main execution function.
    Requires AWS credentials as environment variables.
    """
    # Get AWS configuration from environment
    aws_config: Dict[str, str] = {
        'access_key': os.getenv('AWS_ACCESS_KEY', ''),
        'secret_key': os.getenv('AWS_SECRET_KEY', ''),
        'bucket_name': os.getenv('AWS_S3_BUCKET_NAME', ''),
        'region': os.getenv('AWS_DEFAULT_REGION', 'us-east-1')
    }
    
    if not aws_config['access_key'] or not aws_config['secret_key']:
        raise ValueError("AWS credentials not found in environment variables")
    
    # Initialize Spark
    spark = init_spark(aws_config)
    
    try:
        # Create and run analytics pipeline
        gold_table = create_gold_table(spark, aws_config['bucket_name'])
        print("Daily business analytics gold table created successfully!")
        return gold_table
    finally:
        spark.stop()

if __name__ == "__main__":
    main()