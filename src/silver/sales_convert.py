#!/usr/bin/env python3
"""
E-commerce Data Pipeline: Bronze to Silver with EventStore
Process CSV data from S3 and send events to EventStore for ML analysis
"""

import boto3
import pandas as pd
import json
import uuid
import os
from datetime import datetime
from esdbclient import EventStoreDBClient, NewEvent, StreamState
from io import StringIO
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class ECommerceEventProcessor:
    def __init__(self, eventstore_uri="esdb://localhost:2113?tls=false"):
        """Initialize the processor with EventStore connection"""
        self.client = EventStoreDBClient(uri=eventstore_uri)
        
        # Initialize S3 client with credentials from .env
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=os.getenv('TEST_AWS_ACCESS_KEY'),
            aws_secret_access_key=os.getenv('TEST_AWS_SECRET_KEY'),
            region_name=os.getenv('TEST_AWS_DEFAULT_REGION')
        )
        
    def read_sample_from_s3(self, bucket_name, file_key, sample_size=100):
        """Read a sample of rows from S3 CSV for testing"""
        print(f"📥 Reading sample data from s3://{bucket_name}/{file_key}")
        
        # Get the object from S3
        response = self.s3_client.get_object(Bucket=bucket_name, Key=file_key)
        
        # Read CSV into pandas DataFrame
        csv_content = response['Body'].read().decode('utf-8')
        df = pd.read_csv(StringIO(csv_content))
        
        # Take a sample for testing
        sample_df = df.head(sample_size)
        print(f"✅ Loaded {len(sample_df)} rows for processing")
        print(f"📊 Event types found: {sample_df['event_type'].unique()}")
        
        return sample_df
    
    def process_user_events(self, df):
        """
        Process events and send to EventStore
        Each user session becomes a stream
        """
        print("🔄 Processing events and sending to EventStore...")
        
        # Group by user session to create streams
        session_groups = df.groupby('user_session')
        
        processed_sessions = 0
        
        for session_id, session_events in session_groups:
            stream_name = f"user-session-{session_id}"
            
            # Sort events by time for proper order
            session_events = session_events.sort_values('event_time')
            
            events_to_append = []
            
            for _, row in session_events.iterrows():
                # Create EventStore event from each row
                event = NewEvent(
                    id=uuid.uuid4(),
                    type=f"ecommerce.{row['event_type']}",  # e.g., "ecommerce.view"
                    data=json.dumps({
                        "event_time": row['event_time'],
                        "product_id": str(row['product_id']),
                        "category_id": str(row['category_id']) if pd.notna(row['category_id']) else None,
                        "category_code": row['category_code'] if pd.notna(row['category_code']) else None,
                        "brand": row['brand'] if pd.notna(row['brand']) else None,
                        "price": float(row['price']) if pd.notna(row['price']) else None,
                        "user_id": str(row['user_id']),
                        "user_session": row['user_session']
                    }).encode('utf-8')
                )
                events_to_append.append(event)
            
            # Send all events for this session to EventStore
            try:
                # Try to append to new stream first
                try:
                    self.client.append_to_stream(
                        stream_name=stream_name,
                        events=events_to_append,
                        current_version=StreamState.NO_STREAM  # New stream
                    )
                except Exception as stream_exists_error:
                    # If stream exists, append to existing stream
                    if "not StreamState.NO_STREAM" in str(stream_exists_error):
                        # Get current stream version and append
                        existing_events = list(self.client.read_stream(stream_name))
                        current_version = len(existing_events) - 1
                        
                        self.client.append_to_stream(
                            stream_name=stream_name,
                            events=events_to_append,
                            current_version=current_version
                        )
                    else:
                        raise stream_exists_error
                        
                processed_sessions += 1
                
                if processed_sessions % 10 == 0:  # Progress update every 10 sessions
                    print(f"  📤 Processed {processed_sessions} sessions...")
                    
            except Exception as e:
                print(f"❌ Error processing session {session_id}: {e}")
        
        print(f"✅ Successfully processed {processed_sessions} user sessions")
        return processed_sessions
    
    def verify_data_in_eventstore(self, sample_session_id):
        """Check that data was properly stored in EventStore"""
        print(f"🔍 Verifying data for session: {sample_session_id}")
        
        stream_name = f"user-session-{sample_session_id}"
        
        try:
            events = list(self.client.read_stream(stream_name))
            print(f"✅ Found {len(events)} events in stream '{stream_name}':")
            
            for i, event in enumerate(events):
                event_data = json.loads(event.data.decode('utf-8'))
                print(f"  {i+1}. {event.type} - Product: {event_data['product_id']} - Price: ${event_data['price']} - Time: {event_data['event_time']}")
                
        except Exception as e:
            print(f"❌ Error reading stream: {e}")

def main():
    """Main pipeline execution"""
    print("🚀 Starting E-commerce Data Pipeline")
    
    # Initialize processor
    processor = ECommerceEventProcessor()
    
    # Your actual S3 details
    BUCKET_NAME = os.getenv('TEST_AWS_S3_BUCKET_NAME', 'mock-datalake1-test')
    FILE_KEY = "bronze/ecommerce/events.csv"
    
    print(f"📥 Will process data from s3://{BUCKET_NAME}/{FILE_KEY}")
    
    try:
        # Read sample data from S3 (let's try 200 rows now)
        df = processor.read_sample_from_s3(BUCKET_NAME, FILE_KEY, sample_size=200)
        
        # Process the events
        sessions_processed = processor.process_user_events(df)
        
        # Verify with the first session we find
        if not df.empty:
            first_session = df['user_session'].iloc[0]
            processor.verify_data_in_eventstore(first_session)
        
        print(f"\n🎉 Pipeline completed successfully!")
        print(f"📊 Summary:")
        print(f"  - Processed {len(df)} events")
        print(f"  - Across {sessions_processed} user sessions")
        print(f"  - Event types: {', '.join(df['event_type'].unique())}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Please check your .env file has the correct AWS credentials")

if __name__ == "__main__":
    main()