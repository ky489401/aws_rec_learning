
import boto3
import base64
import json
import time

# Initialize Kinesis client
kinesis_client = boto3.client('kinesis', region_name='ap-southeast-2')

stream_name = "clickstream"

# Get the shard ID
stream = kinesis_client.describe_stream(StreamName=stream_name)
shard_id = stream['StreamDescription']['Shards'][0]['ShardId']

# Get the shard iterator
shard_iterator = kinesis_client.get_shard_iterator(
    StreamName=stream_name,
    ShardId=shard_id,
    ShardIteratorType='LATEST'  # 'TRIM_HORIZON' for all old events
)['ShardIterator']

# Start reading data
while True:
    response = kinesis_client.get_records(ShardIterator=shard_iterator, Limit=10)

    records = response['Records']
    if records:
        for record in records:
            data = record['Data'].decode('utf-8')
            print("Got event:", json.loads(data))

    shard_iterator = response['NextShardIterator']
    time.sleep(2)  # Poll every 2 seconds
