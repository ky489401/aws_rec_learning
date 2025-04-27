import boto3, random, json
import time

# aws kinesis create-stream --stream-name clickstream --shard-count 1

kinesis = boto3.client("kinesis")

while True:
    user_id = random.randint(0, 49)
    item_id = random.randint(0, 99)
    data = json.dumps({"user_id": user_id, "item_id": item_id, "event": "purchase"})
    kinesis.put_record(StreamName="clickstream", Data=data, PartitionKey=str(user_id))
    print("Sent:", data)
    time.sleep(2)  # adjustable