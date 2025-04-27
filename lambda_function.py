import redis
import json
import os
import base64

# Connect to Redis
r = redis.Redis(
    host=os.environ['REDIS_HOST'],
    port=6379,
    password=os.environ.get('REDIS_AUTH', ''),
    decode_responses=True,
    ssl=True  # 🔥 add this line to force TLS
)

def lambda_handler(event, context):
    for record in event["Records"]:
        # Decode base64 data first
        payload = json.loads(base64.b64decode(record["kinesis"]["data"]).decode('utf-8'))

        user_id = payload.get("user_id")
        item_id = payload.get("item_id")

        if user_id is None or item_id is None:
            print("Skipping record: missing user_id or item_id")
            continue  # Skip bad records

        # Update Redis browsing history
        key = f"user_browsed:{user_id}"
        r.lpush(key, item_id)
        r.ltrim(key, 0, 4)  # Keep only last 5 items

    return {"statusCode": 200}
