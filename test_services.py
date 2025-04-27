import redis

# Connect to Redis
r = redis.Redis(
    host='recs-redis-cache-vjt3j6.serverless.apse2.cache.amazonaws.com',
    port=6379,
    password='',    # Add password if needed
    decode_responses=True
)

# List all user_browsed:* keys
for key in r.scan_iter("user_browsed:*"):
    print(key, r.lrange(key, 0, 4))

# Check browsing history for user 12
print(r.lrange("user_browsed:11", 0, 4))