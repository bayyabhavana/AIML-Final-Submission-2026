
import json
import redis
from typing import Any, Optional

# Redis connection
redis_client = redis.Redis(
    host="localhost",
    port=6379,
    decode_responses=True,
    socket_connect_timeout=5
)

CACHE_TTL = 3600  # 1 hour


def get_cached_recommendations(user_id: int) -> Optional[list]:
    """Retrieve recommendations from Redis."""
    key = f"recommendations:user:{user_id}"
    cached_data = redis_client.get(key)

    if cached_data:
        print(f"Cache HIT for user {user_id}")
        return json.loads(cached_data)

    print(f"Cache MISS for user {user_id}")
    return None


def cache_recommendations(user_id: int, recommendations: list) -> bool:
    """Save recommendations in Redis."""
    key = f"recommendations:user:{user_id}"

    redis_client.setex(
        key,
        CACHE_TTL,
        json.dumps(recommendations)
    )

    print(f"Recommendations cached for user {user_id}")
    return True


def clear_user_cache(user_id: int) -> None:
    """Delete cached recommendations for a user."""
    redis_client.delete(f"recommendations:user:{user_id}")


def test_redis():
    print("=" * 45)
    print("REDIS RECOMMENDATION CACHE TEST")
    print("=" * 45)

    try:
        redis_client.ping()
        print("Redis connection: SUCCESS")

        sample_user = 1
        sample_recommendations = [
            {"movieId": 1, "title": "Toy Story (1995)", "score": 4.8},
            {"movieId": 2, "title": "Jumanji (1995)", "score": 4.5}
        ]

        clear_user_cache(sample_user)

        # First request: cache miss
        result = get_cached_recommendations(sample_user)

        if result is None:
            cache_recommendations(
                sample_user,
                sample_recommendations
            )

        # Second request: cache hit
        result = get_cached_recommendations(sample_user)

        print("\nCached recommendations:")
        print(json.dumps(result, indent=4))

        print("\nRedis cache test: PASSED")

    except redis.RedisError as error:
        print("Redis error:", error)
        raise


if __name__ == "__main__":
    test_redis()
