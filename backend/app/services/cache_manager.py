import time
import threading
from typing import Any, Optional

try:
    import redis

    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    redis = None


class InMemoryCache:
    def __init__(self):
        self._cache = {}
        self._lock = threading.Lock()

    def set(self, key: str, value: Any, expire: Optional[int] = None):
        with self._lock:
            expire_at = time.time() + expire if expire else None
            self._cache[key] = (value, expire_at)

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            item = self._cache.get(key)
            if not item:
                return None
            value, expire_at = item
            if expire_at and time.time() > expire_at:
                del self._cache[key]
                return None
            return value

    def delete(self, key: str):
        with self._lock:
            if key in self._cache:
                del self._cache[key]


class CacheManager:
    def __init__(self, redis_url: Optional[str] = None):
        self.memory_cache = InMemoryCache()
        self.redis_cache = None
        if redis_url and REDIS_AVAILABLE:
            self.redis_cache = redis.StrictRedis.from_url(
                redis_url, decode_responses=True
            )

    def set(self, key: str, value: Any, expire: Optional[int] = None):
        self.memory_cache.set(key, value, expire)
        if self.redis_cache:
            self.redis_cache.set(key, value, ex=expire)

    def get(self, key: str) -> Optional[Any]:
        value = self.memory_cache.get(key)
        if value is not None:
            return value
        if self.redis_cache:
            value = self.redis_cache.get(key)
            if value is not None:
                self.memory_cache.set(
                    key, value, expire=60
                )  # cache in memory for 1 min
            return value
        return None

    def delete(self, key: str):
        self.memory_cache.delete(key)
        if self.redis_cache:
            self.redis_cache.delete(key)


# Example usage:
# cache = CacheManager(redis_url="redis://localhost:6379/0")
# cache.set("tts:segment:123", b"audio-bytes", expire=3600)
# audio = cache.get("tts:segment:123")
