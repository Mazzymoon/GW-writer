from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RedisCache:
    redis_url: str

    def client(self):
        try:
            import redis
        except ImportError as exc:
            raise RuntimeError("Missing dependency: redis. Install requirements.txt first.") from exc
        return redis.Redis.from_url(self.redis_url, decode_responses=True)
