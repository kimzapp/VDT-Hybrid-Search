# ==============================================================================
# VDT Hybrid Search — Redis Exact Query Cache
# ==============================================================================

import hashlib
import json
import time
import traceback
from typing import Optional


class RedisCache:
    """Exact-match query cache backed by Redis.

    Cache keys are deterministic SHA-256 hashes of all search parameters
    (query text, mode, top_k, fusion settings, rerank settings, date filters).
    This guarantees that only *identical* requests produce a cache hit.

    The class follows a **graceful degradation** pattern: if Redis is
    unreachable or any operation fails, the error is logged but never
    propagated — the search pipeline continues as if no cache exists.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
        ttl: int = 3600,
        key_prefix: str = "vdt_search:",
        enabled: bool = True,
    ):
        self.ttl = ttl
        self.key_prefix = key_prefix
        self.enabled = enabled
        self._client = None

        if not enabled:
            print("Redis cache is disabled by configuration.")
            return

        try:
            import redis
            self._client = redis.Redis(
                host=host,
                port=port,
                db=db,
                password=password,
                decode_responses=True,
                socket_connect_timeout=3,
                socket_timeout=3,
            )
            # Verify connectivity
            self._client.ping()
            print(f"Redis cache connected: {host}:{port}/{db} (TTL={ttl}s, prefix='{key_prefix}')")
        except ImportError:
            print("Warning: 'redis' package not installed. Cache is disabled.")
            self._client = None
        except Exception as e:
            print(f"Warning: Could not connect to Redis at {host}:{port} — {e}")
            print("Cache will be disabled. The search pipeline will operate normally.")
            self._client = None

    @property
    def is_available(self) -> bool:
        """Check if the Redis client is connected and operational."""
        return self._client is not None and self.enabled

    # -------------------------------------------------------------------------
    # Key construction
    # -------------------------------------------------------------------------

    def _build_cache_key(self, search_params: dict) -> str:
        """Build a deterministic cache key from search parameters.

        The params dict is JSON-serialized with sorted keys to ensure
        that the same logical request always maps to the same key,
        regardless of dict insertion order.
        """
        canonical = json.dumps(search_params, sort_keys=True, ensure_ascii=False)
        digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        return f"{self.key_prefix}{digest}"

    @staticmethod
    def extract_search_params(req) -> dict:
        """Extract cache-relevant parameters from a SearchRequest.

        This is a static helper so that the caller can build the params
        dict once and reuse it for both ``get`` and ``set``.
        """
        return {
            "query": req.query,
            "mode": req.mode,
            "top_k": req.top_k,
            "fusion_strategy": req.fusion_strategy,
            "rrf_k": req.rrf_k,
            "fusion_alpha": req.fusion_alpha,
            "rerank": req.rerank,
            "rerank_top_k": req.rerank_top_k,
            "wcr": req.wcr,
            "wcr_alpha": req.wcr_alpha,
            "date_from": req.date_from,
            "date_to": req.date_to,
        }

    # -------------------------------------------------------------------------
    # Core operations
    # -------------------------------------------------------------------------

    def get(self, search_params: dict) -> Optional[dict]:
        """Look up a cached response for the given search parameters.

        Returns the deserialized response dict on cache hit, or ``None``
        on cache miss (or if Redis is unavailable).
        """
        if not self.is_available:
            return None

        try:
            key = self._build_cache_key(search_params)
            raw = self._client.get(key)
            if raw is not None:
                return json.loads(raw)
        except Exception as e:
            print(f"Warning: Redis cache GET failed — {e}")

        return None

    def set(self, search_params: dict, response: dict, ttl: Optional[int] = None) -> bool:
        """Store a search response in the cache.

        Args:
            search_params: The search parameters dict (used to build the key).
            response: The full response dict to cache.
            ttl: Time-to-live in seconds.  Falls back to ``self.ttl``.

        Returns ``True`` if the entry was stored successfully.
        """
        if not self.is_available:
            return False

        try:
            key = self._build_cache_key(search_params)
            ttl = ttl if ttl is not None else self.ttl
            serialized = json.dumps(response, ensure_ascii=False)
            self._client.setex(key, ttl, serialized)
            return True
        except Exception as e:
            print(f"Warning: Redis cache SET failed — {e}")
            return False

    def invalidate(self, search_params: dict) -> bool:
        """Delete a specific cache entry."""
        if not self.is_available:
            return False

        try:
            key = self._build_cache_key(search_params)
            return bool(self._client.delete(key))
        except Exception as e:
            print(f"Warning: Redis cache INVALIDATE failed — {e}")
            return False

    def flush_all(self) -> bool:
        """Delete *all* cache entries that match the configured key prefix.

        Uses ``SCAN`` + ``DELETE`` instead of ``FLUSHDB`` so that other
        keys in the same Redis database are preserved.
        """
        if not self.is_available:
            return False

        try:
            cursor = 0
            pattern = f"{self.key_prefix}*"
            deleted = 0
            while True:
                cursor, keys = self._client.scan(cursor=cursor, match=pattern, count=500)
                if keys:
                    deleted += self._client.delete(*keys)
                if cursor == 0:
                    break
            print(f"Cache flush: deleted {deleted} key(s) matching '{pattern}'.")
            return True
        except Exception as e:
            print(f"Warning: Redis cache FLUSH failed — {e}")
            return False

    def stats(self) -> dict:
        """Return cache statistics.

        Includes the number of cached keys (matching the prefix) and
        basic Redis server memory info.
        """
        if not self.is_available:
            return {
                "available": False,
                "enabled": self.enabled,
                "cached_queries": 0,
            }

        try:
            # Count keys matching prefix
            cursor = 0
            count = 0
            pattern = f"{self.key_prefix}*"
            while True:
                cursor, keys = self._client.scan(cursor=cursor, match=pattern, count=500)
                count += len(keys)
                if cursor == 0:
                    break

            # Redis server memory info
            info = self._client.info(section="memory")

            return {
                "available": True,
                "enabled": self.enabled,
                "cached_queries": count,
                "ttl_seconds": self.ttl,
                "key_prefix": self.key_prefix,
                "used_memory_human": info.get("used_memory_human", "N/A"),
                "used_memory_peak_human": info.get("used_memory_peak_human", "N/A"),
            }
        except Exception as e:
            print(f"Warning: Redis cache STATS failed — {e}")
            return {
                "available": False,
                "enabled": self.enabled,
                "cached_queries": 0,
                "error": str(e),
            }
