import asyncio
import pickle
import time
from typing import Generic, Optional

from diskcache import Cache
from loguru import logger

from langflow.services.cache.base import AsyncBaseCacheService, AsyncLockType
from langflow.services.cache.utils import CACHE_MISS


class AsyncDiskCache(AsyncBaseCacheService, Generic[AsyncLockType]):  # type: ignore
    def __init__(self, cache_dir, max_size=None, expiration_time=3600):
        self.cache = Cache(cache_dir)
        self.lock = asyncio.Lock()
        self.max_size = max_size
        self.expiration_time = expiration_time

    async def get(self, key, lock: Optional[asyncio.Lock] = None):
        if not lock:
            async with self.lock:
                return await self._get(key)
        else:
            return await self._get(key)

    async def _get(self, key):
        item = await asyncio.to_thread(self.cache.get, key, default=None)
        if item:
            if time.time() - item["time"] < self.expiration_time:
                await asyncio.to_thread(self.cache.touch, key)  # Refresh the expiry time
                return pickle.loads(item["value"]) if isinstance(item["value"], bytes) else item["value"]
            else:
                logger.info(f"Cache item for key '{key}' has expired and will be deleted.")
                await self._delete(key)  # Log before deleting the expired item
        return CACHE_MISS

    async def set(self, key, value, lock: Optional[asyncio.Lock] = None):
        if not lock:
            async with self.lock:
                await self._set(key, value)
        else:
            await self._set(key, value)

    async def _set(self, key, value):
        if self.max_size and len(self.cache) >= self.max_size:
            await asyncio.to_thread(self.cache.cull)
        item = {"value": pickle.dumps(value) if not isinstance(value, (str, bytes)) else value, "time": time.time()}
        await asyncio.to_thread(self.cache.set, key, item)

    async def delete(self, key, lock: Optional[asyncio.Lock] = None):
        if not lock:
            async with self.lock:
                await self._delete(key)
        else:
            await self._delete(key)

    async def _delete(self, key):
        await asyncio.to_thread(self.cache.delete, key)

    async def clear(self, lock: Optional[asyncio.Lock] = None):
        if not lock:
            async with self.lock:
                await self._clear()
        else:
            await self._clear()

    async def _clear(self):
        await asyncio.to_thread(self.cache.clear)

    async def upsert(self, key, value, lock: Optional[asyncio.Lock] = None):
        if not lock:
            async with self.lock:
                await self._upsert(key, value)
        else:
            await self._upsert(key, value)

    async def _upsert(self, key, value):
        existing_value = await self.get(key)
        if existing_value is not CACHE_MISS and isinstance(existing_value, dict) and isinstance(value, dict):
            existing_value.update(value)
            value = existing_value
        await self.set(key, value)

    def __contains__(self, key):
        return asyncio.run(asyncio.to_thread(self.cache.__contains__, key))

    async def teardown(self):
        # Clean up the cache directory
        self.cache.clear(retry=True)
