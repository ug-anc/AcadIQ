"""Response cache.

Prototype: normalized-query exact match with TTL + LRU eviction. This satisfies
the PRD's "cache hit on repeated query within 5 minutes" gate. A semantic cache
(embed the query, match on cosine > 0.97) is the documented upgrade path; the
interface below stays the same when that lands.
"""

from __future__ import annotations

import re
import time
from collections import OrderedDict
from typing import Optional

from app.config import get_settings

_WS = re.compile(r"\s+")


def _normalize(query: str) -> str:
    return _WS.sub(" ", query.strip().lower())


class ResponseCache:
    def __init__(self) -> None:
        self._store: "OrderedDict[str, tuple[float, dict]]" = OrderedDict()

    def get(self, query: str) -> Optional[dict]:
        s = get_settings()
        key = _normalize(query)
        item = self._store.get(key)
        if item is None:
            return None
        ts, payload = item
        if time.time() - ts > s.CACHE_TTL_SECONDS:
            self._store.pop(key, None)
            return None
        self._store.move_to_end(key)
        return payload

    def set(self, query: str, payload: dict) -> None:
        s = get_settings()
        key = _normalize(query)
        self._store[key] = (time.time(), payload)
        self._store.move_to_end(key)
        while len(self._store) > s.CACHE_MAX_ENTRIES:
            self._store.popitem(last=False)

    def clear(self) -> None:
        self._store.clear()


_CACHE: ResponseCache | None = None


def get_cache() -> ResponseCache:
    global _CACHE
    if _CACHE is None:
        _CACHE = ResponseCache()
    return _CACHE
