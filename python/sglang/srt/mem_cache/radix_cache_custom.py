from __future__ import annotations

import logging
from typing import Any, Optional

from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache, MatchResult
from sglang.srt.mem_cache.radix_cache_vanilla import VanillaRadixCacheImpl


logger = logging.getLogger(__name__)


class CustomRadixCacheImpl(BasePrefixCache):
    def __init__(self, params: CacheInitParams):
        logger.warning("[RADIX_CACHE_IMPL] CUSTOM RADIX_CACHE.PY")
        self.impl = VanillaRadixCacheImpl(params)

    def __getattr__(self, name: str):
        return getattr(self.impl, name)

    def reset(self):
        return self.impl.reset()

    def match_prefix(self, key: Any, **kwargs) -> MatchResult:
        return self.impl.match_prefix(key, **kwargs)

    def cache_finished_req(self, req, is_insert: bool = True, **kwargs):
        return self.impl.cache_finished_req(req, is_insert=is_insert, **kwargs)

    def cache_unfinished_req(self, req, **kwargs):
        return self.impl.cache_unfinished_req(req, **kwargs)

    def evict(self, num_tokens: int):
        return self.impl.evict(num_tokens)

    def inc_lock_ref(self, node: Any):
        return self.impl.inc_lock_ref(node)

    def dec_lock_ref(self, node: Any, swa_uuid_for_lock: Optional[str] = None):
        return self.impl.dec_lock_ref(node)

    def evictable_size(self):
        return self.impl.evictable_size()

    def protected_size(self):
        return self.impl.protected_size()

    def total_size(self):
        return self.impl.total_size()

    def pretty_print(self):
        return self.impl.pretty_print()

    def take_events(self):
        return self.impl.take_events()
