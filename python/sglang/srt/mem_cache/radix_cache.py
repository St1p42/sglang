from __future__ import annotations

import logging
from typing import Any, Optional

from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache, MatchResult
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.radix_cache_custom import CustomRadixCacheImpl
from sglang.srt.mem_cache.radix_cache_vanilla import (
    RadixKey,
    TreeNode,
    VanillaRadixCacheImpl,
    _check_extra_key,
    _key_match_page_size1,
    _key_match_paged,
    compute_node_hash_values,
    get_child_key,
    split_node_hash_value,
)


logger = logging.getLogger(__name__)


class RadixCache(BasePrefixCache):
    def __init__(self, params: CacheInitParams):
        impl_name = getattr(params, "radix_cache_impl", "vanilla")
        if impl_name == "custom":
            logger.warning("[RADIX_CACHE_IMPL] SELECTED custom")
            self.impl = CustomRadixCacheImpl(params)
        else:
            logger.warning("[RADIX_CACHE_IMPL] SELECTED vanilla")
            logger.warning("[RADIX_CACHE_IMPL] VANILLA RADIX_CACHE.PY")
            self.impl = VanillaRadixCacheImpl(params)

    @classmethod
    def create_simulated(
        cls,
        disable: bool = False,
        mock_allocator: Optional[Any] = None,
        page_size: int = 1,
        enable_kv_cache_events: bool = False,
        radix_cache_impl: str = "vanilla",
    ) -> "RadixCache":
        params = CacheInitParams(
            disable=disable,
            req_to_token_pool=None,
            token_to_kv_pool_allocator=mock_allocator,
            page_size=page_size,
            radix_cache_impl=radix_cache_impl,
            enable_kv_cache_events=enable_kv_cache_events,
        )
        return cls(params)

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
