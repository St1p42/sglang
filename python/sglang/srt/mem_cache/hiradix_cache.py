from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from sglang.srt.mem_cache.radix_cache import RadixCache
from sglang.srt.mem_cache.vanilla_hiradix_cache import VanillaHiRadixCache

if TYPE_CHECKING:
    from sglang.srt.mem_cache.cache_init_params import CacheInitParams
    from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)


def _resolve_hicache_impl(server_args: ServerArgs) -> str:
    impl_name = getattr(server_args, "hicache_impl", "vanilla").lower()
    if impl_name not in ("vanilla", "custom"):
        raise ValueError(
            f"Unsupported HiCache implementation: {impl_name}. "
            "Expected 'vanilla' or 'custom'."
        )
    return impl_name


def _load_hicache_impl(impl_name: str):
    if impl_name == "custom":
        from sglang.srt.mem_cache.custom_hiradix_cache import CustomHiRadixCache

        return CustomHiRadixCache
    return VanillaHiRadixCache


class HiRadixCache(RadixCache):
    """Route HiCache construction to the selected implementation."""

    def __new__(cls, params: CacheInitParams, server_args: ServerArgs):
        impl_name = _resolve_hicache_impl(server_args)
        impl_cls = _load_hicache_impl(impl_name)
        logger.info("Using %s hicache implementation.", impl_name)
        return impl_cls(params=params, server_args=server_args)
