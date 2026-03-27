# this is the reimplemented version...
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Tuple, Union

if TYPE_CHECKING:
    from sglang.srt.mem_cache.radix_cache import TreeNode


class EvictionStrategy(ABC):
    """Base interface for eviction strategies."""

    @abstractmethod
    def get_priority(self, node: "TreeNode") -> Union[float, Tuple]:
        """Return a comparable priority. Smaller values are evicted first."""
        raise NotImplementedError


class LRUStrategy(EvictionStrategy):
    """Least Recently Used (default strategy)."""

    def get_priority(self, node: "TreeNode") -> float:
        return node.last_access_time


class LFUStrategy(EvictionStrategy):
    """Least Frequently Used with LRU tie-break."""

    def get_priority(self, node: "TreeNode") -> Tuple[int, float]:
        return (node.hit_count, node.last_access_time)


class PriorityStrategy(EvictionStrategy):
    """Priority-based eviction (lower priority evicted first)."""

    def get_priority(self, node: "TreeNode") -> Tuple[int, float]:
        return (node.priority, node.last_access_time)
