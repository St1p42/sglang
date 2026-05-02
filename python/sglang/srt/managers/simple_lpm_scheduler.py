"""Simplified cache-aware schedulers for course project experimentation.
 
SimpleLPMScheduler: our reproduction of LPM — sorts by cached prefix length,
    with the following improvements over the naive version:
      1. Raised FCFS fallback threshold so LPM runs during cold-start phase.
      2. Age-aware tie-breaking: older requests win when prefix lengths are similar.
      3. Bucket-based round-robin: requests grouped into prefix-length buckets,
         then interleaved across buckets to prevent starvation of short-prefix
         requests while still preserving cache locality within each bucket.
 
BaseFIFOScheduler: pure FCFS baseline — no prefix matching, no reordering.
 
Both expose the same calc_priority(waiting_queue) -> bool interface as
SchedulePolicy so they can be swapped in via --cache-aware-scheduling.
"""
from __future__ import annotations
 
import logging
import time
from typing import List
 
import torch
 
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache
from sglang.srt.mem_cache.radix_cache import RadixKey
 
logger = logging.getLogger(__name__)
 
# Raised from 128 → 2048 so LPM is active during the cold-start phase.
# The original 128 caused LPM to silently fall back to FCFS for the entire
# early scheduling window when the queue is large — exactly the regime where
# cache-aware ordering matters most.
FCFS_FALLBACK_THRESHOLD = 2048
 
# Prefix-length bucket boundaries (in tokens).
# Requests are grouped into [0, LOW), [LOW, HIGH), [HIGH, ∞).
# Round-robin across buckets ensures short-prefix requests are never starved
# by long-prefix requests that dominate a strict sort.
BUCKET_LOW = 100
BUCKET_HIGH = 500
 
 
def _clear_prefix_state(waiting_queue: List[Req]) -> None:
    """Reset prefix metadata so stale values from a previous scheduling
    round cannot leak into a fallback FCFS path."""
    for r in waiting_queue:
        r.prefix_indices = torch.empty((0,), dtype=torch.int64)
        r.last_node = None
        r.last_host_node = None
        r.host_hit_length = 0
 
 
def _prefix_bucket(prefix_len: int) -> int:
    """Map a prefix length to a bucket index (0 = short, 1 = medium, 2 = long)."""
    if prefix_len < BUCKET_LOW:
        return 0
    if prefix_len < BUCKET_HIGH:
        return 1
    return 2
 
 
def _arrival_time(req: Req) -> float:
    """Return a sortable arrival timestamp for age-aware tie-breaking.
 
    Uses queue_time_start when available (set by the SGLang runtime),
    falling back to 0.0 so older requests sort before newer ones under
    a min-first key even when the field is missing.
    """
    return getattr(req, 'queue_time_start', 0.0)
 
 
class SimpleLPMScheduler:
    """Improved simplified LPM scheduler.
 
    Core idea: group requests by prefix-length bucket, then within each
    bucket sort longest-cached-prefix-first with age-aware tie-breaking,
    then interleave across buckets via round-robin.
 
    This preserves the cache-locality benefit of LPM (grouping requests
    that share long cached prefixes) while preventing starvation of
    short-prefix requests that would never rise to the top of a strict
    global sort.
    """
 
    def __init__(self, tree_cache: BasePrefixCache):
        self.tree_cache = tree_cache
 
    def calc_priority(self, waiting_queue: List[Req]) -> bool:
        """Reorder *waiting_queue* in-place using bucket round-robin + LPM.
 
        Returns True when prefix matching was performed, False on FCFS fallback.
        Interface matches SchedulePolicy.calc_priority exactly.
        """
        if len(waiting_queue) == 0:
            return False
 
        if not self._should_run_prefix_matching(waiting_queue):
            _clear_prefix_state(waiting_queue)
            logger.warning(
                "[CUSTOM SCHED] queue_len=%d > %d, FCFS fallback",
                len(waiting_queue),
                FCFS_FALLBACK_THRESHOLD,
            )
            return False
 
        self._match_prefixes(waiting_queue)
        reordered = self._bucket_round_robin(waiting_queue)
 
        # Write reordered sequence back into the original list in-place.
        waiting_queue[:] = reordered
 
        top = waiting_queue[:5]
        logger.warning(
            "[CUSTOM SCHED] sorted top=%s",
            [(r.rid, len(r.prefix_indices)) for r in top],
        )
        return True
 
    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
 
    def _should_run_prefix_matching(self, waiting_queue: List[Req]) -> bool:
        if len(waiting_queue) > FCFS_FALLBACK_THRESHOLD:
            return False
        if getattr(self.tree_cache, "disable", True):
            return False
        return True
 
    def _match_prefixes(self, waiting_queue: List[Req]) -> None:
        """Populate prefix fields on every request via the radix tree."""
        for r in waiting_queue:
            prefix_ids = r.origin_input_ids + r.output_ids
            match_result = self.tree_cache.match_prefix(
                rid=r.rid,
                key=RadixKey(token_ids=prefix_ids, extra_key=r.extra_key),
            )
            r.prefix_indices = match_result.device_indices
            r.last_node = match_result.last_device_node
            r.last_host_node = match_result.last_host_node
            r.host_hit_length = match_result.host_hit_length
 
    def _bucket_round_robin(self, waiting_queue: List[Req]) -> List[Req]:
        """Group requests into prefix-length buckets, sort within each bucket
        by (prefix_len DESC, arrival_time ASC), then interleave across buckets
        via round-robin.
 
        The round-robin ensures that even if bucket 2 (long-prefix) is large,
        requests from buckets 0 and 1 are still scheduled regularly, bounding
        the worst-case wait time for short-prefix requests.
        """
        buckets: list[list[Req]] = [[], [], []]
 
        for r in waiting_queue:
            prefix_len = len(r.prefix_indices)
            b = _prefix_bucket(prefix_len)
            buckets[b].append(r)
 
        # Within each bucket: longest prefix first, then oldest request first
        # when prefix lengths are equal (age-aware tie-breaking).
        for b in buckets:
            b.sort(key=lambda r: (-len(r.prefix_indices), _arrival_time(r)))
 
        # Round-robin interleave: pick one request from each non-empty bucket
        # in descending bucket order (long → medium → short), cycling until
        # all buckets are drained.
        result: List[Req] = []
        bucket_order = [2, 1, 0]  # long-prefix first within each round
        while any(buckets):
            for idx in bucket_order:
                if buckets[idx]:
                    result.append(buckets[idx].pop(0))
 
        return result
 
 
class BaseFIFOScheduler:
    """Pure FCFS baseline — no prefix matching, no reordering."""
 
    def calc_priority(self, waiting_queue: List[Req]) -> bool:
        """No-op: keeps arrival order. Returns False (no prefix info computed)."""
        _clear_prefix_state(waiting_queue)
        logger.warning(
            "[CUSTOM SCHED] base FCFS, queue_len=%d",
            len(waiting_queue),
        )
        return False
