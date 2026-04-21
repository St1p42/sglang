"""Simplified cache-aware schedulers for course project experimentation.

SimpleLPMScheduler: our reproduction of LPM — sorts by cached prefix length,
    without the in-batch deprioritization heuristics of the original.
BaseFIFOScheduler: pure FCFS baseline — no prefix matching, no reordering.

Both expose the same calc_priority(waiting_queue) -> bool interface as
SchedulePolicy so they can be swapped in via --cache-aware-scheduling.
"""

from __future__ import annotations

import logging
from typing import List

import torch

from sglang.srt.managers.schedule_batch import Req
from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache
from sglang.srt.mem_cache.radix_cache import RadixKey

logger = logging.getLogger(__name__)

FCFS_FALLBACK_THRESHOLD = 128


def _clear_prefix_state(waiting_queue: List[Req]) -> None:
    """Reset prefix metadata to defaults so stale values from a previous
    scheduling round cannot leak into a fallback FCFS path."""
    for r in waiting_queue:
        r.prefix_indices = torch.empty((0,), dtype=torch.int64)
        r.last_node = None
        r.last_host_node = None
        r.host_hit_length = 0


class SimpleLPMScheduler:
    """Our simplified LPM reproduction.

    Compared to the original SchedulePolicy LPM path this intentionally omits:
      - in-batch prefix caching (waiting_queue_radix_tree / deprioritization)
      - DFS-weight logic
      - priority-scheduling interaction
    It keeps only the core idea: query the radix tree for each request,
    then sort longest-cached-prefix-first.
    """

    def __init__(self, tree_cache: BasePrefixCache):
        self.tree_cache = tree_cache

    def calc_priority(self, waiting_queue: List[Req]) -> bool:
        """Reorder *waiting_queue* in-place by cached prefix length (descending).

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

        waiting_queue.sort(key=lambda r: -len(r.prefix_indices))

        top = waiting_queue[:5]
        logger.warning(
            "[CUSTOM SCHED] sorted top=%s",
            [(r.rid, len(r.prefix_indices)) for r in top],
        )
        return True

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
