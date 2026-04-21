# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Radix attention implementation router."""

from __future__ import annotations

import logging
from enum import Enum
from typing import TYPE_CHECKING, Optional

from torch import nn

if TYPE_CHECKING:
    from sglang.srt.layers.quantization.base_config import QuantizationConfig

logger = logging.getLogger(__name__)


class AttentionType(Enum):
    """
    Attention type.
    Use string to be compatible with `torch.compile`.
    """

    # Decoder attention between previous layer Q/K/V
    DECODER = "decoder"
    # Decoder bidirectional attention between image tokens
    DECODER_BIDIRECTIONAL = "decoder_bidirectional"
    # Encoder attention between previous layer Q/K/V
    ENCODER_ONLY = "encoder_only"


def _get_global_radix_attention_impl() -> str:
    try:
        from sglang.srt.server_args import get_global_server_args

        return getattr(get_global_server_args(), "radix_cache_impl", "vanilla")
    except Exception:
        return "vanilla"


def _resolve_radix_attention_impl() -> str:
    impl_name = _get_global_radix_attention_impl()
    impl_name = impl_name.lower()
    if impl_name not in ("vanilla", "custom"):
        raise ValueError(
            f"Unsupported radix attention implementation: {impl_name}. "
            "Expected 'vanilla' or 'custom'."
        )
    return impl_name


def _load_radix_attention_impl(impl_name: str):
    if impl_name == "custom":
        print("we are in custom radix attention", flush=True)
        from sglang.srt.layers.custom_radix_attention import (
            RadixAttention as CustomRadixAttention,
        )

        return CustomRadixAttention

    print("we are in else vanilla radix attention", flush=True)
    from sglang.srt.layers.vanilla_radix_attention import (
        RadixAttention as VanillaRadixAttention,
    )

    return VanillaRadixAttention


class RadixAttention(nn.Module):
    """Route the original RadixAttention constructor to the selected implementation.

    Many model files import ``RadixAttention`` from this module. Keeping this
    public class as the entrypoint lets those imports stay unchanged while the
    selected implementation lives in either ``vanilla_radix_attention.py`` or
    ``custom_radix_attention.py``.
    """

    def __new__(
        cls,
        num_heads: int,
        head_dim: int,
        scaling: float,
        num_kv_heads: int,
        layer_id: int,
        logit_cap: float = 0.0,
        v_head_dim: int = -1,
        sliding_window_size: int = -1,
        is_cross_attention: bool = False,
        pos_encoding_mode: str = "NONE",
        logit_capping_method: str = "tanh",
        quant_config: Optional[QuantizationConfig] = None,
        attn_type: AttentionType = AttentionType.DECODER,
        use_irope: bool = False,
        prefix: str = "",
    ):
        impl_name = _resolve_radix_attention_impl()
        impl_cls = _load_radix_attention_impl(impl_name)

        logger.info("Using %s radix attention implementation.", impl_name)
        return impl_cls(
            num_heads=num_heads,
            head_dim=head_dim,
            scaling=scaling,
            num_kv_heads=num_kv_heads,
            layer_id=layer_id,
            logit_cap=logit_cap,
            v_head_dim=v_head_dim,
            sliding_window_size=sliding_window_size,
            is_cross_attention=is_cross_attention,
            pos_encoding_mode=pos_encoding_mode,
            logit_capping_method=logit_capping_method,
            quant_config=quant_config,
            attn_type=attn_type,
            use_irope=use_irope,
            prefix=prefix,
        )
