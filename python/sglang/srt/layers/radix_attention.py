from __future__ import annotations

from torch import nn

from .vanilla_radix_attention import RadixAttention as VanillaRadixAttention
from .custom_radix_attention import RadixAttention as CustomRadixAttention


class RadixAttention(nn.Module):
    def __new__(cls, *args, **kwargs):
        radix_cache_impl = kwargs.pop("radix_cache_impl", "vanilla")

        if radix_cache_impl == "custom":
            print("Routing to Custom Radix Attention")
            return CustomRadixAttention(*args, **kwargs)
        else:
            print("Routing to Vanilla Radix Attention")
            return VanillaRadixAttention(*args, **kwargs)
