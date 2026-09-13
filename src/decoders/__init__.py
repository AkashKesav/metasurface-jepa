"""Decoders for the unified 192-D architecture (architecture_v5.md §4).

OccupancyDecoder : predicted occupancy tokens -> occupancy logits (FiLM-conditioned)
ScalarDecoder    : predicted scalar-summary latent -> (l, h, r) heads
"""

from .occupancy_decoder import OccupancyDecoder
from .scalar_decoder import ScalarDecoder

__all__ = ["OccupancyDecoder", "ScalarDecoder"]
