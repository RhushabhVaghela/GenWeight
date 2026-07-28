"""Compact neural generators for weight matrices."""

from .block import BlockWeightGenerator
from .coordinate import CoordinateWeightGenerator

__all__ = ["BlockWeightGenerator", "CoordinateWeightGenerator"]
