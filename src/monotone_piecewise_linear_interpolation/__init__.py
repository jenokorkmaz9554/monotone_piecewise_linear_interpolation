"""Monotone piecewise linear interpolation.

Exports the public API for constructing a monotone-preserving piecewise
linear interpolant and evaluating it at arbitrary points.
"""

from .core import MonotoneInterpolant, build_interpolant

__all__ = ["MonotoneInterpolant", "build_interpolant"]
