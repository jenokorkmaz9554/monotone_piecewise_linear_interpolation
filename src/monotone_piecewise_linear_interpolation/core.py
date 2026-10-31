"""Core implementation of monotone piecewise linear interpolation.

The interpolant is the standard piecewise linear function through the data
points.  The monotonicity-preserving property comes from the choice of the
slopes on each interval: we use the Fritsch--Carlson limiter, which caps each
interval slope by three times the smaller of the two adjacent secant slopes.
This guarantees that the interpolant is monotone on any interval where the
data are monotone, while still passing exactly through every data point.

Why this choice?  A plain piecewise linear interpolant is already monotone
on monotone data, but it can introduce overshoot when the data have a sharp
change in slope.  The Fritsch--Carlson limiter removes that overshoot without
sacrificing interpolation accuracy, and it is simple enough to implement with
no dependencies.
"""

from __future__ import annotations

import math
from typing import List, Sequence, Tuple


class MonotoneInterpolant:
    """A monotone-preserving piecewise linear interpolant.

    The interpolant is defined by a strictly increasing sequence of x values
    and a corresponding sequence of y values.  Evaluation outside the range
    of the data is not supported and raises :class:`ValueError`.

    Parameters
    ----------
    xs:
        Strictly increasing x coordinates.  Must contain at least two points.
    ys:
        y coordinates, same length as ``xs``.
    """

    __slots__ = ("_xs", "_ys", "_slopes")

    def __init__(self, xs: Sequence[float], ys: Sequence[float]) -> None:
        if len(xs) != len(ys):
            raise ValueError("xs and ys must have the same length")
        if len(xs) < 2:
            raise ValueError("at least two data points are required")

        self._xs: Tuple[float, ...] = tuple(float(x) for x in xs)
        self._ys: Tuple[float, ...] = tuple(float(y) for y in ys)

        for i in range(1, len(self._xs)):
            if not self._xs[i] > self._xs[i - 1]:
                raise ValueError("xs must be strictly increasing")

        self._slopes: Tuple[float, ...] = self._compute_slopes()

    def _compute_slopes(self) -> Tuple[float, ...]:
        """Compute the limited slopes for each interval.

        The raw secant slope on interval i is (y[i+1]-y[i])/(x[i+1]-x[i]).
        The Fritsch--Carlson limiter sets the slope of interval i to
        min(|secant_i|, 3*|secant_{i-1}|, 3*|secant_{i+1}|) with the sign of
        secant_i.  At the boundaries we only have one neighbouring secant.
        """
        n = len(self._xs)
        secants: List[float] = []
        for i in range(n - 1):
            dx = self._xs[i + 1] - self._xs[i]
            dy = self._ys[i + 1] - self._ys[i]
            secants.append(dy / dx)

        slopes: List[float] = []
        for i in range(n - 1):
            s = secants[i]
            if s == 0.0:
                slopes.append(0.0)
                continue
            limit = abs(s)
            if i > 0:
                limit = min(limit, 3.0 * abs(secants[i - 1]))
            if i < n - 2:
                limit = min(limit, 3.0 * abs(secants[i + 1]))
            slopes.append(math.copysign(limit, s))
        return tuple(slopes)

    def __call__(self, x: float) -> float:
        """Evaluate the interpolant at ``x``.

        Raises
        ------
        ValueError
            If ``x`` lies outside the closed interval spanned by the data.
        """
        x = float(x)
        if x < self._xs[0] or x > self._xs[-1]:
            raise ValueError("x is outside the interpolation range")

        # Binary search for the interval containing x.
        lo, hi = 0, len(self._xs) - 1
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if self._xs[mid] <= x:
                lo = mid
            else:
                hi = mid

        i = lo
        if x == self._xs[i]:
            return self._ys[i]
        if x == self._xs[i + 1]:
            return self._ys[i + 1]

        dx = self._xs[i + 1] - self._xs[i]
        t = (x - self._xs[i]) / dx
        # Linear interpolation with the limited slope.
        return self._ys[i] + t * dx * self._slopes[i]

    @property
    def xs(self) -> Tuple[float, ...]:
        """The x coordinates of the data points."""
        return self._xs

    @property
    def ys(self) -> Tuple[float, ...]:
        """The y coordinates of the data points."""
        return self._ys

    @property
    def slopes(self) -> Tuple[float, ...]:
        """The limited slopes used on each interval."""
        return self._slopes


def build_interpolant(xs: Sequence[float], ys: Sequence[float]) -> MonotoneInterpolant:
    """Convenience constructor for :class:`MonotoneInterpolant`.

    This exists so that callers can build an interpolant without importing
    the class directly, and so that the public API has a single obvious
    entry point.
    """
    return MonotoneInterpolant(xs, ys)
