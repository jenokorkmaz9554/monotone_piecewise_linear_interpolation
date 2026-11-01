# monotone_piecewise_linear_interpolation

Construct a monotone-preserving piecewise linear interpolant through data points that respects monotonicity of the source data.

```python
from monotone_piecewise_linear_interpolation import build_interpolant

interp = build_interpolant([0.0, 1.0, 2.0, 3.0], [0.0, 1.0, 1.5, 1.6])
value = interp(1.5)  # 1.25
```

The interpolant is callable, exposes the original data as ``xs`` and ``ys``,
and exposes the limited interval slopes as ``slopes``.  Evaluation outside the
range of the data raises ``ValueError``.

## Why this library exists

A plain piecewise linear interpolant is monotone on monotone data, but it can
overshoot when the data have a sharp change in slope.  The Fritsch--Carlson
limiter caps each interval slope by three times the smaller adjacent secant
slope, which removes the overshoot while still passing exactly through every
data point.  The trade-off is that the interpolant is no longer the plain
linear interpolant on non-monotone data: the limited slopes can differ from the
secant slopes, so the function is piecewise linear but not the simple
connect-the-dots line.  This is the standard choice for monotone interpolation
and is what the tests here verify.

## Edge cases

- ``xs`` must be strictly increasing; otherwise ``ValueError`` is raised.
- At least two data points are required.
- ``xs`` and ``ys`` must have the same length.
- Evaluation outside the closed interval spanned by ``xs`` raises ``ValueError``.
- Flat data (all ``ys`` equal) produces a flat interpolant.
