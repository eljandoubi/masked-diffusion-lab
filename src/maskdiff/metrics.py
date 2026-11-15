"""Divergences and fits used by the studies."""

import numpy as np


def kl(p: np.ndarray, q: np.ndarray) -> float:
    """KL(p || q) in nats; q must be positive wherever p is."""
    p, q = p.reshape(-1), q.reshape(-1)
    keep = p > 0
    return float(np.sum(p[keep] * np.log1p((p[keep] - q[keep]) / q[keep])))


def loglog_slope(
    x: np.ndarray, y: np.ndarray, lo: float, hi: float
) -> tuple[float, float]:
    """Least-squares slope of log y on log x over lo <= x <= hi, and the fit R^2."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    keep = (x >= lo) & (x <= hi) & (y > 0)
    if keep.sum() < 2:
        raise ValueError("need at least two points in the fit range")
    lx, ly = np.log(x[keep]), np.log(y[keep])
    slope, intercept = np.polyfit(lx, ly, 1)
    resid = ly - (slope * lx + intercept)
    spread = ((ly - ly.mean()) ** 2).sum()
    r2 = 1.0 - (resid**2).sum() / spread if spread > 0 else 1.0
    return float(slope), float(r2)
