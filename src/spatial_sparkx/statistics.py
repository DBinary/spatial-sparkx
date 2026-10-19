"""Multiple-testing procedures used by the upstream SPARK-X implementation."""

import numpy as np


def acat(pvalues, weights=None, axis=-1):
    """Aggregate p-values using the Cauchy combination test.

    Endpoint handling matches SPARK::ACAT: any exact 0 gives 0, any exact
    1 gives 1, and the presence of both is an error. Nonnegative weights
    are normalized to sum to one. Calculation is rescaled to preserve
    extremely small, nonzero p-values without overflowing 1 / p.
    """
    p = np.asarray(pvalues, dtype=np.float64)
    if p.ndim == 0:
        p = p.reshape(1)
    p = np.moveaxis(p, axis, -1)
    if p.shape[-1] == 0:
        raise ValueError("pvalues must contain at least one test")
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("pvalues must be finite numbers between 0 and 1")
    w = np.ones(p.shape[-1]) if weights is None else np.asarray(weights, dtype=float)
    if w.ndim != 1 or w.size != p.shape[-1]:
        raise ValueError("weights must be one-dimensional and match the test axis")
    if not np.isfinite(w).all() or np.any(w < 0) or w.max() <= 0:
        raise ValueError("weights must be finite, nonnegative, and not all zero")
    w = w / w.max()
    w /= w.sum()
    shape = p.shape[:-1]
    flat = p.reshape(-1, p.shape[-1])
    zero = (flat == 0).any(axis=1)
    one = (flat == 1).any(axis=1)
    if np.any(zero & one):
        raise ValueError("Cannot combine both exact 0 and exact 1 p-values (SPARK ACAT convention)")
    out = np.empty(flat.shape[0])
    out[zero] = 0
    out[one] = 1
    interior = ~(zero | one)
    if interior.any():
        values = flat[interior][:, w > 0]
        active_weights = w[w > 0]
        scale = np.minimum(values, 1 - values).min(axis=1)
        small = values < 1e-16
        terms = np.empty_like(values)
        np.divide(scale[:, None], values, out=terms)
        terms /= np.pi
        # cot(pi*p) == tan(pi*(0.5-p)); choose the endpoint nearest zero
        # to avoid cancellation in 0.5-p for p around 1e-15.
        distance = np.minimum(values, 1 - values)
        with np.errstate(over="ignore", divide="ignore"):
            regular = scale[:, None] / np.tan(np.pi * distance)
        regular[values > 0.5] *= -1
        regular[values == 0.5] = 0
        terms[~small] = regular[~small]
        statistic_scaled = terms @ active_weights
        out[interior] = np.arctan2(scale, statistic_scaled) / np.pi
    out = np.clip(out.reshape(shape), 0, 1)
    return float(out) if out.ndim == 0 else out


def by_adjust(pvalues):
    """Benjamini–Yekutieli adjusted p-values, preserving the original order."""
    p = np.asarray(pvalues, dtype=np.float64)
    if p.ndim != 1 or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("pvalues must be a finite one-dimensional array in [0, 1]")
    if p.size == 0:
        return p.copy()
    order = np.argsort(p, kind="stable")
    ranks = np.arange(1, p.size + 1, dtype=float)
    adjusted = p[order] * (p.size * np.sum(1 / ranks)) / ranks
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result = np.empty_like(adjusted)
    result[order] = np.minimum(adjusted, 1)
    return result
