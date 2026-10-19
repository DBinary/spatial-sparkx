"""Coordinate transformations from SPARK/R/sparkx.R (quantile type 7)."""

from collections import OrderedDict

import numpy as np


def _coordinates(coordinates):
    if np.iscomplexobj(coordinates):
        raise ValueError("coordinates must be real-valued")
    loc = np.asarray(coordinates, dtype=np.float64)
    if loc.ndim == 1:
        loc = loc[:, None]
    if loc.ndim != 2 or loc.shape[1] == 0 or loc.shape[0] < 2:
        raise ValueError("coordinates must have shape (n_observations >= 2, n_dimensions >= 1)")
    if not np.isfinite(loc).all():
        raise ValueError("coordinates must contain only finite real values")
    return loc


def transform_coordinates(coordinates, kernel_index, transform="gaussian"):
    """Apply one of the five Gaussian or cosine coordinate transformations.

    ``kernel_index`` is 1-based, matching R's ``transloc_func``. Each coordinate
    is centered before taking the 20%, 40%, 60%, 80%, or 100% quantile of its
    absolute values. These are feature transformations, not an n-by-n RBF kernel.
    """
    if (isinstance(kernel_index, bool) or not isinstance(kernel_index, (int, np.integer))
            or kernel_index not in (1, 2, 3, 4, 5)):
        raise ValueError("kernel_index must be an integer from 1 to 5")
    if transform not in ("gaussian", "cosine"):
        raise ValueError("transform must be 'gaussian' or 'cosine'")
    loc = _coordinates(coordinates)
    centered = loc - loc.mean(axis=0)
    bandwidth = np.quantile(np.abs(centered), kernel_index / 5, axis=0, method="linear")
    constant = np.all(centered == 0, axis=0)
    if np.any((bandwidth == 0) & ~constant):
        raise ValueError(
            f"{transform} kernel {kernel_index} has a zero quantile bandwidth; "
            "a nonconstant coordinate has too many values at its mean"
        )
    # A constant coordinate contributes no spatial information after centering.
    safe_bandwidth = np.where(constant, 1, bandwidth)
    standardized = centered / safe_bandwidth
    if transform == "gaussian":
        return np.exp(-0.5 * standardized**2)
    return np.cos(2 * np.pi * standardized)


def spatial_kernels(coordinates, option="mixture"):
    """Return the projection features, or all 11 upstream kernel features."""
    if option not in ("single", "mixture"):
        raise ValueError("option must be 'single' or 'mixture'")
    loc = _coordinates(coordinates)
    result = OrderedDict(projection=loc.copy())
    if option == "mixture":
        for kind, prefix in (("gaussian", "gaus"), ("cosine", "cos")):
            for index in range(1, 6):
                result[f"{prefix}{index}"] = transform_coordinates(loc, index, kind)
    return result
