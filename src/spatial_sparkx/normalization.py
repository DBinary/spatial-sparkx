"""Optional upstream NormalizeVSTfast helper, expressed without R's nls/lm."""

import numpy as np
from scipy import linalg, sparse


def normalize_vst(counts, *, gene_axis=1, return_phi=False):
    """Port SPARK::NormalizeVSTfast, including log-library-size regression.

    Input and output use observations by genes unless gene_axis=0. This helper
    returns a dense residual matrix; it is optional and is not called by sparkx.
    The upstream nls model is linear in its only parameter phi, so its least-
    squares solution is calculated analytically. Variance uses n-1, matching
    Armadillo var(..., norm_type=0) in the original sparse C++ helper.
    """
    if isinstance(gene_axis, bool) or gene_axis not in (0, 1):
        raise ValueError("gene_axis must be 0 or 1")
    if np.iscomplexobj(counts.data if sparse.issparse(counts) else counts):
        raise ValueError("counts must be real-valued")
    if sparse.issparse(counts):
        matrix = sparse.csc_matrix(counts, dtype=float, copy=True)
        matrix.sum_duplicates()
        values = matrix.data
    else:
        matrix = np.asarray(counts, dtype=float)
        values = matrix
    if matrix.ndim != 2:
        raise ValueError("counts must be a two-dimensional matrix")
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("normalize_vst requires finite, nonnegative counts")
    if gene_axis == 0:
        matrix = matrix.T
    n, g = matrix.shape
    if n < 2 or g == 0:
        raise ValueError("at least two observations and one gene are required")
    totals = np.asarray(matrix.sum(axis=1)).ravel()
    if np.any(totals <= 0):
        raise ValueError("normalize_vst requires positive library sizes for all observations")
    mean = np.asarray(matrix.mean(axis=0)).ravel()
    squared = matrix.multiply(matrix) if sparse.issparse(matrix) else matrix * matrix
    variance = (np.asarray(squared.sum(axis=0)).ravel() - n * mean**2) / (n - 1)
    predictor = mean**2
    phi = float(predictor @ (variance - mean) / (predictor @ predictor))
    if not np.isfinite(phi) or phi <= 0:
        raise ValueError("estimated overdispersion phi must be positive for NormalizeVSTfast")
    dense = matrix.toarray() if sparse.issparse(matrix) else matrix
    transformed = np.log(dense + 1 / (2 * phi))
    design = np.column_stack([np.ones(n), np.log(totals)])
    coefficients = linalg.lstsq(design, transformed, check_finite=False)[0]
    residuals = transformed - design @ coefficients
    output = residuals.T if gene_axis == 0 else residuals
    return (output, phi) if return_phi else output
