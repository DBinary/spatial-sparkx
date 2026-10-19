"""Sparse, batched implementation of the SPARK-X projection-kernel test."""

from collections.abc import Mapping
from contextlib import nullcontext
from dataclasses import dataclass
import logging
import warnings

import numpy as np
import pandas as pd
from scipy import linalg, sparse, special
from threadpoolctl import threadpool_limits

from .kernels import _coordinates, spatial_kernels
from .statistics import acat, by_adjust

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SparkXResult(Mapping):
    """Upstream-compatible result tables plus input-filtering metadata."""

    stats: pd.DataFrame
    res_stest: pd.DataFrame
    res_mtest: pd.DataFrame
    cell_mask: np.ndarray
    gene_mask: np.ndarray
    kernel_ranks: pd.Series
    degenerate_genes: np.ndarray
    option: str
    rank_policy: str
    degenerate_policy: str

    def __getitem__(self, key):
        if key not in ("stats", "res_stest", "res_mtest"):
            raise KeyError(key)
        return getattr(self, key)

    def __iter__(self):
        return iter(("stats", "res_stest", "res_mtest"))

    def __len__(self):
        return 3


def _basis(matrix, name, rank_policy):
    """Find a scale-invariant orthonormal basis without forming a projector."""
    norms = np.linalg.norm(matrix, axis=0)
    normalized = matrix / np.where(norms > 0, norms, 1)
    q, r, _ = linalg.qr(normalized, mode="economic", pivoting=True, check_finite=False)
    diagonal = np.abs(np.diag(r))
    tolerance = np.finfo(float).eps * max(matrix.shape) * (diagonal.max(initial=0))
    rank = int(np.count_nonzero(diagonal > tolerance))
    if rank < matrix.shape[1]:
        message = f"{name} has rank {rank}, but has {matrix.shape[1]} columns"
        if rank_policy == "raise":
            raise ValueError(message + "; use rank_policy='drop' to test its identifiable subspace")
        warnings.warn(message + "; dependent dimensions were dropped", RuntimeWarning, stacklevel=3)
    return q[:, :rank]


def _prepare(counts, coordinates, covariates, gene_axis, gene_names):
    if isinstance(gene_axis, bool) or gene_axis not in (0, 1):
        raise ValueError("gene_axis must be 1 (observations by genes) or 0 (R orientation)")
    labels = None
    if isinstance(counts, pd.DataFrame):
        labels = counts.columns if gene_axis == 1 else counts.index
        cell_names = counts.index if gene_axis == 1 else counts.columns
        for name, value in (("coordinates", coordinates), ("covariates", covariates)):
            if isinstance(value, (pd.DataFrame, pd.Series)) and not cell_names.equals(value.index):
                raise ValueError(f"{name} index must match count observation labels and order")
    if sparse.issparse(counts):
        if np.iscomplexobj(counts.data):
            raise ValueError("counts must be real-valued")
        matrix = sparse.csc_matrix(counts, dtype=np.float64, copy=True)
        matrix.sum_duplicates()
        matrix.eliminate_zeros()
        if gene_axis == 0:
            matrix = matrix.T.tocsc()
        finite = np.isfinite(matrix.data).all()
    else:
        if np.iscomplexobj(counts):
            raise ValueError("counts must be real-valued")
        matrix = np.asarray(counts, dtype=np.float64)
        if matrix.ndim == 1:
            matrix = matrix[:, None] if gene_axis == 1 else matrix[None, :]
        if matrix.ndim != 2:
            raise ValueError("counts must be a two-dimensional matrix or a single gene vector")
        if gene_axis == 0:
            matrix = matrix.T
        finite = np.isfinite(matrix).all()
    if not finite:
        raise ValueError("counts must contain only finite real values")
    if matrix.shape[0] < 2 or matrix.shape[1] == 0:
        raise ValueError("counts must contain at least two observations and one gene")
    loc = _coordinates(coordinates)
    if loc.shape[0] != matrix.shape[0]:
        raise ValueError("coordinates rows must match the number of count observations")
    if covariates is not None and np.iscomplexobj(covariates):
        raise ValueError("covariates must be real-valued")
    cov = None if covariates is None else np.asarray(covariates, dtype=float)
    if cov is not None:
        if cov.ndim == 1:
            cov = cov[:, None]
        if cov.ndim != 2 or cov.shape[0] != matrix.shape[0] or cov.shape[1] == 0:
            raise ValueError("covariates must have shape (n_observations, n_covariates >= 1)")
        if not np.isfinite(cov).all():
            raise ValueError("covariates must contain only finite real values")
    labels = labels if gene_names is None else gene_names
    if labels is None:
        labels = pd.RangeIndex(matrix.shape[1], name="gene")
    else:
        labels = pd.Index(labels)
        if len(labels) != matrix.shape[1]:
            raise ValueError("gene_names length must match the original number of genes")
        labels = labels.fillna("NAgene")
    # Follow the original R filtering literally, including signed cancellation.
    cell_total = np.asarray(matrix.sum(axis=1)).ravel()
    if not np.isfinite(cell_total).all():
        raise ValueError("counts are too large to sum safely in float64")
    cell_mask = cell_total != 0
    if np.count_nonzero(cell_mask) < 2:
        raise ValueError("fewer than two nonzero observations remain")
    if not cell_mask.all():
        matrix = matrix[cell_mask, :]
        loc = loc[cell_mask]
        if cov is not None:
            cov = cov[cell_mask]
    square = matrix.multiply(matrix) if sparse.issparse(matrix) else matrix * matrix
    norm_squared = np.asarray(square.sum(axis=0)).ravel()
    if not np.isfinite(norm_squared).all():
        raise ValueError("counts are too large to sum squared expression safely in float64")
    gene_mask = np.asarray(matrix.sum(axis=0)).ravel() != 0
    if not gene_mask.any():
        raise ValueError("no nonzero genes remain")
    if not gene_mask.all():
        matrix = matrix[:, gene_mask]
    if np.any(norm_squared[gene_mask] == 0):
        raise ValueError("nonzero expression is too small to square safely in float64")
    return matrix, loc, cov, labels[gene_mask], cell_mask, gene_mask, norm_squared[gene_mask]


def _sparkx(counts, coordinates, covariates, option, gene_axis, gene_names,
            batch_size, rank_policy, backend, verbose, degenerate_policy):
    matrix, loc, cov, names, cell_mask, gene_mask, norm_squared = _prepare(
        counts, coordinates, covariates, gene_axis, gene_names
    )
    n, n_genes = matrix.shape
    if cov is None:
        null_basis = np.full((n, 1), 1 / np.sqrt(n))
    else:
        null_basis = _basis(cov, "covariates", rank_policy)
        if null_basis.shape[1] >= n:
            raise ValueError("covariates leave no residual observation space")
    features = spatial_kernels(loc, option)
    bases = []
    for name, value in features.items():
        residual = value - null_basis @ (null_basis.T @ value)
        # Repeating residualization suppresses roundoff in the nuisance space.
        residual -= null_basis @ (null_basis.T @ residual)
        # Confounded / constant columns must not be normalized into pure noise.
        original_norm = np.linalg.norm(value, axis=0)
        residual_norm = np.linalg.norm(residual, axis=0)
        lost = residual_norm <= 16 * np.finfo(float).eps * np.maximum(original_norm, 1)
        residual[:, lost] = 0
        bases.append(_basis(residual, f"kernel {name}", rank_policy))
    ranks = np.array([basis.shape[1] for basis in bases], dtype=int)
    null_rank = null_basis.shape[1]
    all_bases = np.column_stack([null_basis, *bases])
    if backend == "cupy":
        try:
            import cupy as cp
            import cupyx.scipy.sparse as cpsparse
            device_bases = cp.asarray(all_bases)
        except (ImportError, RuntimeError, OSError) as error:
            raise RuntimeError("backend='cupy' requires a working CUDA/CuPy installation") from error
    statistics = np.zeros((n_genes, len(features)))
    pvalues = np.ones_like(statistics)
    degenerate = np.zeros(n_genes, dtype=bool)
    for start in range(0, n_genes, batch_size):
        stop = min(start + batch_size, n_genes)
        block = matrix[:, start:stop]
        if backend == "cupy":
            device_counts = cpsparse.csc_matrix(block) if sparse.issparse(block) else cp.asarray(block)
            projections = cp.asnumpy(device_counts.T @ device_bases)
            del device_counts
        else:
            projections = np.asarray(block.T @ all_bases)
        norm = norm_squared[start:stop]
        null_energy = np.einsum("ij,ij->i", projections[:, :null_rank], projections[:, :null_rank])
        ratio = np.maximum(1 - null_energy / norm, 0)
        bad = ratio <= 32 * np.finfo(float).eps
        if bad.any() and degenerate_policy == "raise":
            affected = ", ".join(map(str, names[start:stop][bad][:5]))
            raise ValueError(
                f"zero or numerically negligible residual expression variance for genes: {affected}; "
                "the null distribution is degenerate. Remove these genes, or explicitly request "
                "degenerate_policy='conservative' to assign p=1"
            )
        degenerate[start:stop] = bad
        offset = null_rank
        for kernel, rank in enumerate(ranks):
            if rank:
                projected = projections[:, offset:offset + rank]
                score = n * np.einsum("ij,ij->i", projected, projected) / norm
                score[bad] = 0
                statistics[start:stop, kernel] = score
                active = ~bad
                tail = np.ones(stop - start)
                tail[active] = special.gammaincc(rank / 2, score[active] / (2 * ratio[active]))
                pvalues[start:stop, kernel] = tail
            offset += rank
        if verbose:
            logger.info("SPARK-X: tested %d / %d genes", stop, n_genes)
    if degenerate.any():
        warnings.warn(
            f"{degenerate.sum()} genes have no numerically identifiable expression residual; "
            "their statistics are 0 and p-values are 1",
            RuntimeWarning, stacklevel=3,
        )
    combined = acat(pvalues, axis=1)
    table = pd.DataFrame({"combinedPval": combined, "adjustedPval": by_adjust(combined)}, index=names)
    return SparkXResult(
        stats=pd.DataFrame(statistics, index=names, columns=list(features)),
        res_stest=pd.DataFrame(pvalues, index=names, columns=list(features)),
        res_mtest=table, cell_mask=cell_mask, gene_mask=gene_mask,
        kernel_ranks=pd.Series(ranks, index=list(features), name="rank"),
        degenerate_genes=degenerate, option=option, rank_policy=rank_policy,
        degenerate_policy=degenerate_policy,
    )


def sparkx(counts, coordinates, covariates=None, *, option="mixture", gene_axis=1,
           gene_names=None, batch_size=1024, rank_policy="raise", backend="numpy",
           n_threads=None, verbose=False, degenerate_policy="raise"):
    """Test spatial expression using SPARK-X with NumPy/SciPy or optional CuPy.

    Parameters
    ----------
    counts : array, scipy sparse matrix, or pandas DataFrame
        Observations by genes by default. Raw counts are recommended. Finite,
        signed transformed expression is also supported. Observations/genes
        with zero sums are excluded exactly as in R; for nonnegative counts
        these are the all-zero observations/genes. Masks refer to original input.
    coordinates : array, shape (n_observations, n_dimensions)
        Spatial coordinates in the same observation order as counts.
    covariates : array, optional
        Nuisance design matrix. As in R, an intercept is NOT automatically
        appended when covariates are supplied: include a column of ones yourself.
        Without covariates, expression and coordinate kernels are centered.
    option : {'mixture', 'single'}
        Eleven kernels or the projection kernel alone.
    gene_axis : {1, 0}
        Set to 0 for the original R genes-by-observations orientation.
    batch_size : int
        Number of genes per matrix multiplication / GPU transfer.
    rank_policy : {'raise', 'drop'}
        Reject rank-deficient designs (upstream behavior), or explicitly test
        their identifiable subspaces, with rank-zero kernels assigned p=1.
    backend : {'numpy', 'cupy'}
        CPU by default. CuPy is imported only when explicitly requested.
    n_threads : int or None
        Limit CPU BLAS threads; None leaves the existing BLAS policy in place.
    degenerate_policy : {'raise', 'conservative'}
        Reject zero residual expression variance by default. Explicitly select
        'conservative' to assign such genes p=1; that extension is not claimed
        to reproduce undefined or unstable upstream results for these genes.

    Notes
    -----
    For full-rank residualized coordinates Z, Z.T @ Z @ inv(Z.T @ Z) is I.
    Therefore the upstream chi-square mixture has identical nonzero weights
    lambda_G. Its survival probability is exactly chi2.sf(stat / lambda_G, d).
    SciPy's incomplete gamma evaluates this distribution directly, avoiding
    Davies integration error and the need for an R/CompQuadForm dependency.
    No observation-by-observation kernel or nuisance projector is constructed.
    """
    if option not in ("single", "mixture"):
        raise ValueError("option must be 'single' or 'mixture'")
    if rank_policy not in ("raise", "drop"):
        raise ValueError("rank_policy must be 'raise' or 'drop'")
    if backend not in ("numpy", "cupy"):
        raise ValueError("backend must be 'numpy' or 'cupy'")
    if degenerate_policy not in ("raise", "conservative"):
        raise ValueError("degenerate_policy must be 'raise' or 'conservative'")
    if isinstance(batch_size, bool) or not isinstance(batch_size, (int, np.integer)) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    if n_threads is not None and (
        isinstance(n_threads, bool) or not isinstance(n_threads, (int, np.integer)) or n_threads < 1
    ):
        raise ValueError("n_threads must be None or a positive integer")
    context = nullcontext() if n_threads is None else threadpool_limits(limits=int(n_threads), user_api="blas")
    with context:
        return _sparkx(counts, coordinates, covariates, option, gene_axis, gene_names,
                       int(batch_size), rank_policy, backend, verbose, degenerate_policy)


def sparkx_r(count_in, locus_in, X_in=None, numCores=1, option="mixture", verbose=False, **kwargs):
    """R-style argument names and genes-by-observations orientation; native Python."""
    return sparkx(count_in, locus_in, X_in, option=option, gene_axis=0,
                  n_threads=numCores, verbose=verbose, **kwargs)
