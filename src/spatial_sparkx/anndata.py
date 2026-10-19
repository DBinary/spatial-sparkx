"""AnnData integration without making AnnData a core package dependency."""

import numpy as np

from .core import sparkx


def sparkx_adata(adata, *, spatial_key="spatial", layer=None, covariates=None,
                 key_added="sparkx", inplace=True, **kwargs):
    """Run SPARK-X on AnnData and optionally add its results to the object.

    ``adata.X`` or ``adata.layers[layer]`` supplies observations-by-genes
    expression. Coordinates come from ``adata.obsm[spatial_key]``. Covariates
    can be a numeric design matrix or a list of numeric ``adata.obs`` columns;
    supply an intercept explicitly when required. Results are always returned.

    With inplace=True, writes ``<key>_pvalue``, ``<key>_padj``, and
    ``<key>_tested`` in var; kernel arrays in varm; and metadata in uns.
    Excluded all-zero genes have NaN p-values and tested=False.
    """
    if not isinstance(key_added, str) or not key_added:
        raise ValueError("key_added must be a nonempty string")
    if getattr(adata, "isbacked", False):
        raise ValueError("backed AnnData is not supported; load the selected matrix into memory first")
    if inplace and getattr(adata, "is_view", False):
        raise ValueError("inplace writes to an AnnData view are ambiguous; call adata.copy() first")
    if spatial_key not in adata.obsm:
        raise KeyError(f"adata.obsm does not contain {spatial_key!r}")
    for key in ("gene_axis", "gene_names"):
        if key in kwargs:
            raise ValueError(f"{key} is determined by AnnData and must not be supplied")
    if isinstance(covariates, (list, tuple)) and all(isinstance(x, str) for x in covariates):
        covariates = adata.obs.loc[:, list(covariates)].to_numpy(dtype=float)
    expression = adata.X if layer is None else adata.layers[layer]
    result = sparkx(expression, adata.obsm[spatial_key], covariates,
                    gene_names=adata.var_names, **kwargs)
    if inplace:
        shape = (adata.n_vars, result.stats.shape[1])
        stats = np.full(shape, np.nan)
        pvalues = np.full(shape, np.nan)
        stats[result.gene_mask] = result.stats.to_numpy()
        pvalues[result.gene_mask] = result.res_stest.to_numpy()
        combined = np.full((adata.n_vars, 2), np.nan)
        combined[result.gene_mask] = result.res_mtest.to_numpy()
        adata.var[f"{key_added}_pvalue"] = combined[:, 0]
        adata.var[f"{key_added}_padj"] = combined[:, 1]
        adata.var[f"{key_added}_tested"] = result.gene_mask
        adata.varm[f"{key_added}_stats"] = stats
        adata.varm[f"{key_added}_pvalues"] = pvalues
        adata.uns[key_added] = {
            "kernel_names": list(result.stats.columns),
            "kernel_ranks": result.kernel_ranks.to_numpy(),
            "n_observations_tested": int(result.cell_mask.sum()),
            "n_genes_tested": int(result.gene_mask.sum()),
            "option": result.option,
            "rank_policy": result.rank_policy,
            "degenerate_policy": result.degenerate_policy,
            "multiple_testing": "Benjamini-Yekutieli",
            "implementation": "spatial-sparkx 0.1.0; native Python",
        }
    return result
