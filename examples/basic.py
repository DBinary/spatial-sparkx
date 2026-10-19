"""Run a small, reproducible SPARK-X example without downloading any data."""

import numpy as np
from scipy import sparse

from spatial_sparkx import sparkx

rng = np.random.default_rng(2026)
coordinates = rng.uniform(-5, 5, size=(600, 2))
counts = rng.poisson(2, size=(600, 100)).astype(float)
counts[:, 0] = rng.poisson(0.5 + 10 * np.exp(-coordinates[:, 0]**2 / 2))

result = sparkx(
    sparse.csr_matrix(counts), coordinates,
    gene_names=[f"gene_{i}" for i in range(counts.shape[1])], n_threads=4,
)
print(result.res_mtest.sort_values("adjustedPval").head(10))
