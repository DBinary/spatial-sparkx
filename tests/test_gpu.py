import os

import numpy as np
import pytest
from scipy import sparse

from spatial_sparkx import sparkx

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(os.environ.get("SPARKX_TEST_GPU") != "1", reason="set SPARKX_TEST_GPU=1 to run CUDA checks"),
]


@pytest.mark.parametrize("sparse_input", [False, True])
@pytest.mark.parametrize("with_covariates", [False, True])
def test_cupy_matches_numpy(sparse_input, with_covariates):
    cp = pytest.importorskip("cupy")
    assert cp.cuda.runtime.getDeviceCount() > 0
    rng = np.random.default_rng(519)
    x = rng.poisson(1.7, size=(181, 17)).astype(float)
    coords = rng.uniform(-5, 10, size=(181, 2))
    covariates = np.column_stack([np.ones(181), rng.normal(size=181)]) if with_covariates else None
    matrix = sparse.csr_matrix(x) if sparse_input else x
    expected = sparkx(matrix, coords, covariates, batch_size=5, n_threads=1)
    actual = sparkx(matrix, coords, covariates, backend="cupy", batch_size=5, n_threads=1)
    np.testing.assert_allclose(actual.stats, expected.stats, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(actual.res_stest, expected.res_stest, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(actual.res_mtest, expected.res_mtest, rtol=1e-10, atol=1e-12)
