from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import sparse, stats

from spatial_sparkx import acat, by_adjust, normalize_vst, sparkx, sparkx_adata, sparkx_r, spatial_kernels


@pytest.fixture
def data():
    rng = np.random.default_rng(381)
    return rng.poisson(2.3, size=(91, 13)).astype(float), rng.uniform(-4, 11, size=(91, 2))


@pytest.mark.parametrize("design", ["uncorrected", "covariates"])
def test_matches_independent_davies_reference(design):
    path = Path(__file__).parent / "fixtures/davies_reference.npz"
    with np.load(path) as ref:
        cov = ref["covariates"] if design == "covariates" else None
        result = sparkx(ref["counts"], ref["coordinates"], cov, n_threads=1)
        np.testing.assert_allclose(result.stats, ref[f"{design}_stats"], rtol=1e-11, atol=1e-11)
        np.testing.assert_allclose(result.res_stest, ref[f"{design}_pvalues"], rtol=0, atol=2e-8)


def test_actual_r_full_pipeline_fixtures():
    path = Path(__file__).parent / "fixtures/actual_r_reference.npz"
    with np.load(path) as ref:
        for name in ref["case_names"]:
            cov = ref[f"{name}_covariates"]
            if cov.shape[1] == 0:
                cov = None
            result = sparkx(ref[f"{name}_counts"], ref[f"{name}_coordinates"], cov,
                            option=str(ref[f"{name}_option"]), n_threads=1)
            np.testing.assert_allclose(result.stats, ref[f"{name}_default_stats"], rtol=1e-11, atol=1e-10)
            np.testing.assert_allclose(result.res_stest, ref[f"{name}_default_pvalues"], rtol=0, atol=1e-4)
            np.testing.assert_allclose(result.res_stest, ref[f"{name}_analytic_equivalent_pvalues"], rtol=0, atol=1e-9)
            np.testing.assert_allclose(result.res_mtest, ref[f"{name}_analytic_equivalent_combined"], rtol=0, atol=1e-9)


@pytest.mark.parametrize("matrix_type", [np.asarray, sparse.csr_matrix, sparse.csc_matrix,
                                         sparse.csr_array, sparse.coo_matrix])
@pytest.mark.parametrize("batch_size", [1, 4, 50])
def test_dense_sparse_batch_and_r_orientation(data, matrix_type, batch_size):
    x, coords = data
    expected = sparkx(x, coords, n_threads=1)
    actual = sparkx(matrix_type(x), coords, batch_size=batch_size, n_threads=1)
    reversed_axes = sparkx_r(matrix_type(x.T), coords, numCores=1, batch_size=batch_size)
    for result in (actual, reversed_axes):
        np.testing.assert_allclose(result.stats, expected.stats, rtol=1e-11, atol=1e-12)
        np.testing.assert_allclose(result.res_stest, expected.res_stest, rtol=1e-11, atol=1e-12)
        np.testing.assert_allclose(result.res_mtest, expected.res_mtest, rtol=1e-10, atol=1e-12)
    assert list(expected) == ["stats", "res_stest", "res_mtest"]


def test_filters_zero_genes_and_observations_and_labels(data):
    x, coords = data
    x[2] = 0
    x[:, 4] = 0
    names = [f"g{i}" for i in range(x.shape[1])]
    result = sparkx(x, coords, gene_names=names, n_threads=1)
    assert not result.cell_mask[2]
    assert not result.gene_mask[4]
    assert list(result.res_mtest.index) == [n for n in names if n != "g4"]
    expected = sparkx(np.delete(np.delete(x, 2, axis=0), 4, axis=1), np.delete(coords, 2, axis=0), n_threads=1)
    np.testing.assert_allclose(result.res_mtest, expected.res_mtest)


def test_single_gene_and_one_dimensional_coordinates(data):
    x, coords = data
    result = sparkx(x[:, 0], coords[:, 0], option="single", n_threads=1)
    retained = x[:, 0] != 0
    g = x[retained, 0]
    z = coords[retained, 0] - coords[retained, 0].mean()
    ratio = 1 - len(g) * g.mean()**2 / (g @ g)
    score = len(g) * (g @ z)**2 / (z @ z) / (g @ g)
    np.testing.assert_allclose(result.stats.iloc[0, 0], score)
    np.testing.assert_allclose(result.res_mtest.iloc[0, 0], stats.chi2.sf(score / ratio, 1))


def test_covariate_calculation_and_column_scaling(data):
    x, coords = data
    nuisance = np.column_stack([np.ones(len(x)), np.linspace(-1, 1, len(x))])
    expected = sparkx(x, coords, nuisance, n_threads=1)
    actual = sparkx(x, coords, nuisance * np.array([1e-6, 1e4]), n_threads=1)
    np.testing.assert_allclose(actual.stats, expected.stats, rtol=1e-10, atol=1e-11)
    np.testing.assert_allclose(actual.res_mtest, expected.res_mtest, rtol=1e-10, atol=1e-12)
    assert not np.allclose(expected.stats, sparkx(x, coords, n_threads=1).stats)


def test_covariates_without_intercept_are_not_silently_centered(data):
    x, coords = data
    cov = np.linspace(1, 3, len(x))[:, None]
    result = sparkx(x, coords, cov, option="single", n_threads=1)
    z = coords - cov @ np.linalg.inv(cov.T @ cov) @ cov.T @ coords
    g = x[:, 0]
    score = len(x) * (g @ z) @ np.linalg.inv(z.T @ z) @ (z.T @ g) / (g @ g)
    np.testing.assert_allclose(result.stats.iloc[0, 0], score)


def test_rank_deficiency_is_explicit(data):
    x, coords = data
    duplicate = np.column_stack([coords[:, 0], coords[:, 0]])
    with pytest.raises(ValueError, match="rank"):
        sparkx(x, duplicate, option="single", n_threads=1)
    with pytest.warns(RuntimeWarning, match="rank"):
        result = sparkx(x, duplicate, option="single", rank_policy="drop", n_threads=1)
    expected = sparkx(x, coords[:, 0], option="single", n_threads=1)
    np.testing.assert_allclose(result.res_stest, expected.res_stest)


def test_constant_coordinates_and_genes_have_conservative_extension(data):
    x, coords = data
    x[:, 1] = 3
    with pytest.raises(ValueError, match="residual expression variance"):
        sparkx(x, coords, n_threads=1)
    with pytest.warns(RuntimeWarning, match="expression residual"):
        result = sparkx(x, coords, degenerate_policy="conservative", n_threads=1)
    assert result.degenerate_genes[1]
    assert (result.stats.iloc[1] == 0).all()
    assert (result.res_stest.iloc[1] == 1).all()
    with pytest.warns(RuntimeWarning, match="rank"):
        rank_zero = sparkx(x[:, :1], np.ones((len(x), 2)), rank_policy="drop", n_threads=1)
    assert (rank_zero.res_mtest.to_numpy() == 1).all()


def test_sparse_path_does_not_densify_expression(monkeypatch):
    rng = np.random.default_rng(2)
    x = sparse.random(5000, 80, density=0.015, random_state=rng, format="csc")
    coords = rng.uniform(0, 100, size=(5000, 2))
    def forbidden(*args, **kwargs):
        raise AssertionError("expression must not be densified")
    monkeypatch.setattr(sparse.csc_matrix, "toarray", forbidden)
    monkeypatch.setattr(sparse.csc_matrix, "todense", forbidden)
    result = sparkx(x, coords, batch_size=11, n_threads=1)
    assert result.stats.shape == (80, 11)


def test_spatial_signal_is_detected():
    rng = np.random.default_rng(19)
    coords = rng.uniform(-5, 5, (600, 2))
    x = rng.poisson(2, (600, 101)).astype(float)
    x[:, 0] = rng.poisson(0.5 + 10 * np.exp(-coords[:, 0]**2 / 2))
    result = sparkx(x, coords, n_threads=1)
    assert result.res_mtest.iloc[0].adjustedPval < 0.001
    assert result.res_mtest.combinedPval.argmin() == 0


@pytest.mark.parametrize("p", [[0.01, 0.02, 0.05], [0.1, 0.5, 0.9], [0.002, 0.7]])
def test_acat_matches_upstream_formula(p):
    p = np.array(p)
    statistic = np.mean(np.tan((0.5 - p) * np.pi))
    expected = 0.5 - np.arctan(statistic) / np.pi
    np.testing.assert_allclose(acat(p), expected, atol=1e-15)


def test_acat_extreme_tail_endpoints_weights_and_axis():
    np.testing.assert_allclose(acat([1e-300, 0.5]), 2e-300, rtol=1e-14, atol=0)
    assert acat([np.nextafter(0.0, 1.0)]) == np.nextafter(0.0, 1.0)
    for p in (1e-300, 1e-30, 1e-16, 1e-15, 1e-10, 0.5, 0.999):
        np.testing.assert_allclose(acat([p]), p, rtol=1e-14, atol=0)
    assert acat([0, 0.2]) == 0
    assert acat([1, 0.2]) == 1
    assert acat([1e-200, 0.1], weights=[0, 1]) == pytest.approx(0.1)
    p = np.array([[0.1, 0.3], [0.5, 0.7]])
    np.testing.assert_allclose(acat(p, axis=0), [acat(p[:, j]) for j in range(2)])
    with pytest.raises(ValueError, match="both exact"):
        acat([0, 1])
    for invalid in ([np.nan], [-0.01], [1.01], []):
        with pytest.raises(ValueError):
            acat(invalid)
    with pytest.raises(ValueError):
        acat([0.1, 0.2], weights=[0, 0])


def test_by_adjustment_matches_independent_scipy_implementation():
    p = np.array([0.003, 0.07, 1, 0.0001, 0.003, 0.9])
    np.testing.assert_allclose(by_adjust(p), stats.false_discovery_control(p, method="by"))


def test_anndata_mapping_and_h5ad_roundtrip(data, tmp_path):
    ad = pytest.importorskip("anndata")
    x, coords = data
    x[3] = 0
    x[:, 5] = 0
    a = ad.AnnData(sparse.csr_matrix(x))
    a.var_names = [f"gene{i}" for i in range(a.n_vars)]
    a.obsm["spatial"] = coords
    a.layers["counts"] = a.X.copy()
    result = sparkx_adata(a, layer="counts", n_threads=1)
    assert not a.var.iloc[5].sparkx_tested
    assert np.isnan(a.var.iloc[5].sparkx_pvalue)
    assert a.varm["sparkx_stats"].shape == (13, 11)
    assert a.uns["sparkx"]["n_observations_tested"] == 90
    np.testing.assert_allclose(a.var.loc[a.var.sparkx_tested, "sparkx_pvalue"], result.res_mtest.combinedPval)
    path = tmp_path / "results.h5ad"
    a.write_h5ad(path)
    b = ad.read_h5ad(path)
    np.testing.assert_allclose(b.var.sparkx_pvalue, a.var.sparkx_pvalue, equal_nan=True)
    assert b.uns["sparkx"]["multiple_testing"] == "Benjamini-Yekutieli"
    untouched = a.copy()
    sparkx_adata(untouched, key_added="other", inplace=False, n_threads=1)
    assert "other_pvalue" not in untouched.var


def test_dataframe_order_is_checked(data):
    x, coords = data
    counts = pd.DataFrame(x, index=[f"c{i}" for i in range(len(x))])
    locations = pd.DataFrame(coords, index=counts.index)
    assert sparkx(counts, locations, option="single", n_threads=1).stats.shape == (13, 1)
    with pytest.raises(ValueError, match="index must match"):
        sparkx(counts, locations.iloc[::-1], option="single")


@pytest.mark.parametrize("kwargs", [
    {"option": "invalid"}, {"backend": "invalid"}, {"rank_policy": "invalid"},
    {"batch_size": 0}, {"batch_size": True}, {"n_threads": 0}, {"gene_axis": 2},
    {"degenerate_policy": "invalid"},
])
def test_argument_validation(data, kwargs):
    x, coords = data
    with pytest.raises(ValueError):
        sparkx(x, coords, **kwargs)


def test_shape_finiteness_and_covariate_validation(data):
    x, coords = data
    with pytest.raises(ValueError, match="rows must match"):
        sparkx(x, coords[:-1])
    with pytest.raises(ValueError, match="finite"):
        sparkx(x * np.nan, coords)
    with pytest.raises(ValueError, match="nonzero observations"):
        sparkx(np.zeros_like(x), coords)
    with pytest.raises(ValueError, match="covariates must have shape"):
        sparkx(x, coords, np.ones((5, 1)))


def test_kernel_feature_count_and_translation_invariance(data):
    x, coords = data
    kernels = spatial_kernels(coords)
    shifted = spatial_kernels(coords + np.array([4, -7]))
    assert list(kernels) == ["projection", *[f"gaus{i}" for i in range(1, 6)], *[f"cos{i}" for i in range(1, 6)]]
    for name in list(kernels)[1:]:
        np.testing.assert_allclose(kernels[name], shifted[name], atol=1e-13)
    a = sparkx(x, coords, n_threads=1)
    b = sparkx(x, coords + np.array([4, -7]), n_threads=1)
    np.testing.assert_allclose(a.res_mtest, b.res_mtest, rtol=1e-10, atol=1e-12)


def test_vst_matches_original_least_squares_model():
    rng = np.random.default_rng(514)
    x = rng.negative_binomial(2, 0.3, size=(180, 19)).astype(float)
    actual, phi = normalize_vst(sparse.csr_matrix(x), return_phi=True)
    mean = x.mean(axis=0)
    from scipy.optimize import least_squares
    fit = least_squares(lambda p: x.var(axis=0, ddof=1) - mean - p[0] * mean**2, [1.0])
    np.testing.assert_allclose(phi, fit.x[0], rtol=1e-7)
    design = np.column_stack([np.ones(len(x)), np.log(x.sum(axis=1))])
    transformed = np.log(x + 1 / (2 * phi))
    expected = transformed - design @ np.linalg.lstsq(design, transformed, rcond=None)[0]
    np.testing.assert_allclose(actual, expected, atol=1e-13)
    np.testing.assert_allclose(normalize_vst(x.T, gene_axis=0), actual.T, atol=1e-13)
    np.testing.assert_allclose(design.T @ actual, 0, atol=1e-10)


def test_cli_end_to_end(data, tmp_path):
    ad = pytest.importorskip("anndata")
    from spatial_sparkx.cli import main
    x, coords = data
    a = ad.AnnData(sparse.csr_matrix(x))
    a.obsm["spatial"] = coords
    a.layers["counts"] = a.X.copy()
    source = tmp_path / "input.h5ad"
    table = tmp_path / "result.tsv"
    annotated = tmp_path / "annotated.h5ad"
    a.write_h5ad(source)
    main([str(source), "--output", str(table), "--write-h5ad", str(annotated),
          "--layer", "counts", "--n-threads", "1"])
    result = pd.read_csv(table, sep="\t", index_col=0)
    assert result.shape == (13, 2)
    assert "sparkx_padj" in ad.read_h5ad(annotated).var
    with pytest.raises(SystemExit):
        main([str(source), "--output", str(source)])


def test_input_is_not_mutated_and_complex_values_rejected(data):
    x, coords = data
    original = x.copy()
    sparkx(x, coords, n_threads=1)
    np.testing.assert_array_equal(x, original)
    for invalid in (x.astype(complex) + 1j, sparse.csr_matrix(x.astype(complex) + 1j)):
        with pytest.raises(ValueError, match="real-valued"):
            sparkx(invalid, coords)


def test_signed_zero_sum_filter_matches_r(data):
    x, coords = data
    x = x - 1.7
    x[0] = 0
    x[0, :2] = [1, -1]
    result = sparkx(x, coords, n_threads=1)
    assert not result.cell_mask[0]
    expected = sparkx(x[1:], coords[1:], n_threads=1)
    np.testing.assert_allclose(result.res_mtest, expected.res_mtest)
