"""Independently validate the Python port against upstream equations + Davies.

No R runtime is used here. The upstream R equations are evaluated literally
using inverses and a per-gene loop, and CompQuadForm's unmodified qfc.cpp is
compiled with minimal standard-library compatibility headers. This validates
both the QR/batched implementation and the exact chi-square simplification.
Run from the repository root: python reference/validate_upstream.py
"""

import ctypes
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from spatial_sparkx import sparkx  # noqa: E402


def davies_library():
    base = ROOT / "reference/compquadform"
    target = ROOT / ".cache/libqfc.so"
    target.parent.mkdir(exist_ok=True)
    subprocess.run([
        "g++", "-std=c++11", "-O2", "-fPIC", "-shared", "-I", str(base / "stubs"),
        str(base / "qfc.cpp"), "-o", str(target),
    ], check=True)
    lib = ctypes.CDLL(str(target))
    double_ptr = ctypes.POINTER(ctypes.c_double)
    int_ptr = ctypes.POINTER(ctypes.c_int)
    lib.qfc.argtypes = [double_ptr, double_ptr, int_ptr, int_ptr, double_ptr,
                       double_ptr, int_ptr, double_ptr, double_ptr, int_ptr, double_ptr]
    lib.qfc.restype = None
    return lib


def davies(lib, statistic, weights, accuracy=1e-8):
    weights = np.ascontiguousarray(np.sort(weights)[::-1], dtype=np.float64)
    delta = np.zeros_like(weights)
    degrees = np.ones(len(weights), dtype=np.int32)
    rank = ctypes.c_int(len(weights))
    sigma = ctypes.c_double(0)
    score = ctypes.c_double(float(statistic))
    limit = ctypes.c_int(1_000_000)
    tolerance = ctypes.c_double(accuracy)
    trace = np.zeros(7, dtype=np.float64)
    fault = ctypes.c_int()
    cdf = ctypes.c_double()
    dp = ctypes.POINTER(ctypes.c_double)
    ip = ctypes.POINTER(ctypes.c_int)
    lib.qfc(weights.ctypes.data_as(dp), delta.ctypes.data_as(dp), degrees.ctypes.data_as(ip),
            ctypes.byref(rank), ctypes.byref(sigma), ctypes.byref(score), ctypes.byref(limit),
            ctypes.byref(tolerance), trace.ctypes.data_as(dp), ctypes.byref(fault), ctypes.byref(cdf))
    if fault.value != 0:
        raise RuntimeError(f"Davies reference failed: ifault={fault.value}")
    return 1 - cdf.value


def literal_reference(counts, coordinates, covariates, lib):
    """Follow R/sparkx.R, without using any production implementation helpers."""
    centered = coordinates - coordinates.mean(axis=0)
    bandwidths = np.quantile(np.abs(centered), np.arange(1, 6) / 5, axis=0)
    kernels = [coordinates]
    kernels += [np.exp(-centered**2 / (2 * bandwidth**2)) for bandwidth in bandwidths]
    kernels += [np.cos(2 * np.pi * centered / bandwidth) for bandwidth in bandwidths]
    statistics = np.empty((counts.shape[1], len(kernels)))
    pvalues = np.empty_like(statistics)
    n = counts.shape[0]
    for j, features in enumerate(kernels):
        if covariates is None:
            z = features - features.mean(axis=0)
        else:
            nuisance_inverse = np.linalg.inv(covariates.T @ covariates)
            z = features - covariates @ nuisance_inverse @ covariates.T @ features
        inverse = np.linalg.inv(z.T @ z)
        eigenvalues = np.linalg.eigvals(z.T @ (z @ inverse)).real
        for i in range(counts.shape[1]):
            gene = counts[:, i]
            norm = gene @ gene
            projected = gene @ z
            score = n * projected @ inverse @ projected / norm
            if covariates is None:
                expression_lambda = 1 - n * np.mean(gene)**2 / norm
            else:
                cross = gene @ covariates
                expression_lambda = (norm - cross @ nuisance_inverse @ cross) / norm
            statistics[i, j] = score
            pvalues[i, j] = davies(lib, score, expression_lambda * eigenvalues)
    return statistics, pvalues


def main():
    lib = davies_library()
    rng = np.random.default_rng(1021)
    coordinates = rng.uniform(0.3, 13, size=(73, 2))
    counts = rng.poisson(2.2, size=(73, 9)).astype(float)
    covariates = np.column_stack([np.ones(73), rng.normal(size=73), rng.normal(size=73)])
    records = []
    fixtures = {"counts": counts, "coordinates": coordinates, "covariates": covariates}
    for name, design in (("uncorrected", None), ("covariates", covariates)):
        expected_stats, expected_p = literal_reference(counts, coordinates, design, lib)
        result = sparkx(counts, coordinates, design, n_threads=1)
        stat_error = float(np.max(np.abs(expected_stats - result.stats.to_numpy())))
        p_error = float(np.max(np.abs(expected_p - result.res_stest.to_numpy())))
        assert stat_error < 1e-10, stat_error
        assert p_error < 2e-8, p_error
        fixtures[f"{name}_stats"] = expected_stats
        fixtures[f"{name}_pvalues"] = expected_p
        records.append({"case": name, "statistics_compared": int(expected_stats.size),
                        "max_absolute_statistic_error": stat_error,
                        "max_absolute_pvalue_error": p_error})
    target = ROOT / "tests/fixtures"
    target.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target / "davies_reference.npz", **fixtures)
    source = ROOT / "reference/compquadform/qfc.cpp"
    report = {
        "upstream_commit": "a8b4bf27b804604dfda53da42992f100b8e4e727",
        "reference": "Literal upstream R equations + unmodified CompQuadForm qfc.cpp; not an R runtime execution",
        "davies_accuracy": 1e-8,
        "qfc_cpp_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "cases": records,
    }
    (ROOT / "reference/validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
