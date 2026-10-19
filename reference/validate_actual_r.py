"""Execute official sparkx.R in R and compare every result layer with Python.

Requires R with Matrix, Rcpp, RcppArmadillo, CompQuadForm. An isolated
CompQuadForm install in .cache/R-library is added to R_LIBS_USER if present.
Usage: python reference/validate_actual_r.py --rscript /path/to/Rscript
"""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from spatial_sparkx import normalize_vst, sparkx  # noqa: E402


def extract_cpp_helpers():
    source = (ROOT / "reference/upstream/src/spark_float.cpp").read_text()
    header = "#include <RcppArmadillo.h>\n// [[Rcpp::depends(RcppArmadillo)]]\n"
    functions = []
    for name in ("sp_sums_Rcpp", "sp_means_Rcpp", "sp_vars_Rcpp"):
        start = source.index("arma::vec " + name + "(")
        opening = source.index("{", start)
        level = 1
        stop = opening + 1
        while level:
            level += (source[stop] == "{") - (source[stop] == "}")
            stop += 1
        functions.append("// [[Rcpp::export]]\n" + source[start:stop] + "\n")
    (ROOT / ".cache/r_helpers.cpp").write_text(header + "\n".join(functions))


def make_cases():
    rng = np.random.default_rng(20261021)
    cases = []
    for seed in range(4):
        n, g = 83 + seed * 17, 19 + seed * 3
        coords = rng.uniform(-6, 13, (n, 2 + seed % 2))
        counts = rng.poisson(2.4, (n, g)).astype(float)
        cases.append((f"poisson_{seed}", counts, coords, None, "mixture", seed % 2 == 1))
    n, g = 137, 31
    coords = rng.uniform(-5, 10, (n, 2))
    counts = rng.poisson(2.1, (n, g)).astype(float)
    nuisance = rng.normal(size=n)
    cases.extend([
        ("covariates_intercept", counts, coords, np.column_stack([np.ones(n), nuisance]), "mixture", True),
        ("covariates_no_intercept", counts, coords, nuisance[:, None], "mixture", False),
        ("covariates_categorical", counts, coords, np.column_stack([np.ones(n), np.arange(n) % 3 == 0, np.arange(n) % 3 == 1]), "mixture", True),
        ("single_projection", counts, coords, None, "single", False),
    ])
    sparse_counts = counts.copy()
    sparse_counts[rng.random(counts.shape) < 0.85] = 0
    cases.append(("sparse_counts", sparse_counts, coords, None, "mixture", True))
    zero = counts.copy()
    zero[[0, 17], :] = 0
    zero[:, [2, 9]] = 0
    cases.append(("zero_filter", zero, coords, None, "mixture", True))
    strong = counts.copy()
    strong[:, 0] = rng.poisson(0.1 + 25 * (coords[:, 0] > 3))
    strong[:, 1] = rng.poisson(0.2 + 30 * np.exp(-coords[:, 0]**2 / 3))
    cases.append(("strong_patterns", strong, coords, None, "mixture", True))
    constant = counts.copy()
    constant[:, 0] = 3
    cases.append(("edge_constant_gene", constant, coords, None, "mixture", True))
    signed = counts.copy() - 1.7
    signed[0] = 0
    signed[0, :2] = [1, -1]
    cases.append(("edge_signed_cancellation", signed, coords, None, "mixture", False))
    cases.append(("edge_rank_deficient", counts, np.column_stack([coords[:, 0], coords[:, 0]]), None, "single", False))
    cases.append(("edge_single_gene", counts[:, :1], coords, None, "mixture", False))
    vst = rng.negative_binomial(2, 0.3, size=(180, 19)).astype(float)
    return cases, vst


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rscript", default=shutil.which("Rscript"))
    args = parser.parse_args()
    if args.rscript is None:
        parser.error("Rscript must be on PATH or supplied with --rscript")
    base = ROOT / ".cache/actual-r"
    inputs, output = base / "inputs", base / "outputs"
    inputs.mkdir(parents=True, exist_ok=True)
    cases, vst = make_cases()
    manifest = []
    for name, counts, coords, cov, option, is_sparse in cases:
        directory = inputs / name
        directory.mkdir(exist_ok=True)
        genes = [f"g{i}" for i in range(counts.shape[1])]
        pd.DataFrame(counts.T, index=genes).to_csv(directory / "counts.csv", float_format="%.17g")
        pd.DataFrame(coords).to_csv(directory / "coordinates.csv", float_format="%.17g")
        if cov is not None:
            pd.DataFrame(cov).to_csv(directory / "covariates.csv", float_format="%.17g")
        manifest.append({"case": name, "option": option, "sparse": is_sparse})
    pd.DataFrame(manifest).to_csv(inputs / "manifest.tsv", sep="\t", index=False)
    pd.DataFrame(vst.T).to_csv(inputs / "vst_counts.csv", float_format="%.17g")
    extract_cpp_helpers()
    env = os.environ.copy()
    env["PATH"] = str(Path(args.rscript).resolve().parent) + os.pathsep + env.get("PATH", "")
    env["TMPDIR"] = str(ROOT / ".cache/r-tmp")
    env["R_LIBS_USER"] = str(ROOT / ".cache/R-library")
    Path(env["TMPDIR"]).mkdir(exist_ok=True)
    with (base / "R-output.log").open("w") as stream:
        subprocess.run([args.rscript, "--vanilla", str(ROOT / "reference/run_actual_r.R"),
                        str(ROOT), str(inputs), str(output)], env=env, cwd=ROOT,
                       stdout=stream, stderr=subprocess.STDOUT, check=True)
    statuses = pd.read_csv(output / "statuses.csv").fillna("")
    records = []
    fixtures = {}
    fixture_cases = []
    for name, counts, coords, cov, option, is_sparse in cases:
        try:
            python = sparkx(sparse.csr_matrix(counts) if is_sparse else counts, coords, cov,
                            option=option, gene_names=[f"g{i}" for i in range(counts.shape[1])], n_threads=1)
            python_error = ""
        except Exception as error:
            python = None
            python_error = str(error)
        matched_both = True
        for mode in ("default", "analytic_equivalent"):
            status = statuses[(statuses["case"] == name) & (statuses["mode"] == mode)].iloc[0]
            record = {"case": name, "mode": mode, "r_success": bool(status.success),
                      "python_success": python is not None, "r_error": str(status.error),
                      "python_error": python_error}
            if status.success and python is not None:
                directory = output / mode / name
                r_stats = pd.read_csv(directory / "stats.csv", index_col=0)
                r_pvalues = pd.read_csv(directory / "pvalues.csv", index_col=0)
                r_combined = pd.read_csv(directory / "combined.csv", index_col=0)
                aligned = list(r_stats.index) == list(python.stats.index) and r_stats.shape == python.stats.shape
                record["same_genes_and_shape"] = aligned
                if aligned:
                    record["statistics_compared"] = int(r_stats.size)
                    record["max_absolute_statistic_error"] = float(np.max(np.abs(r_stats.to_numpy() - python.stats.to_numpy())))
                    record["max_absolute_single_pvalue_error"] = float(np.max(np.abs(r_pvalues.to_numpy() - python.res_stest.to_numpy())))
                    record["max_absolute_combined_pvalue_error"] = float(np.max(np.abs(r_combined.combinedPval.to_numpy() - python.res_mtest.combinedPval.to_numpy())))
                    record["max_absolute_adjusted_pvalue_error"] = float(np.max(np.abs(r_combined.adjustedPval.to_numpy() - python.res_mtest.adjustedPval.to_numpy())))
                    record["different_fdr_005_decisions"] = int(np.count_nonzero((r_combined.adjustedPval.to_numpy() < 0.05) != (python.res_mtest.adjustedPval.to_numpy() < 0.05)))
                    assert record["max_absolute_statistic_error"] < 1e-9, record
                    if mode == "analytic_equivalent":
                        for key in ("max_absolute_single_pvalue_error", "max_absolute_combined_pvalue_error", "max_absolute_adjusted_pvalue_error"):
                            assert record[key] < 1e-9, record
                    else:
                        assert record["max_absolute_single_pvalue_error"] < 1e-4, record
                    fixtures[f"{name}_{mode}_stats"] = r_stats.to_numpy()
                    fixtures[f"{name}_{mode}_pvalues"] = r_pvalues.to_numpy()
                    fixtures[f"{name}_{mode}_combined"] = r_combined.to_numpy()
            else:
                matched_both = False
            records.append(record)
        if matched_both:
            fixture_cases.append(name)
            fixtures[f"{name}_counts"] = counts
            fixtures[f"{name}_coordinates"] = coords
            fixtures[f"{name}_covariates"] = np.empty((len(counts), 0)) if cov is None else cov
            fixtures[f"{name}_option"] = np.array(option)
    expected_vst = pd.read_csv(output / "vst.csv", index_col=0).to_numpy().T
    actual_vst = normalize_vst(vst)
    report = {
        "reference": "Actual R 4.4.3 execution of unmodified sparkx.R and original extracted Rcpp sparse reductions",
        "upstream_commit": "a8b4bf27b804604dfda53da42992f100b8e4e727",
        "R_default_davies_accuracy": 1e-4,
        "analytic_control": "Same official R source; Davies probability evaluator is replaced with R pchisq after asserting all mixture weights are equal to relative 1e-10",
        "vst_max_absolute_error": float(np.max(np.abs(expected_vst - actual_vst))),
        "cases": records,
    }
    summary = {}
    for mode in ("default", "analytic_equivalent"):
        comparable = [r for r in records if r["mode"] == mode and r.get("same_genes_and_shape")]
        summary[mode] = {
            "comparable_datasets": len(comparable),
            "kernel_statistics_compared": sum(r["statistics_compared"] for r in comparable),
            "max_absolute_statistic_error": max(r["max_absolute_statistic_error"] for r in comparable),
            "max_absolute_single_pvalue_error": max(r["max_absolute_single_pvalue_error"] for r in comparable),
            "max_absolute_combined_pvalue_error": max(r["max_absolute_combined_pvalue_error"] for r in comparable),
            "max_absolute_adjusted_pvalue_error": max(r["max_absolute_adjusted_pvalue_error"] for r in comparable),
            "different_fdr_005_decisions": sum(r["different_fdr_005_decisions"] for r in comparable),
        }
    report["summary"] = summary
    fixtures["case_names"] = np.array(fixture_cases)
    np.savez_compressed(ROOT / "tests/fixtures/actual_r_reference.npz", **fixtures)
    target = ROOT / "reference/actual_r_validation.json"
    target.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
