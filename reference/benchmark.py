"""Record an actual sparse CPU scaling run, without an observation-square matrix."""

import json
from pathlib import Path
import resource
import sys
import time

import numpy as np
import scipy
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from spatial_sparkx import sparkx  # noqa: E402

rng = np.random.default_rng(512)
n, g = 100_000, 1_000
other = sparse.random(n, g - 1, density=0.005, format="csc", random_state=rng,
                      data_rvs=lambda size: rng.integers(1, 7, size=size).astype(float))
anchor = sparse.csc_matrix((1 + rng.poisson(1, size=n))[:, None])
counts = sparse.hstack([anchor, other], format="csc")
coordinates = rng.uniform(0, 1000, size=(n, 2))
start = time.perf_counter()
result = sparkx(counts, coordinates, batch_size=128, n_threads=1)
elapsed = time.perf_counter() - start
report = {
    "n_observations": n, "n_genes": g, "nnz": counts.nnz,
    "n_kernels": result.stats.shape[1], "seconds": elapsed,
    "peak_process_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
    "backend": "numpy", "n_threads": 1,
    "numpy": np.__version__, "scipy": scipy.__version__,
}
(ROOT / "reference/benchmark.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
