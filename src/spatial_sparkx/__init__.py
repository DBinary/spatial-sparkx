"""SPARK-X spatial gene-expression tests implemented without an R runtime."""

from .core import SparkXResult, sparkx, sparkx_r
from .kernels import spatial_kernels, transform_coordinates
from .statistics import acat, by_adjust
from .anndata import sparkx_adata
from .normalization import normalize_vst

__all__ = [
    "SparkXResult", "sparkx", "sparkx_r", "sparkx_adata", "acat", "by_adjust",
    "spatial_kernels", "transform_coordinates", "normalize_vst",
]
__version__ = "0.1.0"
