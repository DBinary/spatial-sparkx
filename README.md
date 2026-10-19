# spatial-sparkx

SPARK-X 的原生 Python 移植，可直接在 `AI_training` 中使用。核心计算只依赖
NumPy、SciPy、Pandas 和 threadpoolctl；运行不需要 R、rpy2 或原版 SPARK。
这是独立移植，并非 xzhoulab 发布的官方 Python 包。

## 安装

Python 3.10 及以上，可直接从本仓库安装：

```bash
python -m pip install "spatial-sparkx @ git+https://github.com/DBinary/spatial-sparkx.git"

# 同时安装 AnnData 支持，用于 .h5ad 文件和 sparkx_adata
python -m pip install "spatial-sparkx[anndata] @ git+https://github.com/DBinary/spatial-sparkx.git"
```

私有仓库需要先配置有访问权限的 GitHub 凭据。本包尚未发布到 PyPI。
GPU 后端需要另行安装与本机 CUDA 匹配的 CuPy；默认安装使用 CPU。

从源码进行开发、测试和打包：

```bash
git clone https://github.com/DBinary/spatial-sparkx.git
cd spatial-sparkx
python -m pip install -e ".[test]" build
python -m pytest -q -m "not gpu"
python -m build
```

`dist/` 下生成 wheel 和源码压缩包。GitHub Actions 会运行 Python 3.10 / 3.12
的 CPU 测试、检查包元数据，并保存构建产物；GPU 测试需在有 CUDA 的机器运行。

## 快速使用

```python
from spatial_sparkx import sparkx

# counts: 细胞/空间点 × 基因；支持 NumPy、SciPy 稀疏矩阵、Pandas DataFrame
# coordinates: 细胞/空间点 × 空间维度，行顺序必须和 counts 一致
result = sparkx(counts, coordinates, option="mixture", n_threads=4)

result.res_mtest              # combinedPval、adjustedPval（BY 校正）
result.res_stest              # 每个基因在 11 个核下的 p 值
result.stats                 # 每个基因在 11 个核下的统计量
result.res_mtest.to_csv("sparkx_results.csv")
```

默认采用 Python/AnnData 的“空间点 × 基因”方向。原 R 实现实际采用
“基因 × 空间点”方向，使用 `gene_axis=0`，或调用兼容入口：

```python
from spatial_sparkx import sparkx_r

result = sparkx_r(count_in=gene_by_cell_counts, locus_in=coordinates, numCores=4)
```

`numCores` 在 Python 兼容入口中限制 BLAS 线程数；没有逐基因启动 R 式进程。
`result["stats"]`、`result["res_stest"]` 和 `result["res_mtest"]` 也可使用。

原版辅助函数 `NormalizeVSTfast` 对应 `normalize_vst(counts)`，支持
`gene_axis=0` 和 `return_phi=True`。此函数会返回稠密残差矩阵；
SPARK-X 主检验不会自动调用它。其单参数方差模型用解析最小二乘求解，
随后回归掉截距和 log 总计数，与原版 `nls` / `lm` 的目标相同。

## 直接处理 AnnData

```python
from spatial_sparkx import sparkx_adata

result = sparkx_adata(
    adata,
    spatial_key="spatial",       # adata.obsm["spatial"]
    layer="counts",              # 原始计数；没有该层时可省略，使用 adata.X
    n_threads=4,
)

adata.var["sparkx_pvalue"]
adata.var["sparkx_padj"]
adata.var["sparkx_tested"]
adata.varm["sparkx_stats"]
adata.varm["sparkx_pvalues"]
adata.uns["sparkx"]

svg = adata.var_names[adata.var["sparkx_padj"] < 0.05]
```

被排除的全零基因在 `adata.var` 中保留位置，`sparkx_tested=False`，p 值为 NaN。
默认向 AnnData 写结果；`inplace=False` 仅返回结果。不支持 backed 模式；
向 view 写入前需先调用 `.copy()`。

## 协变量校正

```python
import numpy as np

# 保持原 R 语义：传入协变量时不会自动添加截距。
design = np.column_stack([
    np.ones(adata.n_obs),
    adata.obs["numeric_batch_covariate"].to_numpy(),
])
result = sparkx_adata(adata, layer="counts", covariates=design, n_threads=4)
```

分类变量需先进行数值编码，例如 `pd.get_dummies(..., drop_first=True)`。
不提供协变量时自动消除截距；提供时使用所给设计矩阵的列空间。

## 可选 GPU

```python
result = sparkx_adata(
    adata, layer="counts", backend="cupy", batch_size=512, n_threads=4,
)
```

CuPy 只在 `backend="cupy"` 时导入。稀疏表达矩阵逐基因批次传到 GPU；
不会整体转为稠密矩阵。空间特征分解和最终 p 值计算仍在 CPU 上进行，
返回结果始终为 Pandas 表。可用 `CUDA_VISIBLE_DEVICES` 或 CuPy device context
选择 GPU。小数据上 CPU 可能更快；`batch_size` 控制基因批次和显存需求。

## 命令行

```bash
conda activate AI_training
sparkx input.h5ad --layer counts --output sparkx_results.csv \
  --write-h5ad input_with_sparkx.h5ad --n-threads 4

# GPU
CUDA_VISIBLE_DEVICES=1 sparkx input.h5ad --layer counts \
  --backend cupy --batch-size 512 --output sparkx_results.tsv
```

输出路径必须与输入文件不同。命令行目前不接收协变量；有协变量时使用 Python API。

## 保留的算法

- 1 个原始坐标投影核、5 个 Gaussian 变换投影核、5 个 cosine 变换投影核；
  `option="single"` 仅使用第一个核。
- Gaussian/cosine 使用逐维中心化坐标，其绝对值的 20%、40%、60%、80%、100%
  分位数作为尺度，与 R `quantile(type=7)` 一致。
- 保留 SPARK-X 原始统计量、表达核的特征值因子、协变量残差化和样本量因子。
- ACAT 使用原版端点约定：出现精确 0 返回 0；出现精确 1 返回 1；两者同时出现时报错。
  采用等价的 cotangent 恒等式和缩放计算，避免端点附近的消减误差与 `1/p` 溢出。
- 多重检验使用 Benjamini–Yekutieli（BY），不是 BH。

### 为什么不需要 Davies / R？

原版每个空间核的残差化坐标记为 `Z`。原源码计算的空间特征值为
`eigen(Z.T @ Z @ inverse(Z.T @ Z))`，满秩时全部为 1。
因此每个基因的原始零假设分布 `sum(lambda_G * chi2_1)` **恰好等于**
`lambda_G * chi2_d`。本包用 SciPy incomplete-gamma survival function
直接计算该分布，以 QR 正交基计算投影统计量，不构造观测点之间的方阵。

这保留了统计模型，但不要求和原版 Davies 数值积分逐位一致：原版默认绝对误差
容限为 `1e-4`，本包直接计算该特殊情形的分布尾概率。极端尾概率也可能存在浮点下溢。

### 明确处理的边界情况

- 默认 `rank_policy="raise"`：坐标核或协变量秩不足时报错，对应原版求逆失败。
  可显式设为 `"drop"`，使用可识别子空间的维度作为自由度；秩零核 p 值设为 1，
  并发出警告。这是为退化输入提供的扩展。
- 按 R 原码去除总和为零的空间点和基因，返回原始位置对应的
  `cell_mask` / `gene_mask`。非负计数输入下即为全零筛选；带正负号的
  表达值如果总和恰好相互抵消，也会和 R 版一样被排除。
- 常量基因或完全由协变量解释的基因：默认报错，避免自动赋予与原版无定义或
  不稳定结果不同的 p 值。需要保守处理时，可显式指定
  `degenerate_policy="conservative"`，将统计量设为 0、p 值设为 1，并发出警告。
  此选项是额外扩展，不属于“与 R 版只差数值”的保证范围。
- 非常重复的非恒定坐标可能产生零分位数尺度；此时明确报错，不产生 NaN 后继续检验。
- DataFrame 的空间点索引必须与坐标/协变量 DataFrame 顺序一致，不自动重排。

## 安装与验证

```bash
python -m pip install --no-build-isolation -e .
python -m pytest -q

# 具备可用 CUDA/CuPy 时额外运行 GPU 一致性测试
SPARKX_TEST_GPU=1 CUDA_VISIBLE_DEVICES=1 python -m pytest -q -m gpu

# 可重新生成独立参考数据：仅此验证步骤需要 g++，运行本包不需要编译器
python reference/validate_upstream.py
python reference/benchmark.py

# 实际执行 R 版；R 侧需有 Matrix、Rcpp、RcppArmadillo、CompQuadForm
python reference/validate_actual_r.py --rscript /path/to/Rscript
```

验证包括两个独立层次：

1. 官方公式 + 未修改的 Davies C++ 实现，覆盖 198 个基因/核组合。
   结果见 `reference/validation.json`。
2. **实际运行 R 4.4.3 中的官方 `sparkx.R` 和原始稀疏 C++ 辅助函数**，
   使用 CompQuadForm 1.4.4，对照 12 组 R 能完成的数据、3430 个核统计量。
   R 默认 Davies 下单核 p 值最大绝对差为 `1.64e-5`；只把 R 的尾概率求解器
   换成数学等价的 `pchisq` 后，单核/合并/BY 校正 p 值最大绝对差均小于 `5.5e-14`。
   这区分了数学算法差异与求解器的数值差异；未把默认 Davies 的误差说成机器舍入误差。

详细结果、比较范围及退化输入说明见 [VALIDATION.md](VALIDATION.md) 和
`reference/actual_r_validation.json`。其他测试覆盖稠密/稀疏一致性、输入方向、
分批计算、GPU/CPU、协变量、秩不足、常量基因、ACAT、BY 和 AnnData 文件读写。

输入为 `n × g` 时，空间投影部分使用约 `n × (协变量秩 + 各核秩之和)` 的特征，
再按基因分批计算；不会生成 `n × n` 距离、核或投影矩阵。

## 来源和引用

本移植依据 [xzhoulab/SPARK 的 R/sparkx.R](https://github.com/xzhoulab/SPARK/blob/a8b4bf27b804604dfda53da42992f100b8e4e727/R/sparkx.R)，
固定提交 `a8b4bf27b804604dfda53da42992f100b8e4e727`。

Jiaqiang Zhu, Shiquan Sun, and Xiang Zhou. SPARK-X: non-parametric modeling enables
scalable and robust detection of spatial expression patterns for large spatial
transcriptomic studies. *Genome Biology* (2021).
[官方项目](https://xzhoulab.github.io/SPARK/)；[论文](https://doi.org/10.1186/s13059-021-02404-0)。

原项目及本移植采用 GPL-3.0；Davies 验证源码采用 GPL >= 2。
许可证及归属见 `LICENSE`、`NOTICE` 与 `reference/`。
