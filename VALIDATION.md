# Python 与 R 的一致性验证

结论：对本次验证中 R 能正常完成的满秩、非退化输入，Python 版保留了
SPARK-X 的数学检验方法；计算差异来自线性代数和概率求解的数值实现。
不能把这个结论扩展成“任意输入、任意数值条件下逐位相同”。

## 真实 R 对照

- 固定源码：`xzhoulab/SPARK`，提交 `a8b4bf27b804604dfda53da42992f100b8e4e727`。
- 实际执行未修改的 `R/sparkx.R`；其稀疏 sum/mean/variance 辅助函数从同一提交的
  C++ 源码原样提取，通过 Rcpp 编译。没有用 Python 模拟 R 端的这些结果。
- R 4.4.3、Matrix 1.7-6、CompQuadForm 1.4.4。
- 12 组可比较数据，3430 个基因/核统计量。覆盖 2D/3D 坐标、稀疏计数、
  协变量有/无截距、分类协变量、单核、零计数筛选、强空间模式及带正负号的零和筛选。

| 最大绝对差 | R 原版默认 Davies | R 等价卡方尾概率对照 |
|---|---:|---:|
| 检验统计量 | 6.25e-13 | 6.25e-13 |
| 单核 p 值 | 1.64e-5 | 4.23e-14 |
| ACAT 合并 p 值 | 2.46e-4 | 5.48e-14 |
| BY 校正 p 值 | 3.61e-5 | 2.09e-14 |
| FDR < 0.05 判定不一致的基因数 | 0 | 0 |

原版 Davies 的默认绝对误差容限为 `1e-4`。ACAT 对输入 p 值进行非线性变换，
因此不能把单核误差容限当成合并 p 值的误差上界。

第二列完全保留原版的默认 Davies 调用。第三列仍然运行同一份 R 主函数、
同一统计量、ACAT 和 BY 代码，只把其概率计算器换成 R 自身的 `pchisq`；
替换前逐次断言所有混合权重在相对 `1e-10` 内相等。
这是对“数值求解器是误差来源”的独立控制实验，不是伪称 R 原版默认配置。

另验证了 `NormalizeVSTfast`：实际 R `nls/lm` 与 Python 的解析最小二乘/回归残差
最大绝对差为 `2.85e-10`。

## 数学对应关系

原版针对每个核，先得到去掉协变量后的坐标 `Z`，计算：

```
stat = n * g.T @ Z @ inverse(Z.T @ Z) @ Z.T @ g / (g.T @ g)
lambda_G = 1 - ||P_X g||² / ||g||²
lambda_K = eigen(Z.T @ Z @ inverse(Z.T @ Z))
```

满秩时 `lambda_K` 全部为 1。因此，原版要计算的分布
`sum(lambda_G * chi2_1)` 与 `lambda_G * chi2_d` 完全相同。
Python 使用正交基和 incomplete-gamma survival function，避免生成 `n × n` 方阵。
没有把 SPARK-X 改成 Moran's I、置换检验或另一个空间统计方法。
空间变换、分位数、协变量语义、ACAT 端点规则和 BY 方法均保留。

## 不应混称为“仅数值差异”的情况

- 常量基因 / 零残差方差：R 测试样例报错，Python 默认也报错。
  `degenerate_policy="conservative"` 赋 p=1 是显式扩展，不声称复现 R 的无定义结果。
- 秩不足：R 求逆失败，Python 默认报错。`rank_policy="drop"` 是显式扩展。
- 原 R 主函数对只有一个基因的某些输入会发生维度/S4 类型错误；Python 支持单基因。
  这是输入处理修复，不是一个可直接比较两套最终结果的 R 输入。
- 这批测试中 FDR 判定一致，不代表任何数据在阈值附近的判定都不会受积分误差影响。

结构化报告：`reference/actual_r_validation.json`。
实际 R 产生的数据保存在测试 fixture `tests/fixtures/actual_r_reference.npz`，
可在没有 R 的环境中运行回归测试。

重新执行真实 R 对照：

```bash
python reference/validate_actual_r.py --rscript /path/to/Rscript
```

R 侧需要 Matrix、Rcpp、RcppArmadillo、CompQuadForm。
这些只用于验证；使用 Python 包和 GPU 后端均不依赖 R。
