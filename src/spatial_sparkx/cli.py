"""Command-line interface for AnnData files."""

import argparse
from pathlib import Path

from .anndata import sparkx_adata


def main(argv=None):
    parser = argparse.ArgumentParser(description="Native Python SPARK-X on an .h5ad file")
    parser.add_argument("input", type=Path, help="Input .h5ad file")
    parser.add_argument("--output", type=Path, required=True, help="Result table (.csv or .tsv)")
    parser.add_argument("--write-h5ad", type=Path, help="Optional annotated .h5ad output")
    parser.add_argument("--spatial-key", default="spatial")
    parser.add_argument("--layer", help="Expression layer; otherwise uses X")
    parser.add_argument("--option", choices=("single", "mixture"), default="mixture")
    parser.add_argument("--backend", choices=("numpy", "cupy"), default="numpy")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--n-threads", type=int)
    parser.add_argument("--rank-policy", choices=("raise", "drop"), default="raise")
    parser.add_argument("--degenerate-policy", choices=("raise", "conservative"), default="raise")
    args = parser.parse_args(argv)
    if args.output.suffix.lower() not in (".csv", ".tsv"):
        parser.error("--output must end in .csv or .tsv")
    if args.write_h5ad is not None and args.write_h5ad.suffix.lower() != ".h5ad":
        parser.error("--write-h5ad must end in .h5ad")
    if args.input.resolve() == args.output.resolve() or (
        args.write_h5ad is not None and args.input.resolve() == args.write_h5ad.resolve()
    ):
        parser.error("output paths must differ from the input file")
    try:
        import anndata as ad
    except ImportError:
        parser.error("AnnData support requires: pip install 'spatial-sparkx[anndata]'")
    adata = ad.read_h5ad(args.input)
    result = sparkx_adata(
        adata, spatial_key=args.spatial_key, layer=args.layer, option=args.option,
        backend=args.backend, batch_size=args.batch_size, n_threads=args.n_threads,
        rank_policy=args.rank_policy, degenerate_policy=args.degenerate_policy,
        inplace=args.write_h5ad is not None,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.res_mtest.to_csv(args.output, sep="\t" if args.output.suffix.lower() == ".tsv" else ",", index_label="gene")
    if args.write_h5ad is not None:
        args.write_h5ad.parent.mkdir(parents=True, exist_ok=True)
        adata.write_h5ad(args.write_h5ad)
    print(f"Tested {result.gene_mask.sum()} genes across {result.cell_mask.sum()} observations")
    print(f"Results: {args.output}")


if __name__ == "__main__":
    main()
