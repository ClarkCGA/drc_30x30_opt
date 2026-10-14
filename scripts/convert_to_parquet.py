"""
One-off conversion of the planning-unit CSV (~1.5 GB) to Parquet.

Parquet lets the optimizer read only the columns a scenario needs, which is
much faster and lighter than parsing the full CSV for every run.

Usage (from the repository root):
    python -m scripts.convert_to_parquet \
        --csv data/drc_1km_data_planning_units.csv \
        --parquet data/drc_1km_data_planning_units.parquet

The CSV is read in chunks with a fixed schema (all columns float64 except the
string `neighbors` column), so memory use stays bounded. The legacy column
name `cf_frac` is renamed to `cfcl_frac` on the way.
"""

import argparse

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from scripts.utils import COLUMN_RENAMES

STRING_COLS = {"neighbors"}


def convert(csv_path, parquet_path, chunksize=250_000):
    header = pd.read_csv(csv_path, nrows=0).columns.tolist()
    dtypes = {c: (str if c in STRING_COLS else "float64") for c in header}
    out_cols = [COLUMN_RENAMES.get(c, c) for c in header]
    schema = pa.schema(
        [(c, pa.string() if c in STRING_COLS else pa.float64()) for c in out_cols]
    )

    n = 0
    with pq.ParquetWriter(parquet_path, schema, compression="zstd") as writer:
        for chunk in pd.read_csv(csv_path, dtype=dtypes, chunksize=chunksize):
            chunk.columns = out_cols
            writer.write_table(pa.Table.from_pandas(chunk, schema=schema, preserve_index=False))
            n += len(chunk)
            print(f"  {n:,} rows written", flush=True)
    print(f"Parquet written to {parquet_path} ({n:,} rows, {len(out_cols)} columns)")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", default="data/drc_1km_data_planning_units.csv")
    p.add_argument("--parquet", default="data/drc_1km_data_planning_units.parquet")
    p.add_argument("--chunksize", type=int, default=250_000)
    args = p.parse_args()
    convert(args.csv, args.parquet, args.chunksize)
