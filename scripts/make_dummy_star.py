"""
Write a small synthetic STAR species Parquet (and species table) in the same
format as the real STAR pipeline output, for testing the optimizer before the
real data is ready. The output is gitignored (data/star_dummy/).

Format (matching drc_30x30_data_prep/star):
  - star_1km_species.parquet: `grid_id` (int64) + one float32 column
    `sp_<id_no>` per species, written in row groups of 50,000 cells. Values
    are per-cell STAR scores, already multiplied by the category weight.
  - star_species.csv: id_no, name, class, category, weight, global_km2,
    drc_km2, source, included.

Each synthetic species has a circular range of random size and position;
its per-cell value is weight * 100 / global_km2 inside the range.

Usage (from the repository root):
    python -m scripts.make_dummy_star --n-species 300
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from scripts.utils import read_planning_units

CATEGORY_WEIGHTS = {"LC": 0.0, "NT": 1.0, "VU": 2.0, "EN": 3.0, "CR": 4.0}
CLASSES = ["MAMMALIA", "AVES", "AMPHIBIA", "REPTILIA", "ACTINOPTERYGII"]


def make_dummy(planning_units, out_dir, n_species=300, row_group=50_000, seed=42, bbox=None):
    rng = np.random.default_rng(seed)
    pu = read_planning_units(planning_units, ["grid_id", "x_native", "y_native"], bbox=bbox)
    pu = pu.sort_values("grid_id").reset_index(drop=True)
    x, y = pu["x_native"].to_numpy(), pu["y_native"].to_numpy()

    # Species: circular ranges (radius 10-300 km) centred on random planning units.
    centres = rng.integers(0, len(pu), n_species)
    cx, cy = x[centres], y[centres]
    radius = rng.uniform(10_000, 300_000, n_species)
    categories = rng.choice(list(CATEGORY_WEIGHTS), n_species, p=[0.0, 0.3, 0.35, 0.25, 0.1])
    weights = np.array([CATEGORY_WEIGHTS[c] for c in categories], dtype=np.float32)
    drc_km2 = np.zeros(n_species)
    for j in range(n_species):  # cells (km2) inside each range
        drc_km2[j] = np.count_nonzero((x - cx[j]) ** 2 + (y - cy[j]) ** 2 <= radius[j] ** 2)
    global_km2 = drc_km2 * rng.uniform(1.0, 5.0, n_species)  # part of the range lies outside DRC
    keep = drc_km2 > 0
    id_no = np.sort(rng.choice(np.arange(1_000, 10_000_000), n_species, replace=False))
    value = np.where(keep, weights * 100.0 / np.maximum(global_km2, 1), 0).astype(np.float32)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "id_no": id_no,
        "name": [f"Dummy species {i}" for i in id_no],
        "class": rng.choice(CLASSES, n_species),
        "category": categories,
        "weight": weights,
        "global_km2": global_km2.round(1),
        "drc_km2": drc_km2,
        "source": "synthetic",
        "included": keep,
    }).to_csv(out_dir / "star_species.csv", index=False)

    cols = [f"sp_{i}" for i in id_no[keep]]
    sel = np.flatnonzero(keep)
    schema = pa.schema([("grid_id", pa.int64())] + [(c, pa.float32()) for c in cols])
    with pq.ParquetWriter(out_dir / "star_1km_species.parquet", schema, compression="zstd") as w:
        for lo in range(0, len(pu), row_group):
            hi = min(lo + row_group, len(pu))
            xs, ys = x[lo:hi], y[lo:hi]
            arrays = [pa.array(pu["grid_id"].to_numpy()[lo:hi])]
            for j in sel:
                inside = (xs - cx[j]) ** 2 + (ys - cy[j]) ** 2 <= radius[j] ** 2
                arrays.append(pa.array(np.where(inside, value[j], 0).astype(np.float32)))
            w.write_table(pa.Table.from_arrays(arrays, schema=schema))
    print(f"Dummy STAR written to {out_dir}: {len(cols)} species x {len(pu):,} cells")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--planning-units", default="data/drc_1km_data_planning_units.parquet")
    p.add_argument("--out-dir", default="data/star_dummy")
    p.add_argument("--n-species", type=int, default=300)
    p.add_argument("--row-group", type=int, default=50_000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    make_dummy(args.planning_units, args.out_dir, args.n_species, args.row_group, args.seed)
