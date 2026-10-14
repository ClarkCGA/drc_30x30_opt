"""
Test the optimization pipeline before the real STAR data is available.

  1. Unit checks: theme weights follow the balancing rules, and the chunked
     STAR reader matches a plain in-memory sum.
  2. Score diagnostics on the FULL grid for every scenario (no solve): how
     many cells have a positive score / pass the exclusion rules, and whether
     the minimize side dominates the maximize side.
  3. End-to-end runs of every scenario on a small spatial subset (solve,
     maps, GeoTIFFs, summary CSV incl. oil overlap) and the consensus map.

Usage (from the repository root; uses the dummy STAR file by default):
    python -m scripts.make_dummy_star          # once
    python -m scripts.test_pipeline --out-dir output_test
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_BBOX = {"xmin": 2_025_000, "xmax": 2_175_000, "ymin": 75_000, "ymax": 225_000}


def check_weights(config_paths):
    from scripts import utils
    for p in config_paths:
        c = utils.load_config(p)
        w = utils.compute_weights(c)
        prio = c["weighting"].get("priority") or {}
        mx = w[w["direction"] > 0].groupby("theme")["weight"].sum()
        for theme, share in mx.items():
            expected = float(prio.get(theme, 1.0))
            assert np.isclose(share, expected), f"{p}: theme {theme} share {share} != {expected}"
        mn = w[w["direction"] < 0]
        expected_min = mn["theme"].map(lambda t: float(prio.get(t, 1.0)))
        assert np.allclose(mn["weight"], expected_min), f"{p}: minimize weights != priority multiplier"
        bio = w[w["theme"] == "biodiversity"].set_index("column")["weight"]
        if utils.species_included(c):
            assert np.isclose(bio["kba_frac"], bio["star_total"]), f"{p}: KBA/species split not 50/50"
        else:
            assert list(bio.index) == ["kba_frac"], f"{p}: no-species run still has species"
    print(f"[ok] theme weights for {len(config_paths)} configs")


def check_star_reader(star_path):
    from scripts import utils
    ref = pd.read_parquet(star_path)
    ref_total = ref.filter(like=utils.SPECIES_PREFIX).to_numpy(np.float64).sum(axis=1)
    for budget in (512, 1):
        t = utils.load_star_total(star_path, memory_budget_mb=budget)
        assert (t["grid_id"].to_numpy() == ref["grid_id"].to_numpy()).all()
        assert np.allclose(t[utils.SPECIES_COL].to_numpy(), ref_total)
    print("[ok] chunked STAR total matches in-memory sum")


def diagnostics(config_paths, out_dir):
    from scripts import utils
    rows = []
    for p in config_paths:
        c = utils.load_config(p)
        df = utils.load_data(c, connectivity=False)
        w = utils.compute_weights(c)
        max_part, min_part, themes = utils.score_components(df, w)
        eligible = utils.eligible_mask(df, c)
        d = utils.score_diagnostics(max_part, min_part, eligible=eligible)
        non_base = eligible & (df["protected"] == 0)
        score = max_part - min_part
        d.update({
            "name": c["name"],
            "max_weight_sum": w.loc[w["direction"] > 0, "weight"].sum(),
            "min_weight_sum": w.loc[w["direction"] < 0, "weight"].sum(),
            "baseline_cells": int(df["protected"].sum()),
            "eligible_non_baseline_cells": int(non_base.sum()),
            "candidates_non_baseline": int((non_base & (score > 0)).sum()),
            "budget_new_cells": int(c["optimization"]["protection_frac"] * len(df) - df["protected"].sum()),
        })
        d.update({f"mean_{k}": float(v.mean()) for k, v in themes.items()})
        rows.append(d)
        print(f"{c['name']:<50} score>0 {d['score_pos_pct']:5.1f}%  "
              f"eligible non-baseline candidates {d['candidates_non_baseline']:,} "
              f"(budget {d['budget_new_cells']:,} new cells)")
    out = pd.DataFrame(rows).set_index("name")
    out_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_dir / "score_diagnostics_full_grid.csv")
    return out


def end_to_end(config_paths, out_dir, bbox, configs_dir):
    from scripts.pipeline import run_scenario
    from scripts.consensus import build_consensus
    for p in config_paths:
        run_scenario(p, bbox=bbox, output_dir=str(out_dir), threads=4, verbose=False)
    summary = pd.read_csv(out_dir / "scenario_summary.csv")
    print(summary[["name", "status", "selected_pct", "new_area_km2", "new_in_oil_km2",
                   "new_in_oil_pct", "score_pos_pct", "eligible_min_gt_max_pct"]].to_string(index=False))
    assert (summary["status"] == "Optimal").all(), "Not all runs solved to optimality"
    assert (summary["selected_pct"] <= 30.0 + 1e-6).all(), "Area cap violated"
    build_consensus(configs_dir, str(out_dir), show=False)
    print(f"[ok] end-to-end runs and consensus map written to {out_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--configs-dir", default="configs")
    p.add_argument("--star", default="data/star_dummy/star_1km_species.parquet",
                   help="STAR Parquet to use (sets DRC_STAR_PATH)")
    p.add_argument("--out-dir", default="output_test")
    p.add_argument("--skip-diagnostics", action="store_true")
    p.add_argument("--skip-e2e", action="store_true")
    args = p.parse_args()

    os.environ["DRC_STAR_PATH"] = args.star
    configs = sorted(Path(args.configs_dir).glob("*.yml"))
    out_dir = Path(args.out_dir)

    check_weights(configs)
    check_star_reader(args.star)
    if not args.skip_diagnostics:
        diagnostics(configs, out_dir)
    if not args.skip_e2e:
        end_to_end(configs, out_dir, DEFAULT_BBOX, args.configs_dir)
