"""
Per-scenario optimization pipeline: load data, compute the thematically
weighted score, identify candidates, solve, and export maps, GeoTIFFs and a
summary row (incl. score diagnostics and oil-block overlap).

Used by main_opt.ipynb and by scripts/test_pipeline.py.
"""

from pathlib import Path
import time

from pulp import LpStatus, value

from scripts import utils


def run_scenario(config_path, bbox=None, output_dir="output", threads=None,
                 time_limit=None, connectivity=None, verbose=True):
    """
    Run the full optimization pipeline for a single config file and export its outputs.
    Parameters
    ----------
    config_path : str or Path
        Scenario config (merged with the shared files it `extends`).
    bbox : dict, optional
        Spatial subset {"xmin", "xmax", "ymin", "ymax"} in native coordinates,
        for test runs. The 30% area cap then applies to the subset's area.
    output_dir : str, optional
        Where maps, GeoTIFFs and scenario_summary.csv are written.
    threads, time_limit, connectivity : optional
        Override the solver/model settings of the config (optimization.solver,
        optimization.connectivity).
    Returns
    -------
    dict
        The summary row written to <output_dir>/scenario_summary.csv.
    """

    start = time.time()
    config = utils.load_config(config_path)
    name = config["name"]
    opt = config["optimization"]
    solver_cfg = opt.get("solver", {})
    threads = threads or solver_cfg.get("threads", 4)
    time_limit = time_limit or solver_cfg.get("time_limit", 2000)
    connectivity = opt.get("connectivity", True) if connectivity is None else connectivity
    protection_frac = opt.get("protection_frac", 0.30)

    if verbose:
        print(f"=== Running scenario: {name} ({config_path}) ===")

    drc_df = utils.load_data(config, connectivity=connectivity, bbox=bbox)

    # Thematically weighted multi-objective score
    weights = utils.compute_weights(config)
    if verbose:
        print(weights.to_string(index=False))
    max_part, min_part, _ = utils.score_components(drc_df, weights)
    drc_df["objective_score"] = max_part - min_part
    diag = utils.score_diagnostics(max_part, min_part, eligible=utils.eligible_mask(drc_df, config))
    if verbose:
        print(f"Score > 0 in {diag['score_pos_pct']:.1f}% of cells; minimize side >= maximize side "
              f"in {diag['min_gt_max_pct']:.1f}% of cells "
              f"({diag['eligible_min_gt_max_pct']:.1f}% of cells passing the exclusion rules)")
    utils.visualize(drc_df,
                    "objective_score",
                    title=f"{name} - Objective Score",
                    file_name=f"{name} - Objective Score.png",
                    show=False, output_dir=output_dir)

    # Candidate planning units
    drc_df = utils.iden_units(drc_df, config)
    utils.visualize(drc_df,
                    "candidate",
                    colorbar=False,
                    title=f"{name} - Candidate Spatial Units for Protection",
                    file_name=f"{name} - Candidates.png",
                    show=False, output_dir=output_dir)

    # Optimize
    model, x = utils.setup_model(drc_df, protection_frac=protection_frac, connectivity=connectivity)
    drc_df = utils.solve_model(model, x, drc_df, threads=threads, time_limit=time_limit,
                               gap_rel=solver_cfg.get("gap_rel", 0.001))

    used_area = float((drc_df["area"] * drc_df["selected"]).sum())
    total_area = float(drc_df["area"].sum())
    baseline_area = float(drc_df.loc[drc_df["protected"] == 1, "area"].sum())
    if verbose:
        print(f"Selected area (baseline + new): {used_area:,.0f} km2 ({used_area / total_area:.2%} of total)")

    # Selected areas to protect
    utils.visualize(drc_df,
                    "selected",
                    colorbar=False,
                    file_name=f"{name}.png",
                    title=f"{name} - Selected Areas to Protect",
                    show=False, output_dir=output_dir)
    utils.export_to_geotiff(drc_df, "selected", f"{name}.tif", output_dir=output_dir)

    # Baseline (existing protected areas, plus CFCLs in the PAs+CFCLs runs)
    utils.visualize(drc_df,
                    "protected",
                    colorbar=False,
                    file_name=f"{name} - Baseline Areas.png",
                    title=f"{name} - Baseline Areas ({config.get('baseline', '')})",
                    show=False, output_dir=output_dir)
    utils.export_to_geotiff(drc_df, "protected", f"{name} - Baseline Areas.tif", output_dir=output_dir)

    oil = utils.oil_overlap(drc_df)
    if verbose:
        print(f"Newly selected: {oil['new_area_km2']:,.0f} km2, of which {oil['new_in_oil_km2']:,.0f} km2 "
              f"({oil['new_in_oil_pct']:.1f}%) inside oil blocks")

    row = {
        "name": name,
        "config_path": str(config_path),
        "baseline": config.get("baseline"),
        "species": utils.species_included(config),
        "priority": ";".join(f"{k}x{v:g}" for k, v in (config["weighting"].get("priority") or {}).items()) or "none",
        "subset": "full" if bbox is None else "bbox",
        "status": LpStatus[model.status],
        "objective": value(model.objective),
        "total_area_km2": total_area,
        "baseline_area_km2": baseline_area,
        "selected_area_km2": used_area,
        "selected_pct": used_area / total_area * 100 if total_area else float("nan"),
        "n_candidates": int(drc_df["candidate"].sum()),
        **oil,
        "max_weight_sum": float(weights.loc[weights["direction"] > 0, "weight"].sum()),
        "min_weight_sum": float(weights.loc[weights["direction"] < 0, "weight"].sum()),
        **diag,
        "elapsed_sec": round(time.time() - start, 1),
    }
    utils.upsert_summary(row, Path(output_dir) / "scenario_summary.csv")
    return row
