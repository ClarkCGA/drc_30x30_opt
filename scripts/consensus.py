"""
Consensus map across scenario runs, with oil-block outlines as an overlay.

For every scenario config, the run's selection GeoTIFF (`<name>.tif`) and
baseline GeoTIFF (`<name> - Baseline Areas.tif`) are read from the output
folder. The consensus value of a planning unit is the number of runs in which
it is *newly* selected (selected and not part of that run's baseline). Units
that are baseline in any run (PAs; CFCLs in the PAs+CFCLs runs) are shown as
a separate class. Oil-block outlines are drawn from `oil_frac` > 0.

Used by visualize_consensus.ipynb and scripts/test_pipeline.py.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import rasterio

from scripts import utils


def scenario_names(configs_dir="configs"):
    """Scenario names (config `name`) of every *.yml/*.yaml in configs_dir, in file order."""
    paths = sorted([*Path(configs_dir).glob("*.yml"), *Path(configs_dir).glob("*.yaml")])
    names = []
    for p in paths:
        name = utils.load_config(p).get("name")
        if isinstance(name, str) and name.strip():
            names.append(name.strip())
        else:
            print(f"Warning: missing/invalid 'name' in {p.name}")
    return names


def _read_mask(path):
    with rasterio.open(path) as src:
        arr = src.read(1, masked=True)
        valid = ~np.ma.getmaskarray(arr)
        return (np.ma.filled(arr, 0) > 0) & valid, valid, src.transform, src.crs, src.profile


def build_consensus(configs_dir="configs", output_dir="output", oil_overlay=True,
                    planning_units=None, file_name="consensus_map", show=True):
    """
    Build, plot and save the consensus map across all scenario runs.
    Parameters
    ----------
    configs_dir : str
        Folder with the scenario configs (their `name` identifies the GeoTIFFs).
    output_dir : str
        Folder with the run GeoTIFFs; the map (<file_name>.png) and the
        consensus GeoTIFF (<file_name>.tif) are written here.
    oil_overlay : bool
        Draw oil-block outlines (from `oil_frac` of the planning-unit table).
    planning_units : str, optional
        Planning-unit table for the oil overlay; defaults to the data.file_path
        of the first config.
    Returns
    -------
    numpy.ma.MaskedArray
        Consensus counts (0..n_runs); masked outside the study area and on
        baseline units.
    """

    output_dir = Path(output_dir)
    names = scenario_names(configs_dir)
    found = [n for n in names if (output_dir / f"{n}.tif").exists()
             and (output_dir / f"{n} - Baseline Areas.tif").exists()]
    missing = sorted(set(names) - set(found))
    if missing:
        print(f"Warning: {len(missing)} scenario(s) have no outputs in {output_dir}: {missing}")
    if not found:
        raise FileNotFoundError(f"No scenario GeoTIFFs found in {output_dir} for configs in {configs_dir}.")

    counts = baseline_any = valid_all = None
    ref = None
    for n in found:
        sel, valid, transform, crs, profile = _read_mask(output_dir / f"{n}.tif")
        base, _, b_transform, _, _ = _read_mask(output_dir / f"{n} - Baseline Areas.tif")
        if ref is None:
            ref = (sel.shape, transform, crs, profile)
            counts = np.zeros(sel.shape, np.int16)
            baseline_any = np.zeros(sel.shape, bool)
            valid_all = valid.copy()
        elif sel.shape != ref[0] or transform != ref[1] or crs != ref[2]:
            raise ValueError(f"Raster alignment mismatch: {n}.tif")
        if b_transform != transform or base.shape != sel.shape:
            raise ValueError(f"Raster alignment mismatch: {n} - Baseline Areas.tif")
        counts += (sel & ~base).astype(np.int16)
        baseline_any |= base
        valid_all &= valid

    shape, transform, crs, profile = ref
    n_runs = len(found)
    print(f"Consensus across {n_runs} run(s):")
    for n in found:
        print(f" - {n}")

    # Save the consensus counts (-1 = baseline unit, nodata outside the study area)
    out = np.where(baseline_any, -1, counts).astype(np.float32)
    out[~valid_all] = np.nan
    profile = {**profile, "dtype": "float32", "nodata": np.nan, "count": 1}
    with rasterio.open(output_dir / f"{file_name}.tif", "w", **profile) as dst:
        dst.write(out, 1)

    # Plot
    left, bottom, right, top = rasterio.transform.array_bounds(shape[0], shape[1], transform)
    extent = [left, right, bottom, top]
    fig = plt.figure(figsize=(8, 6))
    title_ax = fig.add_axes([0.2, 0.91, 0.65, 0.06])
    title_ax.axis("off")
    title_ax.text(0.5, 0.5, f"Agreement on Priority Areas across {n_runs} Scenarios (new areas beyond the baseline)",
                  ha="center", va="center", wrap=True)
    ax = fig.add_axes([0.10, 0.10, 0.75, 0.75])

    colors = ["#D2D2D2B9"] + [plt.cm.viridis(i / max(n_runs, 1)) for i in range(1, n_runs + 1)]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(np.arange(-0.5, n_runs + 1.5, 1), cmap.N)
    im = ax.imshow(np.ma.array(counts, mask=~valid_all | baseline_any), extent=extent,
                   cmap=cmap, norm=norm, interpolation="nearest")
    ax.imshow(np.ma.array(np.ones(shape), mask=~(baseline_any & valid_all)), extent=extent,
              cmap=ListedColormap(["#8C6D4F"]), interpolation="nearest")
    handles = [Patch(color="#8C6D4F", label="Baseline (PAs; + CFCLs in PAs+CFCLs runs)")]

    if oil_overlay:
        pu_path = planning_units or utils.load_config(
            sorted(Path(configs_dir).glob("*.y*ml"))[0])["data"]["file_path"]
        bbox = {"xmin": left, "xmax": right, "ymin": bottom, "ymax": top}
        oil = utils.read_planning_units(pu_path, ["grid_id", "x_native", "y_native", utils.OIL_COL], bbox=bbox)
        oil["oil"] = (oil[utils.OIL_COL].fillna(0) > 0).astype(float)
        arr, xs, ys = utils.grid_array(oil, "oil")
        arr = np.nan_to_num(arr)
        # Contours at cell centres (ys ascending -> origin lower)
        ax.contour(xs, ys, arr, levels=[0.5], colors="red", linewidths=0.6)
        handles.append(Line2D([0], [0], color="red", lw=1, label="Oil blocks"))

    ax.set_xlim(left, right)
    ax.set_ylim(bottom, top)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_aspect("equal")
    ax.legend(handles=handles, loc="lower left", fontsize=7, framealpha=0.9)

    cax = fig.add_axes([0.88, 0.10, 0.03, 0.75])
    cbar = fig.colorbar(im, cax=cax, ticks=np.arange(0, n_runs + 1))
    cbar.set_label("Number of scenarios in which the unit is a new priority area")

    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / f"{file_name}.png", dpi=300)
    if show:
        plt.show()
    else:
        plt.close(fig)

    levels, cnts = np.unique(counts[valid_all & ~baseline_any], return_counts=True)
    print("\nPixel counts by agreement level (excluding baseline units):")
    for lvl, cnt in zip(levels, cnts):
        print(f"{lvl}: {cnt}")
    print(f"Baseline units: {int((baseline_any & valid_all).sum())}")

    return np.ma.array(counts, mask=~valid_all | baseline_any)
