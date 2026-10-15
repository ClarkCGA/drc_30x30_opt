import os
import copy
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pulp import LpProblem, LpVariable, LpMaximize, LpAffineExpression, LpBinary, PULP_CBC_CMD, LpStatus
import pyarrow.parquet as pq
import rasterio
from rasterio.transform import Affine
from rasterio.crs import CRS
import re
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]

# Legacy input column names -> names used in the code.
# CFCL = Concession Forestiere de Communaute Locale (Local Community Forest Concession).
COLUMN_RENAMES = {"cf_frac": "cfcl_frac"}

# Per-cell STAR total (sum of the sp_<id_no> columns of the STAR Parquet).
SPECIES_COL = "star_total"
SPECIES_PREFIX = "sp_"

# Columns always read from the planning-unit table.
ID_COLS = ["grid_id", "x_native", "y_native"]
OIL_COL = "oil_frac"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def _resolve(path):
    """Resolve a path relative to the repository root (absolute paths are kept)."""
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def _deep_merge(base, override):
    """Recursively merge `override` into a copy of `base` (dicts merge, everything else replaces)."""
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(config_path):
    """
    Load a scenario config and merge in the shared files it `extends`.
    Each file listed under `extends` (paths relative to the repository root) is
    deep-merged in order, and the scenario's own keys are merged last so they
    take precedence. The environment variables DRC_STAR_PATH,
    DRC_STAR_SPECIES_TABLE and DRC_STAR_SUMMARY, when set, override
    species.star_path, species.species_table and species.summary_path. The shared files hold the theme definitions
    (configs/common/themes.yml) and the defaults common to all scenarios
    (configs/common/defaults.yml: data paths, exclusion rules, solver settings).
    Parameters
    ----------
    config_path : str or Path
        Path to the scenario YAML file.
    Returns
    -------
    dict
        The merged configuration.
    """

    with open(config_path, "r") as f:
        scenario = yaml.safe_load(f) or {}

    config = {}
    for extra in scenario.get("extends", []):
        with open(_resolve(extra), "r") as f:
            config = _deep_merge(config, yaml.safe_load(f) or {})
    config = _deep_merge(config, {k: v for k, v in scenario.items() if k != "extends"})
    config["config_path"] = str(config_path)

    # Environment overrides for the STAR inputs (e.g. to point at a test file).
    for env, key in (("DRC_STAR_PATH", "star_path"), ("DRC_STAR_SPECIES_TABLE", "species_table"),
                     ("DRC_STAR_SUMMARY", "summary_path")):
        if os.environ.get(env):
            config.setdefault("species", {})[key] = os.environ[env]
    return config


def species_included(config):
    """True when the scenario uses the STAR species layer."""
    return bool(config.get("species", {}).get("include", False))


def constraint_cols(config):
    """Exclusion rules: shared `constraint_cols` plus the scenario's `extra_constraint_cols`."""
    opt = config["optimization"]
    return list(opt.get("constraint_cols", [])) + list(opt.get("extra_constraint_cols", []))


# ---------------------------------------------------------------------------
# Thematic weighting
# ---------------------------------------------------------------------------

def compute_weights(config, species_available=None):
    """
    Build the per-layer weights from the theme definitions.
    Rules:
      - Each theme gets a share of `weighting.base_share` (1.0) times its
        priority multiplier (`weighting.priority`, e.g. {"biodiversity": 2.0}),
        on the maximize side and, separately, on the minimize side.
      - The maximize share is split equally across the theme's maximize
        layers, or according to the theme's explicit `shares` (fractions of
        the share). A layer that is unavailable in this run (the STAR species
        layer in a no-species run) is dropped and the remaining shares are
        renormalized.
      - The minimize share is split equally across the theme's minimize layers.
      - A theme without layers on one side contributes 0 on that side.
    Parameters
    ----------
    config : dict
        Merged configuration (see load_config) with `themes` and `weighting`.
    species_available : bool, optional
        Whether the STAR species layer is used. Defaults to species_included(config).
    Returns
    -------
    pandas.DataFrame
        One row per layer with columns: theme, column, direction (+1 maximize,
        -1 minimize), weight, theme_share.
    """

    if species_available is None:
        species_available = species_included(config)

    themes = config["themes"]
    w_cfg = config.get("weighting", {})
    base_share = float(w_cfg.get("base_share", 1.0))
    priority = w_cfg.get("priority") or {}

    unknown = set(priority) - set(themes)
    assert not unknown, f"Unknown theme(s) in weighting.priority: {sorted(unknown)}"

    rows = []
    for key, theme in themes.items():
        multiplier = float(priority.get(key, 1.0))
        share = base_share * multiplier
        max_cols = [c for c in theme.get("maximize", []) if species_available or c != SPECIES_COL]
        shares = theme.get("shares") or {}
        if shares:
            missing = [c for c in max_cols if c not in shares]
            assert not missing, f"Theme '{key}' lists shares but not for: {missing}"
            raw = {c: float(shares[c]) for c in max_cols}
        else:
            raw = {c: 1.0 for c in max_cols}
        total = sum(raw.values())
        for c in max_cols:
            rows.append({"theme": key, "column": c, "direction": 1.0,
                         "weight": share * raw[c] / total, "theme_share": share})
        min_cols = theme.get("minimize", [])
        for c in min_cols:
            rows.append({"theme": key, "column": c, "direction": -1.0,
                         "weight": share / len(min_cols), "theme_share": share})

    weights = pd.DataFrame(rows)
    dup = weights["column"][weights["column"].duplicated()].tolist()
    assert not dup, f"Layer(s) assigned to more than one theme/direction: {dup}"
    return weights


def required_columns(config):
    """Columns of the planning-unit table needed to run a scenario."""
    weights = compute_weights(config)
    cols = list(ID_COLS)
    cols += list(config["data"]["protected_classes"])
    cols += [c for c, _ in constraint_cols(config)]
    cols += [c for c in weights["column"] if c != SPECIES_COL]
    cols.append(OIL_COL)
    return list(dict.fromkeys(cols))  # de-duplicate, keep order


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def read_planning_units(file_path, columns, bbox=None):
    """
    Read selected columns of the planning-unit table (Parquet or CSV).
    Only the requested columns are read. If `file_path` is a .parquet file
    that does not exist yet but a .csv with the same stem does, the CSV is
    read instead (convert it once with `python -m scripts.convert_to_parquet`
    to speed this up). Legacy column names (COLUMN_RENAMES, e.g. cf_frac) are
    accepted and renamed (to cfcl_frac).
    Parameters
    ----------
    file_path : str or Path
        Path to the planning-unit table.
    columns : list[str]
        Column names (new names) to read.
    bbox : dict, optional
        Spatial subset in native coordinates: {"xmin", "xmax", "ymin", "ymax"}.
    Returns
    -------
    pandas.DataFrame
    """

    path = _resolve(file_path)
    if path.suffix == ".parquet" and not path.exists() and path.with_suffix(".csv").exists():
        print(f"{path.name} not found; reading {path.with_suffix('.csv').name} instead "
              f"(run `python -m scripts.convert_to_parquet` once to speed this up)")
        path = path.with_suffix(".csv")

    if path.suffix == ".parquet":
        available = pq.ParquetFile(path).schema_arrow.names
    else:
        available = pd.read_csv(path, nrows=0).columns.tolist()
    inverse = {new: old for old, new in COLUMN_RENAMES.items()}
    read_cols = [c if c in available else inverse.get(c, c) for c in columns]
    missing = [c for c in read_cols if c not in available]
    assert not missing, f"Missing columns in {path.name}: {missing}"

    if path.suffix == ".parquet":
        filters = None
        if bbox:
            filters = [("x_native", ">=", bbox["xmin"]), ("x_native", "<=", bbox["xmax"]),
                       ("y_native", ">=", bbox["ymin"]), ("y_native", "<=", bbox["ymax"])]
        df = pd.read_parquet(path, columns=read_cols, filters=filters)
    else:
        df = pd.read_csv(path, usecols=read_cols)
        if bbox:
            df = df[df["x_native"].between(bbox["xmin"], bbox["xmax"])
                    & df["y_native"].between(bbox["ymin"], bbox["ymax"])]
    df = df.rename(columns=COLUMN_RENAMES).reset_index(drop=True)
    df["grid_id"] = df["grid_id"].astype(np.int64)
    return df


def load_star_total(star_path, memory_budget_mb=512):
    """
    Sum the STAR species columns (sp_<id_no>) of a wide Parquet file into a
    single per-cell STAR total, without loading all columns at once.
    The file is processed one row group at a time, and within a row group in
    batches of columns sized so that each batch stays below `memory_budget_mb`.
    The per-species values already include the IUCN category weights, so the
    plain sum is the STAR score of the cell. If the file has no sp_ columns
    but has a precomputed `star_total` column, that column is used directly.
    Parameters
    ----------
    star_path : str or Path
        Path to the STAR Parquet (key column `grid_id`).
    memory_budget_mb : int, optional
        Approximate memory budget for one column batch. Default 512 MB.
    Returns
    -------
    pandas.DataFrame
        Columns `grid_id` (int64) and `star_total` (float64), one row per cell
        in the file.
    """

    pf = pq.ParquetFile(_resolve(star_path))
    names = pf.schema_arrow.names
    assert "grid_id" in names, f"No grid_id column in {star_path}"
    sp_cols = [c for c in names if c.startswith(SPECIES_PREFIX)]

    if not sp_cols:
        assert SPECIES_COL in names, f"No {SPECIES_PREFIX}* or {SPECIES_COL} columns in {star_path}"
        t = pf.read(columns=["grid_id", SPECIES_COL]).to_pandas()
        return t.astype({"grid_id": np.int64, SPECIES_COL: np.float64})

    parts = []
    for rg in range(pf.num_row_groups):
        n_rows = pf.metadata.row_group(rg).num_rows
        batch = max(1, int(memory_budget_mb * 2**20 // (max(n_rows, 1) * 8)))
        gid = pf.read_row_group(rg, columns=["grid_id"]).column(0).to_numpy()
        total = np.zeros(n_rows, dtype=np.float64)
        for start in range(0, len(sp_cols), batch):
            cols = sp_cols[start:start + batch]
            t = pf.read_row_group(rg, columns=cols)
            for col in t.columns:
                total += col.to_numpy(zero_copy_only=False).astype(np.float64, copy=False)
            del t
        parts.append(pd.DataFrame({"grid_id": gid.astype(np.int64), SPECIES_COL: total}))

    out = pd.concat(parts, ignore_index=True)
    print(f"STAR total computed from {len(sp_cols):,} species columns over {len(out):,} cells "
          f"({pf.num_row_groups} row group(s))")
    return out


def summarize_species_table(table_path, star_path=None):
    """
    Print a short summary of the STAR species table (CSV or Parquet) and check
    that its species match the sp_<id_no> columns of the STAR Parquet.
    Expected columns include id_no, name (or sci_name), class, category,
    weight, global_km2, drc_km2, source; an `included` column, if present,
    restricts the check to included species.
    """

    path = _resolve(table_path)
    sp = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    if "included" in sp.columns:
        sp = sp[sp["included"].astype(bool)]
    print(f"Species table: {len(sp):,} species")
    for c in ("class", "category"):
        if c in sp.columns:
            print(f"  by {c}: " + ", ".join(f"{k}={v}" for k, v in sp[c].value_counts().items()))
    if star_path is not None:
        cols = {c for c in pq.ParquetFile(_resolve(star_path)).schema_arrow.names
                if c.startswith(SPECIES_PREFIX)}
        ids = {f"{SPECIES_PREFIX}{int(i)}" for i in sp["id_no"]}
        if cols and cols != ids:
            print(f"  WARNING: {len(ids - cols)} species in the table have no column, "
                  f"{len(cols - ids)} columns are not in the table")
    return sp


def load_data(config, bbox=None):
    """
    Load the planning units needed by a scenario, add metadata columns and,
    if the scenario uses species, the per-cell STAR total.
    Parameters
    ----------
    config : dict
        Merged configuration (see load_config). Uses:
          - data.file_path: planning-unit table (Parquet or CSV)
          - data.protected_classes / data.protected_th: baseline definition
          - species.include / species.star_path: STAR species layer
    bbox : dict, optional
        Spatial subset in native coordinates (see read_planning_units).
    Returns
    -------
    pandas.DataFrame
        Planning units with the added columns:
          - "area" (int): 1 (km2) for every row.
          - "protected" (int): 1 when any protected class value > protected_th
            (the baseline: PAs, or PAs + CFCLs).
          - "star_total" (float): only when species are included; cells
            missing from the STAR file get 0.
        NaN values in the scored layers are set to 0.
    """

    df = read_planning_units(config["data"]["file_path"],
                             required_columns(config), bbox=bbox)
    df["area"] = 1
    protected_classes = config["data"]["protected_classes"]
    protected_th = config["data"]["protected_th"]
    df["protected"] = ((df[protected_classes] > protected_th).any(axis=1)).astype(int)

    if species_included(config):
        star = load_star_total(config["species"]["star_path"],
                               config["species"].get("memory_budget_mb", 512))
        df = df.merge(star, on="grid_id", how="left")
        n_missing = int(df[SPECIES_COL].isna().sum())
        if n_missing:
            print(f"{n_missing:,} planning units have no STAR value; set to 0")
    df = pre_process_data(df, config)
    return df


def pre_process_data(df, config):
    """
    Fill no-data (NaN or non-numeric) cells of all scored layers.
    The rule per layer comes from `fill_na` (configs/common/themes.yml):
    `fill_na.columns` maps a layer to a number, "median" or "mean" (computed
    over the layer's non-missing cells in df, i.e. the full grid in normal
    runs); every other scored layer gets `fill_na.default` (0.0). See
    the README ("Weighting") for the rationale.
    """

    fill = config.get("fill_na", {})
    default = fill.get("default", 0.0)
    rules = fill.get("columns") or {}
    cols = [c for c in compute_weights(config)["column"] if c in df.columns]
    df[cols] = df[cols].apply(pd.to_numeric, errors="coerce")
    for c in cols:
        rule = rules.get(c, default)
        if rule == "median":
            value = df[c].median()
        elif rule == "mean":
            value = df[c].mean()
        else:
            value = float(rule)
        n_missing = int(df[c].isna().sum())
        if n_missing and c in rules:
            print(f"{c}: {n_missing:,} missing cells ({n_missing / len(df):.1%}) filled with {rule} = {value:.4g}")
        df[c] = df[c].fillna(value)
    return df


# ---------------------------------------------------------------------------
# Scoring and candidate identification
# ---------------------------------------------------------------------------

def layer_scaling(config):
    """Per-layer scaling method: `scaling.columns` overrides `scaling.default` ("minmax")."""
    scaling = config.get("scaling", {})
    default = scaling.get("default", "minmax")
    columns = scaling.get("columns") or {}
    return {c: columns.get(c, default) for c in compute_weights(config, True)["column"]}


def rescale(values, method="minmax"):
    """
    Rescale a layer to 0-1.
      - "minmax": (v - min) / (max - min); a constant layer gives 0.
      - "rank": percentile rank among the cells with a value > 0, so the
        smallest positive value gets 1/n_pos and the largest 1; tied values
        share their average rank; cells with value <= 0 (e.g. no species)
        get 0. Order is kept, magnitudes are not.
    """

    s = np.asarray(values, dtype=np.float64)
    if method == "minmax":
        mn, mx = np.nanmin(s), np.nanmax(s)
        if not np.isfinite(mn) or not np.isfinite(mx) or mx == mn:
            return np.zeros_like(s)  # constant or invalid -> contributes 0
        return (s - mn) / (mx - mn)
    if method == "rank":
        out = np.zeros_like(s)
        pos = s > 0
        if pos.any():
            out[pos] = pd.Series(s[pos]).rank(method="average", pct=True).to_numpy()
        return out
    raise ValueError(f"Unknown scaling method: {method!r} (use 'minmax' or 'rank')")


def score_components(df, weights, scaling=None):
    """
    Rescale each scored layer to 0-1 (min-max over df by default, or the
    method given in `scaling`, see rescale) and return the weighted maximize
    and minimize parts of the composite score.
    Parameters
    ----------
    df : pandas.DataFrame
        Planning units containing every column in weights["column"].
    weights : pandas.DataFrame
        Output of compute_weights.
    scaling : dict, optional
        Layer -> scaling method (output of layer_scaling). Default: min-max for all.
    Returns
    -------
    (pandas.Series, pandas.Series, pandas.DataFrame)
        max_part (>= 0), min_part (>= 0, to be subtracted), and the weighted
        contribution of every theme/direction (columns "<theme>:max"/"<theme>:min").
    """

    missing = [c for c in weights["column"] if c not in df.columns]
    assert not missing, f"Missing columns in df: {missing}"

    scaling = scaling or {}
    max_part = np.zeros(len(df))
    min_part = np.zeros(len(df))
    themes = {}
    for row in weights.itertuples(index=False):
        contrib = row.weight * rescale(df[row.column].to_numpy(dtype=np.float64),
                                       scaling.get(row.column, "minmax"))
        key = f"{row.theme}:{'max' if row.direction > 0 else 'min'}"
        themes[key] = themes.get(key, 0.0) + contrib
        if row.direction > 0:
            max_part += contrib
        else:
            min_part += contrib

    idx = df.index
    return pd.Series(max_part, index=idx), pd.Series(min_part, index=idx), pd.DataFrame(themes, index=idx)


def multi_c_obj(df, config):
    """
    Compute the thematically weighted composite score for every planning unit:
    score = sum(weight * rescaled maximize layers) - sum(weight * rescaled minimize layers),
    with each layer rescaled as set in `scaling` (layer_scaling).
    Returns
    -------
    pandas.Series
    """

    max_part, min_part, _ = score_components(df, compute_weights(config, SPECIES_COL in df.columns),
                                             layer_scaling(config))
    return max_part - min_part


def score_diagnostics(max_part, min_part, eligible=None):
    """
    Summarize the balance between the maximize and minimize sides of the score.
    Returns a dict with quantiles of both parts and of the score, and the share
    of cells (overall and among cells passing the exclusion rules) where the
    minimize side outweighs the maximize side (score <= 0, i.e. not a candidate).
    """

    score = max_part - min_part
    q = [0.05, 0.25, 0.5, 0.75, 0.95]
    out = {"n_cells": int(len(score)),
           "score_pos_cells": int((score > 0).sum()),
           "score_pos_pct": float((score > 0).mean() * 100)}
    for name, s in (("max", max_part), ("min", min_part), ("score", score)):
        out[f"{name}_mean"] = float(s.mean())
        for qq, v in zip(q, s.quantile(q)):
            out[f"{name}_p{int(qq * 100):02d}"] = float(v)
    out["min_gt_max_pct"] = float((min_part >= max_part).mean() * 100)
    if eligible is not None:
        e = eligible.to_numpy(dtype=bool)
        out["eligible_cells"] = int(e.sum())
        out["eligible_score_pos_cells"] = int((score[e] > 0).sum())
        out["eligible_min_gt_max_pct"] = float((min_part[e] >= max_part[e]).mean() * 100) if e.any() else np.nan
    return out


def eligible_mask(df, config):
    """True for units that pass every exclusion rule (constraint column <= threshold)."""
    ineligible = pd.Series(False, index=df.index)
    for col, threshold in constraint_cols(config):
        ineligible |= df[col].fillna(0) > threshold
    return ~ineligible


def iden_units(df, config):
    """
    Identify candidate planning units based on eligibility criteria.
    This function filters planning units from a DataFrame based on constraint thresholds
    and objective scores, marking eligible units as candidates for optimization.
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing planning unit data with columns for constraints,
        objective scores, and protection status.
    config : dict
        Configuration dictionary. Exclusion rules are the (column_name, threshold)
        pairs of optimization.constraint_cols plus optimization.extra_constraint_cols.
    Returns
    -------
    pd.DataFrame
        Modified DataFrame with an added 'candidate' column where:
        - 1 indicates units meeting eligibility criteria (either objective_score > 0
          and passing all constraints, or protected == 1)
        - 0 indicates ineligible units
    Notes
    -----
    - Units are considered ineligible if any constraint column value exceeds its threshold
    - Protected units (protected == 1) are always included as candidates
    - NaN values in constraint columns are treated as 0
    - Prints summary statistics of identified candidates
    """

    eligible = eligible_mask(df, config)
    candid_idx = df.index[(eligible & (df["objective_score"] > 0)) | (df["protected"] == 1)]

    df["candidate"] = 0
    df.loc[candid_idx, "candidate"] = 1

    print(f"N. of Identified Candidates: {len(candid_idx):,} out of {len(df):,} total planning units")

    return df


# ---------------------------------------------------------------------------
# Optimization
# ---------------------------------------------------------------------------

def setup_model(df, protection_frac=0.30, connectivity_weight=0.0, edges=None):
    """
    Set up the integer linear programming model (CBC path) for the protection planning problem.
    The model maximizes the total objective score of selected units, plus an
    optional connectivity reward for adjacent selected pairs, subject to the
    area cap. The same problem is solved exactly and much faster by
    scripts.connectivity.solve_mincut; this CBC formulation is kept for
    verification on subsets and for scenarios needing extra constraints.
    Args:
        df (pandas.DataFrame): Planning units with columns:
            - "candidate" (int): 1 = candidate for protection, 0 = not candidate
            - "objective_score" (float): Score to maximize for each unit
            - "protected" (int): 1 = baseline unit (fixed to selected)
            - "area" (numeric): Area of each unit (coerced to float)
        protection_frac (float, optional): Fraction of total area that can be protected. Default is 0.30 (30%).
        connectivity_weight (float, optional): Reward w per pair of adjacent selected units. Default 0 (off).
        edges (numpy.ndarray, optional): Rook adjacency from scripts.connectivity.rook_edges,
            positional indices into df. Required when connectivity_weight > 0.
    Returns:
        tuple: (model, x) with the PuLP problem and a dict of binary decision variables
        keyed by the df index of each candidate.
    Notes:
        - Baseline units are fixed to be selected (x[i] = 1).
        - For each pair of adjacent free candidates a continuous y_ij with
          y_ij <= x_i and y_ij <= x_j earns w (the objective is maximized, so
          y_ij = x_i * x_j at the optimum). A pair of a free candidate and a
          baseline unit is a fixed bonus w on the free unit; pairs of two
          baseline units are a constant w each (kept so the objective matches
          scripts.connectivity.objective_value).
    """

    candid_idx = df[df["candidate"]==1].index

    # Assign Binary Decision (protected (1) or not (0))
    x = {i: LpVariable(f"x_{i}", cat=LpBinary) for i in candid_idx}

    # Problem
    model = LpProblem("30x30_Protection_Plan", LpMaximize)

    score = df["objective_score"]
    terms = [(x[i], float(score.at[i])) for i in candid_idx]
    constant = 0.0

    if connectivity_weight:
        assert edges is not None, "edges are required when connectivity_weight > 0"
        idx = df.index.to_numpy()
        candidate = df["candidate"].to_numpy() == 1
        protected = (df["protected"].to_numpy() == 1) & candidate
        ce = edges[candidate[edges[:, 0]] & candidate[edges[:, 1]]]
        pa, pb = protected[ce[:, 0]], protected[ce[:, 1]]
        constant += connectivity_weight * float((pa & pb).sum())
        for k in np.r_[ce[pa & ~pb, 1], ce[~pa & pb, 0]]:
            terms.append((x[idx[k]], connectivity_weight))
        for n, (i, j) in enumerate(ce[~pa & ~pb]):
            y = LpVariable(f"y_{n}", 0, 1)
            terms.append((y, connectivity_weight))
            model += y - x[idx[i]] <= 0
            model += y - x[idx[j]] <= 0

    model += LpAffineExpression(terms, constant=constant), "CompositeScore_WithConnectivity"

    #  Fix already-protected areas
    for i in candid_idx:
        if bool(df.at[i, "protected"]):
            x[i].lowBound = 1
            x[i].upBound  = 1

    #  Area budget constraint
    assert "area" in df.columns
    df["area"] = pd.to_numeric(df["area"], errors="coerce").fillna(0.0)
    total_area = float(df["area"].sum())
    model += LpAffineExpression([(x[i], float(df.at[i, "area"])) for i in candid_idx]) \
        <= protection_frac * total_area, "AreaBudget"
    return model, x


def solve_model(model, x, df, threads=4, time_limit=2000, gap_rel=0.001):
    """
    Solve an optimization model and update a dataframe with selected candidates.
    Args:
        model: A PuLP optimization model to solve.
        x: A dictionary or list of PuLP decision variables indexed by dataframe indices.
        df: A pandas DataFrame containing candidate data with a "candidate" column.
        threads (int, optional): Number of threads for the CBC solver. Defaults to 4.
        time_limit (int, optional): CBC time limit in seconds. Defaults to 2000.
        gap_rel (float, optional): CBC relative gap tolerance. Defaults to 0.001 (0.1%).
    Returns:
        pd.DataFrame: The input dataframe with a new "selected" column containing the
                      optimized variable values (0.0 if variable value is None).
    Raises:
        None: Check LpStatus[model.status] in returned output for solver status.
    Notes:
        - Only updates "selected" values for rows where "candidate" column equals 1.
        - If a decision variable has no value, defaults to 0.0.
    """

    solver = PULP_CBC_CMD(
    timeLimit = time_limit,
    options = ["preprocess off", "log 2"],
    threads = threads,
    gapRel = gap_rel,
    msg = True
    )

    model.solve(solver)

    print(f"Status: {LpStatus[model.status]}")

    df["selected"] = 0.0
    cand_idx = df[df["candidate"]==1].index
    for i in cand_idx:
        df.at[i, "selected"] = x[i].varValue or 0.0

    return df


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def oil_overlap(df):
    """
    Newly selected area (selected and not baseline) and the part of it that
    falls inside oil blocks. The overlap is area-weighted by `oil_frac`
    (fraction of each 1 km2 cell covered by oil blocks).
    Returns
    -------
    dict
        new_area_km2, new_in_oil_km2, new_in_oil_pct, new_cells_touching_oil.
    """

    new = (df["selected"] > 0.5) & (df["protected"] == 0)
    area = df["area"].astype(float)
    oil = df[OIL_COL].fillna(0).clip(0, 1)
    new_area = float(area[new].sum())
    in_oil = float((area * oil)[new].sum())
    return {
        "new_area_km2": new_area,
        "new_in_oil_km2": in_oil,
        "new_in_oil_pct": in_oil / new_area * 100 if new_area else np.nan,
        "new_cells_touching_oil": int((new & (oil > 0)).sum()),
    }


def star_capture(df, config):
    """
    Share of the national STAR total (and of threatened-only STAR) captured
    by the newly selected units and by baseline + new units, from the STAR
    summary Parquet (`species.summary_path`: grid_id, star_total,
    star_threatened). Computed for every run, with or without species in the
    score. Returns NaNs if the summary file is missing.
    """

    keys = ["star_share_new_pct", "star_share_selected_pct",
            "star_threatened_share_new_pct", "star_threatened_share_selected_pct"]
    path = config.get("species", {}).get("summary_path")
    if not path or not _resolve(path).exists():
        print(f"STAR summary not found ({path}); STAR capture not reported")
        return {k: np.nan for k in keys}
    star = pd.read_parquet(_resolve(path), columns=["grid_id", "star_total", "star_threatened"])
    d = df[["grid_id", "selected", "protected"]].merge(star, on="grid_id", how="left").fillna(0.0)
    sel = d["selected"] > 0.5
    new = sel & (d["protected"] == 0)
    out = {}
    for col, prefix in (("star_total", "star"), ("star_threatened", "star_threatened")):
        total = float(d[col].sum())
        out[f"{prefix}_share_new_pct"] = float(d.loc[new, col].sum()) / total * 100 if total else np.nan
        out[f"{prefix}_share_selected_pct"] = float(d.loc[sel, col].sum()) / total * 100 if total else np.nan
    return out


def upsert_summary(row, path="output/scenario_summary.csv"):
    """Insert or replace (by `name`) one scenario's row in the summary CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    new = pd.DataFrame([row])
    if path.exists():
        old = pd.read_csv(path)
        old = old[old["name"] != row["name"]]
        new = pd.concat([old, new], ignore_index=True)
    new.to_csv(path, index=False)
    return new


def grid_array(df, var):
    """
    Pivot a column to a 2D array (rows = northing, ascending; cols = easting,
    ascending) covering the bounding box of df's planning units.
    Returns (array, x_coords, y_coords).
    """

    arr = df.pivot(index="y_native", columns="x_native", values=var)
    return arr.to_numpy(), arr.columns.to_numpy(), arr.index.to_numpy()


def visualize(df, var, colorbar=True, file_name=None, title=None, show=True, output_dir="output"):
    """
    Create a spatial map visualization of a column from a DataFrame.
    This function generates a 2D map showing the spatial distribution of values
    across a grid defined by x_native and y_native coordinates. The map uses a viridis
    colormap and can optionally display a colorbar and save the figure to a file.
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the data to be mapped. Must include columns 'x_native',
        'y_native', and the column specified by var.
    var : str
        Name of the column in df to visualize on the map.
    colorbar : bool, optional
        Whether to display a colorbar. Default is True.
    file_name : str, optional
        If provided, saves the figure to this file path with 300 dpi resolution.
        Default is None (figure is not saved).
    title : str, optional
        Title for the plot. If not provided, defaults to "Spatial extent of {var}".
        Default is None.
    show : bool, optional
        If True (default), displays the plot with plt.show(). If False, the
        figure is closed after saving instead of displayed — useful when
        batch-running many scenarios to avoid accumulating open figures.
    output_dir : str, optional
        Directory the figure is saved to. Default is "output".
    Returns
    -------
    None
    Notes
    -----
    - The y-axis is flipped to align with typical coordinate system conventions.
    - The figure size is fixed at 8x6 inches.
    - Aspect ratio is set to 'equal' for accurate spatial representation.
    """

    arr, lons, lats = grid_array(df, var)

    fig = plt.figure(figsize=(8, 6))
    fig.set_constrained_layout(False)
    try:
        fig.set_tight_layout(False)
    except Exception:
        pass

    # Fixed title area (inside the figure)
    title_ax = fig.add_axes([0.2, 0.91, 0.65, 0.06])
    title_ax.axis("off")
    title_text = title if title else "Spatial extent of " + var
    title_ax.text(0.5, 0.5, title_text, ha="center", va="center", wrap=True)

    if colorbar:
        ax = fig.add_axes([0.10, 0.10, 0.75, 0.75])
    else:
        ax = fig.add_axes([0.10, 0.10, 0.80, 0.75])  # fixed plot area

    im = ax.imshow(
        np.flipud(arr),
        extent=[lons.min(), lons.max(), lats.min(), lats.max()],
        cmap="viridis",
        aspect="auto",
    )
    ax.set_xlabel("X")
    ax.set_ylabel("Y")


    if colorbar:
        cax = fig.add_axes([0.88, 0.10, 0.03, 0.75])  # fixed colorbar area
        fig.colorbar(im, cax=cax, label=var)

    ax.set_aspect("equal")

    if file_name:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, file_name), dpi=300)

    if show:
        plt.show()
    else:
        plt.close(fig)


def export_to_geotiff(
    df,
    var,
    file_name,
    projection_info_path="data/projection_info.csv",
    output_dir="output"
):
    """
    Export a DataFrame column as a GeoTIFF raster using a custom CRS.
    The CRS and pixel size are read from a GEE-exported CSV (projection_info.csv).
    The raster origin is derived from the planning units themselves
    (x_native/y_native are cell centres), so the raster is correctly
    georeferenced for the full grid and for spatial subsets alike.
    Args:
        df: DataFrame with 'x_native' (easting), 'y_native' (northing), and var.
        var: Name of the column to export.
        file_name: Output GeoTIFF file name.
        projection_info_path: Path to CSV with columns: scale, transform, wkt.
        output_dir: Directory the GeoTIFF is written to. Defaults to "output".
    """

    arr, xs, ys = grid_array(df, var)

    # Load projection info (WKT + affine transform) exported from GEE
    proj_df = pd.read_csv(_resolve(projection_info_path))
    if proj_df.empty:
        raise ValueError(f"Projection info file is empty: {projection_info_path}")
    wkt = str(proj_df.loc[0, "wkt"])
    transform_str = str(proj_df.loc[0, "transform"])

    # Pixel size and rotation from the GEE transform string (PARAM_MT); the
    # origin is the outer corner of the top-left planning unit.
    params = dict(re.findall(r'PARAMETER\["([^"]+)",\s*([-0-9\.eE]+)\]', transform_str))
    a = float(params.get("elt_0_0", 1.0))
    b = float(params.get("elt_0_1", 0.0))
    d = float(params.get("elt_1_0", 0.0))
    e = float(params.get("elt_1_1", -1.0))
    c = float(xs.min()) - a / 2
    f = float(ys.max()) - e / 2
    transform = Affine(a, b, c, d, e, f)

    # Write GeoTIFF
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, file_name)
    with rasterio.open(
        output_path,
        'w',
        driver='GTiff',
        height=arr.shape[0],
        width=arr.shape[1],
        count=1,
        dtype=arr.dtype,
        crs=CRS.from_wkt(wkt),
        transform=transform,
        nodata=np.nan if np.issubdtype(arr.dtype, np.floating) else None
    ) as dst:
        dst.write(np.flipud(arr), 1)
    print(f"GeoTIFF written to {file_name}")
