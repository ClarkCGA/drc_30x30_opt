import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pulp import LpProblem, LpVariable, LpMaximize, lpSum, LpBinary, PULP_CBC_CMD, LpStatus
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
import rasterio
from rasterio.transform import Affine
from rasterio.crs import CRS
import re

def load_data(config, preprocess = True):
    """
    Load a CSV into a pandas DataFrame, add metadata columns, and optionally preprocess it.
    Parameters
    ----------
    config : dict
        Configuration dictionary expected to contain:
          - data.file_path (str): Path to the CSV file to load.
          - data.protected_classes (str or list[str]): Column name(s) used to determine protected status.
          - data.protected_th (number): Threshold used to mark a row as protected when any protected class value exceeds it.
    preprocess : bool, optional
        If True (default), the loaded DataFrame is passed to pre_process_data(df, config) before being returned.
    Returns
    -------
    pandas.DataFrame
        DataFrame loaded from the specified CSV with two additional columns:
          - "area" (int): constant value 1 for every row.
          - "protected" (int): binary indicator (0 or 1) set to 1 when any protected class value > protected_th.
    """

    file_path = config["data"]["file_path"]
    df = pd.read_csv(file_path)
    df["area"] = 1
    protected_classes = config["data"]["protected_classes"]
    protected_th = config["data"]["protected_th"]
    df["protected"] = ((df[protected_classes] > protected_th).any(axis=1)).astype(int)
    if preprocess: 
        df = pre_process_data(df, config)
    
    return df


def pre_process_data(df, config):
    """
    Pre-process data by converting specified columns to numeric values and handling missing data.
    This function takes a DataFrame and converts columns defined in the config's 'data.pre_process' 
    key to numeric values. Any cells that cannot be converted to numeric values (including NaN) 
    are filled with 0.0.
    Args:
        df (pd.DataFrame): The input DataFrame to be pre-processed.
        config (dict): A configuration dictionary containing a 'data' key with a 'pre_process' 
                      key that specifies which columns to process. Expected structure:
                      {'data': {'pre_process': [list of column names]}}
    Returns:
        pd.DataFrame: The pre-processed DataFrame with specified columns converted to numeric 
                     values and missing data filled with 0.0.
    """

    # Data Pre-processing:
    # Setting cells with no data value for pre_process columns (defined in config) to zero
    cols = config["data"]["pre_process"]
    df[cols] = df[cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)

    return df


def visualize(df, var, colorbar=True, file_name=None, title=None):
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
    Returns
    -------
    None
        Displays the plot using plt.show().
    Notes
    -----
    - The y-axis is flipped to align with typical coordinate system conventions.
    - The figure size is fixed at 8x6 inches.
    - Aspect ratio is set to 'equal' for accurate spatial representation.
    """
    
    # unique sorted coords
    lats = np.sort(df["y_native"].unique())
    lons = np.sort(df["x_native"].unique())
    
    # create a 2D array for the values
    arr = df.pivot(index="y_native", columns="x_native", values=var).to_numpy()
    
    fig, ax = plt.subplots(figsize=(8,6))
    im = ax.imshow(
        np.flipud(arr),              
        extent=[lons.min(), lons.max(), lats.min(), lats.max()],
        cmap="viridis",
        aspect="auto",
    )
    plt.xlabel("X")
    plt.ylabel("Y")
    if title:
        plt.title(title)
    else:
        plt.title("Spatial extent of " + var)
    if colorbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cb = plt.colorbar(im, cax=cax, label=var)
    ax.set_aspect('equal')
    if file_name:
        os.makedirs("output", exist_ok=True)
        plt.savefig(os.path.join("output", file_name), bbox_inches="tight", dpi=300)
    plt.show()


def export_to_geotiff(
    df,
    var,
    file_name,
    projection_info_path="data/projection_info.csv"
):
    """
    Export a DataFrame column as a GeoTIFF raster using a custom CRS/transform.
    The CRS and affine transform are read from a GEE-exported CSV (projection_info.csv).
    Args:
        df: DataFrame with 'x_native' (easting), 'y_native' (northing), and var.
        var: Name of the column to export.
        file_name: Output GeoTIFF file name.
        projection_info_path: Path to CSV with columns: scale, transform, wkt.
    """

    # Create 2D array
    arr = df.pivot(index="y_native", columns="x_native", values=var).to_numpy()

    # Load projection info (WKT + affine transform) exported from GEE
    proj_df = pd.read_csv(projection_info_path)
    if proj_df.empty:
        raise ValueError(f"Projection info file is empty: {projection_info_path}")
    wkt = str(proj_df.loc[0, "wkt"])
    transform_str = str(proj_df.loc[0, "transform"])

    # Parse GEE transform string (PARAM_MT) into affine parameters
    params = dict(re.findall(r'PARAMETER\["([^"]+)",\s*([-0-9\.eE]+)\]', transform_str))
    a = float(params.get("elt_0_0", 1.0))
    b = float(params.get("elt_0_1", 0.0))
    c = float(params.get("elt_0_2", 0.0))
    d = float(params.get("elt_1_0", 0.0))
    e = float(params.get("elt_1_1", -1.0))
    f = float(params.get("elt_1_2", 0.0))
    transform = Affine(a, b, c, d, e, f)

    # Write GeoTIFF
    os.makedirs("output", exist_ok=True)
    with rasterio.open(
        "output/" + file_name,
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


def multi_c_obj(df, config):
    """
    Compute a weighted multi-objective composite score for a DataFrame.
    Normalizes specified columns to the range [0, 1], applies direction weights
    (maximize or minimize), and combines them into a single composite score.
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing the columns to optimize.
    config : dict
        Configuration dictionary with structure:
        {
            "optimization": {
                "maximize_cols": list of str,
                    Column names to maximize,
                "minimize_cols": list of str,
                    Column names to minimize,
                "weights": dict, optional
                    Mapping of column names to weights (default: 1.0 for all columns)
            }
        }
    Returns
    -------
    pd.Series
        Composite score for each row in df, where higher values indicate
        better solutions according to the optimization objectives.
    Raises
    ------
    AssertionError
        If any column specified in maximize_cols or minimize_cols is not
        present in df.
    """

    maximize_cols = config["optimization"]["maximize_cols"]
    minimize_cols = config["optimization"]["minimize_cols"]
    weights = config["optimization"].get("weights", {})

    all_cols = maximize_cols + minimize_cols

    # Ensure all columns exist
    missing = [c for c in all_cols if c not in df.columns]
    assert not missing, f"Missing columns in df: {missing}"

    # Build normalized columns (0..1), robust to constants/NaNs
    norm = {}
    for c in all_cols:
        s = df[c]
        mn, mx = float(s.min(skipna=True)), float(s.max(skipna=True))
        if not np.isfinite(mn) or not np.isfinite(mx) or mx == mn:
            # constant or invalid → contributes 0
            norm[c] = pd.Series(0.0, index=df.index)
        else:
            norm[c] = (s - mn) / (mx - mn)

    # Direction: +1 for maximize, -1 for minimize
    direction = {c: 1.0 for c in maximize_cols}
    direction.update({c: -1.0 for c in minimize_cols})

    # Per-cell composite score
    score = pd.Series(0.0, index=df.index)
    for c in all_cols:
        w = float(weights.get(c, 1.0))
        score = score + direction[c] * w * norm[c]

    return score


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
        Configuration dictionary containing optimization parameters. Must include
        'optimization' > 'constraint_cols' key with a list of tuples specifying
        (column_name, threshold) pairs for constraint evaluation.
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

    constraint_cols = config["optimization"]["constraint_cols"]
    ineligible = pd.Series(False, index=df.index)
    for col, threshold in constraint_cols:
        ineligible |= df[col].fillna(0) > threshold
    eligible = ~ineligible
    candid_idx = df.index[(eligible & (df["objective_score"] > 0)) | (df["protected"] == 1)]
    
    df["candidate"] = 0
    df.loc[candid_idx, "candidate"] = 1

    print(f"N. of Identified Candidates: {len(candid_idx):,} out of {len(df):,} total planning units")

    return df


def setup_model(df, protection_frac = 0.30, connectivity = True):
    """
    Set up a linear programming optimization model for a conservation protection planning problem.
    This function creates an integer linear programming model that maximizes the total objective score
    of protected areas while respecting area budget constraints. It optionally includes a connectivity
    bonus to encourage protection of neighboring areas.
    Args:
        df (pandas.DataFrame): A DataFrame containing candidate areas with the following required columns:
            - "candidate" (int): Binary indicator (1 = candidate for protection, 0 = not candidate)
            - "objective_score" (float): Score to maximize for each candidate area
            - "protected" (bool/int): Binary indicator of already-protected areas (1 = protected, 0 = not)
            - "neighbors" (list): List of neighboring area indices for connectivity calculation
            - "area" (numeric): Area size of each candidate (will be coerced to float)
        protection_frac (float, optional): Fraction of total area that can be protected. Default is 0.30 (30%).
        connectivity (bool, optional): If True, adds a connectivity bonus to the objective function
            that incentivizes protection of adjacent areas. Default is True.
    Returns:
        tuple: A tuple containing:
            - model (LpProblem): The configured optimization model (maximization)
            - x (dict): Dictionary mapping candidate indices to binary decision variables
    Notes:
        - Already-protected areas are fixed to be protected (x[i] = 1)
        - Connectivity weight is set to 0.5 when connectivity is enabled
        - Double-counting of edges is prevented by only considering pairs where i < j
        - Area column values are coerced to float and missing values filled with 0.0
    """

    candid_idx = df[df["candidate"]==1].index
    
    # Assign Binary Decision (protected (1) or not (0))
    x = {i: LpVariable(f"x_{i}", cat=LpBinary) for i in candid_idx}
    
    # Problem
    model = LpProblem("30x30_Protection_Plan", LpMaximize)
    
    objective_expr = lpSum(df.loc[i, "objective_score"] * x[i] for i in candid_idx)
    
    if connectivity:
        connectivity_weight = 0.5
        connectivity_bonus = lpSum(
            connectivity_weight * (x[i] + x[j])
            for i in candid_idx
            for j in (df.at[i, "neighbors"])
            if j in candid_idx and i < j   # prevents double counting
        )
        objective_expr += connectivity_bonus
    
    model += objective_expr, "CompositeScore_WithConnectivity"
    
    #  Fix already-protected areas 
    for i in candid_idx:
        if bool(df.loc[i, "protected"]):  
            x[i].lowBound = 1
            x[i].upBound  = 1
    
    #  Area budget constraint 
    assert "area" in df.columns
    df["area"] = pd.to_numeric(df["area"], errors="coerce").fillna(0.0)
    total_area = float(df["area"].sum())
    model += lpSum(df.loc[i, "area"] * x[i] for i in candid_idx) <= protection_frac * total_area, "AreaBudget"
    return model, x

                        
def solve_model(model, x, df, threads=4):
    """
    Solve an optimization model and update a dataframe with selected candidates.
    Args:
        model: A PuLP optimization model to solve.
        x: A dictionary or list of PuLP decision variables indexed by dataframe indices.
        df: A pandas DataFrame containing candidate data with a "candidate" column.
        threads (int, optional): Number of threads for the CBC solver. Defaults to 4.
    Returns:
        pd.DataFrame: The input dataframe with a new "selected" column containing the 
                      optimized variable values (0.0 if variable value is None).
    Raises:
        None: Check LpStatus[model.status] in returned output for solver status.
    Notes:
        - Uses PuLP's CBC solver with a 2000 second time limit and 0.1% relative gap tolerance.
        - Only updates "selected" values for rows where "candidate" column equals 1.
        - If a decision variable has no value, defaults to 0.0.
    """

    solver = PULP_CBC_CMD(
    timeLimit = 2000,
    options = ["preprocess off", "log 2"],
    threads = threads, 
    gapRel = 0.001,
    msg = True
    )
    
    model.solve(solver)

    print(f"Status: {LpStatus[model.status]}")
    
    df["selected"] = 0.0
    cand_idx = df[df["candidate"]==1].index
    for i in cand_idx:
        df.at[i, "selected"] = x[i].varValue or 0.0

    return df