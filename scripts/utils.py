import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pulp import LpProblem, LpVariable, LpMaximize, lpSum, LpBinary, PULP_CBC_CMD, LpStatus
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

def load_data(config, preprocess = True):

    file_path = config["data"]["file_path"]
    df = pd.read_csv(file_path)
    df["area"] = 1
    df["protected"] = (df["protected_frac"] > config["data"]["protected_th"]).astype(int)
    if preprocess: 
        df = pre_process_data(df)
    
    return df


def pre_process_data(df):
    # Data Pre-processing:
    # Setting cells with no data value for mean_GSOC to zero
    # Setting cells with no data value for mean_GCP_Dinerstein or mean_GCP_Jung or mean_FLII or mean_agb to zero
    cols = ["mean_GSOC", "mean_FLII", "mean_agb", "mean_GCP_Dinerstein", "mean_GCP_Jung"]
    df[cols] = df[cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)

    return df


def map_col(df, plot_col, colorbar=True, file_name=None, title=None):
    # unique sorted coords
    lats = np.sort(df["y_native"].unique())
    lons = np.sort(df["x_native"].unique())
    
    # create a 2D array for the values
    arr = df.pivot(index="y_native", columns="x_native", values=plot_col).to_numpy()
    
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
        plt.title("Spatial extent of " + plot_col)
    if colorbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cb = plt.colorbar(im, cax=cax, label=plot_col)
    ax.set_aspect('equal')
    if file_name:
        plt.savefig(file_name, bbox_inches="tight", dpi=300)
    plt.show()


def multi_c_obj(df, config):

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

    constraint_cols = config["optimization"]["constraint_cols"]
    ineligible = (df[constraint_cols].fillna(0) > 0).any(axis=1)
    eligible = ~ineligible
    cand_idx = df.index[(eligible & (df["objective_score"] > 0)) | (df["protected"] == 1)]
    
    print(f"N. of Identified Candidates: {len(cand_idx):,} out of {len(df):,} total planning units")

    df["candidate"] = 0
    df.loc[cand_idx, "candidate"] = 1

    return df


def setup_model(df, protection_frac = 0.30, connectivity = True):

    cand_idx = df[df["candidate"]==1].index
    
    # Assign Binary Decision (protected (1) or not (0))
    x = {i: LpVariable(f"x_{i}", cat=LpBinary) for i in cand_idx}
    
    # Problem
    model = LpProblem("30x30_Protection_Plan", LpMaximize)
    
    objective_expr = lpSum(df.loc[i, "objective_score"] * x[i] for i in cand_idx)
    
    if connectivity:
        connectivity_weight = 0.5
        connectivity_bonus = lpSum(
            connectivity_weight * (x[i] + x[j])
            for i in cand_idx
            for j in (df.at[i, "neighbors"])
            if j in cand_idx and i < j   # prevents double counting
        )
        objective_expr += connectivity_bonus
    
    model += objective_expr, "CompositeScore_WithConnectivity"
    
    # --- Fix already-protected areas ---
    for i in cand_idx:
        if bool(df.loc[i, "protected"]):  
            x[i].lowBound = 1
            x[i].upBound  = 1
    
    # --- Area budget constraint ---
    assert "area" in df.columns
    df["area"] = pd.to_numeric(df["area"], errors="coerce").fillna(0.0)
    total_area = float(df["area"].sum())
    model += lpSum(df.loc[i, "area"] * x[i] for i in cand_idx) <= protection_frac * total_area, "AreaBudget"

    return model, x

                        
def solve_model(model, x, df):
    solver = PULP_CBC_CMD(
    timeLimit=2000,    # seconds
    # gapRel=0.005,     # 0.5% MIP gap
    # msg=1,
    options=["preprocess off", "log 2"],
    threads=5
    )
    
    model.solve(solver)

    print(f"Status: {LpStatus[model.status]}")
    
    df["selected"] = 0.0
    cand_idx = df[df["candidate"]==1].index
    for i in cand_idx:
        df.at[i, "selected"] = x[i].varValue or 0.0

    return df
