"""
Spatial connectivity for the 30x30 optimization.

The connectivity term rewards pairs of adjacent planning units (sharing a
1 km edge, "rook" adjacency) that are both selected:

    maximize  sum_i s_i x_i + w * sum_{(i,j) adjacent} x_i x_j
    s.t.      sum_i area_i x_i <= protection_frac * total area
              x_i = 1 for baseline units, x_i = 0 for non-candidates

Edges between a candidate and a baseline unit (fixed at 1) reduce to a fixed
bonus w on the candidate; edges between two baseline units are constant.

Two solvers are provided:
  - solve_mincut: exact parametric min-cut (scipy). The area cap is handled
    by a price lambda per km2 (Lagrangian relaxation); for a given lambda the
    problem is a minimum s-t cut, solved exactly. lambda is bisected until the
    selection just fits under the cap, a greedy fill then uses the remaining
    budget, and the Lagrangian dual gives an upper bound on the optimum, so the
    remaining gap is known. Handles the full grid in ~1-2 minutes.
  - the CBC model in utils.setup_model (y_ij <= x_i, y_ij <= x_j), kept for
    verification on subsets and for scenarios needing extra constraints.

The min-cut method only supports this structure: one area cap and
non-negative pair rewards. Additional budget-type constraints (per-province
budgets, per-theme targets) or patch-size constraints need the CBC path.
"""

import time

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import breadth_first_order, connected_components, maximum_flow


def rook_edges(df, cell_size=1000.0):
    """
    Undirected rook (shared-edge) adjacency between planning units, built from
    the cell-centre coordinates x_native/y_native.
    Returns an (n_edges, 2) array of positional row indices (i < j not implied).
    """

    x = df["x_native"].to_numpy()
    y = df["y_native"].to_numpy()
    ix = np.round((x - x.min()) / cell_size).astype(np.int64)
    iy = np.round((y - y.min()) / cell_size).astype(np.int64)
    width = int(iy.max()) + 2
    key = ix * width + iy
    order = np.argsort(key)
    sorted_key = key[order]
    edges = []
    for dx, dy in ((1, 0), (0, 1)):
        target = (ix + dx) * width + (iy + dy)
        pos = np.searchsorted(sorted_key, target)
        pos = np.minimum(pos, len(sorted_key) - 1)
        found = sorted_key[pos] == target
        edges.append(np.c_[np.flatnonzero(found), order[pos[found]]])
    return np.vstack(edges)


def objective_value(score, selected, candidate, edges, weight):
    """
    Objective of a selection: sum of scores of selected candidates plus
    `weight` per adjacent pair of selected candidates (including the constant
    baseline terms). Same definition for both solvers, so they can be compared.
    """

    sel = selected & candidate
    value = float(score[sel].sum())
    if weight:
        both = sel[edges[:, 0]] & sel[edges[:, 1]]
        value += weight * float(both.sum())
    return value


def patch_metrics(selected, edges):
    """Number of patches (rook-connected components), single-cell patches and boundary edges."""
    ids = np.flatnonzero(selected)
    if len(ids) == 0:
        return {"patches": 0, "single_cell_patches": 0, "boundary_edges": 0}
    pos = -np.ones(len(selected), np.int64)
    pos[ids] = np.arange(len(ids))
    inner = edges[selected[edges[:, 0]] & selected[edges[:, 1]]]
    graph = coo_matrix((np.ones(len(inner)), (pos[inner[:, 0]], pos[inner[:, 1]])),
                       shape=(len(ids), len(ids)))
    n, labels = connected_components(graph, directed=False)
    sizes = np.bincount(labels)
    return {"patches": int(n), "single_cell_patches": int((sizes == 1).sum()),
            "boundary_edges": int((selected[edges[:, 0]] != selected[edges[:, 1]]).sum())}


def _candidate_edges(edges, candidate, protected):
    """Split candidate-candidate edges into free-free edges and free-baseline edges."""
    ce = edges[candidate[edges[:, 0]] & candidate[edges[:, 1]]]
    pa, pb = protected[ce[:, 0]], protected[ce[:, 1]]
    free_free = ce[~pa & ~pb]
    free_base = np.r_[ce[pa & ~pb, 1], ce[~pa & pb, 0]]  # the free end of each free-baseline edge
    return free_free, free_base


def solve_mincut(df, edges, weight, protection_frac=0.30, scale=1e6, tol=1e-6, max_iter=60,
                 verbose=True):
    """
    Solve the connectivity-weighted selection exactly by parametric min-cut.
    Parameters
    ----------
    df : pandas.DataFrame
        Planning units with "objective_score", "candidate", "protected", "area".
    edges : numpy.ndarray
        Rook adjacency (rook_edges), positional indices into df.
    weight : float
        Reward per adjacent pair of selected units (score units). 0 = no connectivity.
    protection_frac : float
        Area cap as a fraction of the total area (baseline included).
    scale : float
        Scores are scaled by this factor and rounded to integers for the
        max-flow solver (resolution 1/scale).
    tol, max_iter : float, int
        Bisection tolerance on the area price and maximum iterations.
    Returns
    -------
    (numpy.ndarray, dict)
        Boolean selection per row of df, and solver info: objective,
        upper_bound, gap_rel, area_price, unused_area, greedy_added, iterations.
    """

    start = time.time()
    score = df["objective_score"].to_numpy(dtype=np.float64)
    candidate = df["candidate"].to_numpy() == 1
    protected = (df["protected"].to_numpy() == 1) & candidate
    area = df["area"].to_numpy(dtype=np.float64)
    budget = protection_frac * float(area.sum())
    base_area = float(area[protected].sum())
    if base_area > budget:
        raise ValueError(f"Baseline area {base_area:,.0f} exceeds the area cap {budget:,.0f}")

    free = np.flatnonzero(candidate & ~protected)
    m = len(free)
    pos = -np.ones(len(df), np.int64)
    pos[free] = np.arange(m)
    free_free, free_base = _candidate_edges(edges, candidate, protected)
    a, b = pos[free_free[:, 0]], pos[free_free[:, 1]]
    deg = np.bincount(np.r_[a, b], minlength=m).astype(np.float64)
    base_bonus = np.bincount(pos[free_base], minlength=m).astype(np.float64) * weight
    s_free, a_free = score[free], area[free]
    cap_left = budget - base_area

    src, snk = m, m + 1
    pair_cap = int(round(weight / 2 * scale))
    pair_rows = np.r_[a, b]
    pair_cols = np.r_[b, a]

    def lagrangian(lam):
        """Exact maximizer of sum (s - lam*area + bonus) x + w sum x_i x_j over free units."""
        # x_i x_j = (x_i + x_j - |x_i - x_j|) / 2 -> unary c_i, pair cost w/2 |x_i - x_j|
        c = s_free - lam * a_free + base_bonus + weight * deg / 2
        cs = np.where(c > 0, np.round(c * scale), 0).astype(np.int64)
        ct = np.where(c < 0, np.round(-c * scale), 0).astype(np.int64)
        rows = np.r_[np.full(m, src), np.arange(m)]
        cols = np.r_[np.arange(m), np.full(m, snk)]
        caps = np.r_[cs, ct]
        if pair_cap > 0:
            rows, cols = np.r_[rows, pair_rows], np.r_[cols, pair_cols]
            caps = np.r_[caps, np.full(len(pair_rows), pair_cap, np.int64)]
        keep = caps > 0
        graph = csr_matrix((caps[keep], (rows[keep], cols[keep])), shape=(m + 2, m + 2), dtype=np.int64)
        graph.sum_duplicates()
        flow = maximum_flow(graph, src, snk, method="dinic").flow
        # Residual graph: remaining capacity forward, plus pushed flow backwards
        residual = (graph - flow).tocsr()
        residual.data[residual.data < 0] = 0
        pushed = flow.tocsr()
        pushed.data[pushed.data < 0] = 0
        residual = (residual + pushed.T).tocsr()
        residual.eliminate_zeros()
        reach = breadth_first_order(residual, src, directed=True, return_predecessors=False)
        x = np.zeros(m, bool)
        x[reach[reach < m]] = True  # source side = selected (smallest optimal set)
        return x

    def value(x):
        both = x[a] & x[b]
        return float(s_free[x].sum() + base_bonus[x].sum() + weight * both.sum())

    # Bisection on the area price: the selected area decreases as the price rises
    lo, hi = 0.0, float(max(s_free.max(initial=0), 0) + weight * 4 + 1.0)
    x_hi, x_lo = np.zeros(m, bool), None
    x0 = lagrangian(0.0)
    iterations = 1
    if a_free[x0].sum() <= cap_left:
        x_hi, hi = x0, 0.0
    else:
        x_lo = x0
        while iterations < max_iter and hi - lo > tol:
            lam = (lo + hi) / 2
            x = lagrangian(lam)
            iterations += 1
            if a_free[x].sum() <= cap_left:
                x_hi, hi = x, lam
            else:
                x_lo, lo = x, lam

    # Upper bound from the Lagrangian dual: L(lam) = max_x f(x) - lam (A(x) - B) >= OPT
    bounds = [value(x_hi) + hi * (cap_left - a_free[x_hi].sum())]
    if x_lo is not None:
        bounds.append(value(x_lo) + lo * (cap_left - a_free[x_lo].sum()))
    upper_bound_free = min(bounds)

    # Greedy fill of the unused budget: add the unit with the largest marginal gain
    x = x_hi.copy()
    used = float(a_free[x].sum())
    nbr = coo_matrix((np.ones(2 * len(a)), (np.r_[a, b], np.r_[b, a])), shape=(m, m)).tocsr()
    sel_nbrs = np.asarray(nbr @ x.astype(np.float64)).ravel()
    added = 0
    while True:
        gain = s_free + base_bonus + weight * sel_nbrs
        gain[x | (a_free > cap_left - used + 1e-9)] = -np.inf
        k = int(np.argmax(gain))
        if not np.isfinite(gain[k]) or gain[k] <= 0:
            break
        x[k] = True
        used += a_free[k]
        added += 1
        row = nbr.indices[nbr.indptr[k]:nbr.indptr[k + 1]]
        sel_nbrs[row] += 1

    selected = protected.copy()
    selected[free[x]] = True
    objective = objective_value(score, selected, candidate, edges, weight)
    constant = objective - value(x)  # baseline scores and baseline-baseline terms
    upper_bound = upper_bound_free + constant
    info = {
        "solver": "mincut",
        "objective": objective,
        "upper_bound": upper_bound,
        "gap_rel": (upper_bound - objective) / abs(upper_bound) if upper_bound else 0.0,
        "area_price": hi,
        "unused_area": cap_left - used,
        "greedy_added": added,
        "iterations": iterations,
        "free_free_edges": int(len(a)),
        "solve_sec": round(time.time() - start, 1),
    }
    if verbose:
        print(f"Min-cut: {iterations} cuts, area price {hi:.4f}, greedy fill +{added} units, "
              f"objective {objective:,.1f}, upper bound {upper_bound:,.1f} (gap {info['gap_rel']:.2e}), "
              f"{info['solve_sec']}s")
    return selected, info
