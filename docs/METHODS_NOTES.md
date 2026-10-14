# Methods notes

Running notes on methodological choices and data caveats in the optimizer,
written to be carried into the report once the analysis is final. Figures
refer to the 1 km planning-unit table (2,328,518 cells).

## 1. No-data handling of scored layers

All scored layers are rescaled to 0-1 before weighting (min-max, except the
STAR species total, see section 3). Missing
cells must be filled first. The rule is configurable under `fill_na` in
`configs/common/themes.yml` (a number, `"median"` or `"mean"` per layer; the
median/mean is taken over the layer's observed cells on the full grid; in
spatial-subset test runs, over the subset).

| Layer | Missing cells | Fill |
|---|---|---|
| `mean_Anthropogenic_current_pressure` | 644,756 (27.7%) | median of observed cells (0.414; 0.39 after rescaling) |
| `mean_Anthropogenic_future_pressure` | 644,756 (27.7%) | median of observed cells (0.688; 0.64 after rescaling) |
| `mean_climate_vulnerability` | 1,590,411 (68.3%) | median of observed cells (0.434; 0.46 after rescaling) |
| all other scored layers | | 0 |

### Coverage of the pressure and climate vulnerability layers

- **Anthropogenic pressure (current and future).** Source: Earth Engine asset
  `projects/ee-madibipacifique/assets/Anthropogenicpressure` (band `b1` =
  current, `b2` = future), aggregated to the 1 km grid in
  `drc_30x30_data_prep/scripts/gee/gee_export.py`. The asset **stops at about
  6°S**: every cell south of 6°S is missing (615,251 cells, 26.4% of the
  country); north of 6°S only 29,505 cells (1.7%) are missing.
- **Climate vulnerability.** Source: Earth Engine asset
  `projects/ee-madibipacifique/assets/ClimateVulnerability_VF` (band `b4`).
  It also has no data south of about 6°S, and north of 6°S it covers mainly
  the forest block: 56.9% of northern cells are missing; the covered cells are
  87% primary forest (FACET) on average, against 38% for the missing northern
  cells. Overall only 31.7% of cells have a value.
- The provenance of both assets is not documented in the data-prep repository.

### Why the median

Previously all missing values were set to 0. For the pressure layers 0 is
below the observed minimum (0.037 current, 0.141 future), so a quarter of the
country, essentially everything south of 6°S, was scored as having the
lowest possible pressure, the best value for a minimize layer. That
biased the selection towards the south. Filling with the median of the
observed cells makes those cells neutral: they are neither favoured nor
penalized relative to a typical observed cell.

### Caveat for climate vulnerability

The median rule is applied to climate vulnerability as well, but the
missingness is not random: the observed cells are mostly forest, so their
median may not represent savanna or densely populated areas. With 68% of
cells set to the same value, the layer effectively only differentiates cells
within its covered area (mostly the northern forest block); elsewhere the
Climate adaptation theme is flat. It should be confirmed with the data
provider whether no-data means "not assessed" or "out of scope (e.g.
non-forest)"; in the latter case another fill (or a different layer) may be
appropriate.

## 2. Theme weighting

- Seven themes (Biodiversity, Ecological integrity, Climate mitigation,
  Climate adaptation, Freshwater, Ecosystem services, Conservation priorities &
  spatial planning); layers per theme in `configs/common/themes.yml`.
- Each theme gets a share of 1 on the maximize side and, separately, a share
  of 1 on the minimize side (2 on both sides in its priority run), split
  equally across the theme's layers on that side. A theme with no layers on
  one side contributes 0 there. Totals: maximize 6 (Climate adaptation has no
  maximize layer), minimize 3 (only Ecological integrity, Climate adaptation
  and Freshwater have minimize layers).
- Biodiversity's maximize share is split 50% KBA / 50% STAR species total
  (per-cell sum of the STAR values, which already include the IUCN category
  weights; rank-scaled, see section 3). Without species, KBA receives the
  full share.
- Score = Σ weight × rescaled maximize layer − Σ weight × rescaled minimize
  layer; cells with score ≤ 0 or failing an exclusion rule are not candidates.

### Rescaling and the role of minimize layers

All layers keep the 0-1 min-max rescaling. With maximize and minimize sides
each balanced per theme, the maximize side totals 6 and the minimize side 3
(4 in the *Ecological integrity x2* runs). Most minimize layers are
right-skewed, so after rescaling they vary little across eligible cells. The
outcome is that **pressure and other minimize layers act mainly as
tiebreakers**: the selection is driven by the maximize layers. This was
accepted as the intended behaviour.

Dominance check on the full grid (eligible, non-baseline cells; top-N =
the 455,487 cells the 30% budget allows in the *PAs as baseline* runs;
STAR species from a synthetic test file):

| Scenario | Variance of weighted maximize side | Variance of weighted minimize side | Top-N overlap with maximize-only ranking | Top-N overlap with minimize-only ranking |
|---|---|---|---|---|
| Balanced, no species | 0.587 | 0.022 | 94.7% | 50.4% |
| Balanced | 0.517 | 0.022 | 93.5% | 51.7% |
| Ecological integrity x2 | 0.589 | 0.034 | 91.8% | 59.3% |

For comparison, during development: with unbalanced minimize layers
(weight 1.0 each, sum 10) the two sides carried similar variance (top-N
overlap 66% with maximize-only, 30% with minimize-only), and doubling them
in the Ecological integrity x2 run made the minimize side dominate (34% vs
64%). About 93% of cells now have a positive score, so the score threshold
removes few cells; eligibility is set mainly by the exclusion rules.

## 3. STAR species layer

### Input

The species layer is the per-cell STAR total from the STAR pipeline of
`drc_30x30_data_prep` (branch `star-pipeline`, commit 71f1844): the sum over
4,251 species of category weight × share of the species' global range (or
BirdLife Area of Habitat, for all 1,116 birds) inside the cell. The project
uses its own category weights (CR 5, EN 4, VU 3, DD 3, NT 2, LC 1; documented
in the data-prep `docs/STAR_METHODS.md`), so Least Concern species count with
weight 1, unlike standard STAR (LC 0, DD excluded). Checks: `grid_id` joins
1:1 with all 2,328,518 planning units; 2,022 cells (mostly lake and coastal
cells) have no species; the optimizer's batched sum of the species columns
equals the pipeline's `star_total` to within 6×10⁻⁸.

### Why the STAR total is rank-scaled

The STAR total is extremely heavy-tailed. Each species' values sum to its
weight × the share of its global range inside the DRC, so species with tiny
global ranges concentrate their whole weight in a few cells. The median cell
has 0.00054, the 99th percentile 0.0089 and the maximum 3.84, about 7,000
times the median. That maximum cell holds two Critically Endangered
chameleons (*Rhampholeon hattinghi*, *Kinyongia mulyai*) with global ranges of
about 1.6 km².

With min-max rescaling, used for all other layers, the maximum maps to 1 and
almost every other cell to ~0: the rescaled layer averages 0.0003; only 687
cells exceed 0.01 and 21 exceed 0.1. The species half of the Biodiversity
theme then had virtually no influence. The *Biodiversity x2* run (KBA and
species both at weight 1) produced the same selection as the no-species run
(overlap 1.000; the species term was worth ~0.01% of the objective), and
the with-species runs differed from the no-species runs only because KBA's
weight was halved to make room for species.

The STAR total is therefore rescaled by **percentile rank**: cells without
species get 0; cells with species get their rank among those cells divided
by their number (ties share their average rank), so the lowest positive
value gets ~0 and the highest 1. The species layer then differentiates all
cells instead of a few hundred, while keeping their order.

**Trade-off.** Rank scaling keeps the order of cells but drops the
magnitudes: a micro-endemic hotspot ranks at the top but no longer outweighs
thousands of other cells. Absolute STAR differences remain visible in the
reported STAR capture (below). It is also inconsistent with the other layers,
which keep min-max: a rank-scaled layer has a uniform distribution, while
most min-max layers are skewed towards 0, so per unit of weight the species
layer varies more across cells than most other layers. This is intended (it
restores the species share agreed in the weighting) and is limited to this
one layer, whose distribution makes min-max uninformative. The method is
configurable per layer (`scaling` in `configs/common/themes.yml`).

### Effect (full grid, before → after)

| | Min-max | Rank |
|---|---|---|
| Overlap of new areas, Balanced vs Balanced, no species (PAs / PAs+CFCLs) | 0.933 / 0.933 | 0.892 / 0.890 |
| Overlap, Biodiversity x2 vs Balanced, no species | 1.000 / 1.000 | 0.844 / 0.836 |
| Overlap, Biodiversity x2 vs Balanced | 0.933 / 0.933 | 0.893 / 0.870 |
| National STAR captured by new areas, Balanced (PAs) | 14.3% | 15.3% |
| National STAR captured by new areas, Biodiversity x2 (PAs) | 14.4% | 16.5% |
| Threatened-only STAR captured by new areas, Biodiversity x2 (PAs) | 9.0% | 10.0% |
| Units newly selected in all 10 runs | 399,412 | 365,860 |

The gain in captured STAR is modest. The new areas (19.6% of the country in
the *PAs as baseline* runs) still capture less than their area share of STAR
(14-17%) and of threatened STAR (8-10%). The reason is spatial: 72% of the top
1% STAR cells lie in the east (Albertine Rift side, east of 27.5°E), where
the other maximize layers (carbon, conservation priorities, green corridor,
freshwater) are lower than in the central basin, and the species layer is one
half of one of six themes. A stronger species influence would be a weighting
decision (e.g. a larger species share), or a STAR target constraint.

### STAR capture reporting

Every run reports, in `output/scenario_summary.csv`, the share of the national
STAR total captured by the new areas and by baseline + new areas, for all
species (`star_share_*`) and threatened species only (CR, EN, VU;
`star_threatened_share_*`), from `star_1km_summary.parquet`. It is reported
for the no-species runs too, so all scenarios can be compared.

## 4. Spatial connectivity

### Formulation

The objective adds a reward `w` for every pair of selected planning units
that share a 1 km edge (rook adjacency, built from the cell-centre
coordinates):

    maximize  Σ_i s_i x_i + w Σ_(i,j adjacent) x_i x_j
    subject to Σ_i area_i x_i ≤ 30% of total area,
               x_i = 1 for baseline units, x_i = 0 for excluded / score ≤ 0 units.

A pair of a candidate and a baseline unit is a fixed bonus `w` on the
candidate, so new areas are drawn towards existing PAs (and CFCLs in the
*PAs+CFCLs as baseline* runs). This is the standard pairwise
connectivity/boundary reward used in systematic conservation planning
(as in Marxan's boundary length modifier or prioritizr's connectivity
penalties); with equal cell sizes, rewarding shared edges is equivalent to
penalizing the outer boundary of the selection.

The previous implementation had a connectivity term that never took effect
(the neighbour list was read as text, so no neighbour was matched) and that,
as written, rewarded neighbour counts rather than adjacent selections. All
runs before this change therefore had no connectivity effect.

### Weight: w = 0.25

`w` is in score units per shared edge. A typical newly selected cell scores
about 2.3, so w = 0.25 is roughly a tenth of a cell's score per shared edge.
Trade-off on the full grid (*PAs as baseline - Balanced, no species*;
455,487 new cells in every row):

| w | Theme score of new areas | Patches | Single-cell patches | Boundary edges |
|---|---|---|---|---|
| 0 | 1,053,572 | 5,331 | 2,980 | 159,229 |
| 0.1 | -0.08% | 988 | 279 | 109,107 (-31%) |
| **0.25** | **-0.20%** | **414** | **65** | **93,688 (-41%)** |
| 0.5 | -0.37% | 192 | 7 | 83,688 (-47%) |
| 1.0 | -0.74% | 117 | 3 | 72,962 (-54%) |

w = 0.25 removes most of the fragmentation (patches -92%, single-cell
patches -98%, boundary -41%) for 0.2% of the theme score; larger weights
give diminishing returns in compactness for a growing loss of score. `w` is
configurable (`optimization.connectivity.weight`).

(With the earlier, unbalanced scoring the same w = 0.25 cost 1.3% of the
score for a 52% shorter boundary; the current score is spatially smoother,
so the unconnected solution is already less fragmented.)

### Solver

- **Parametric min-cut (default, `solver.method: mincut`).** For a fixed
  price λ per km² of selected area, the problem without the area cap is a
  minimum s-t cut on the adjacency graph and is solved exactly (scipy max
  flow). λ is bisected until the selection just fits under the cap, a greedy
  step then adds the units with the largest marginal gain to use the
  remaining budget, and the Lagrangian dual gives an upper bound on the
  optimum, so the remaining gap is known and reported. On the full grid:
  ~1 minute per run, ~5.5 GB RAM, gap ≈ 5×10⁻⁷.
- **CBC (`solver.method: cbc`).** The same objective as an integer program
  (y_ij ≤ x_i, y_ij ≤ x_j per adjacent pair) with the original CBC settings
  (2000 s, 0.1% gap). On the full grid this means ~1.5M binary variables,
  ~2.8M pair variables and ~5.6M constraints; CBC took 17 s on a 22.5k-cell
  subset and 1,331 s on a 75k-cell subset, so it is not tractable at full
  size. It is kept for verification and for scenarios needing extra
  constraints.
- **Cross-check** (`scripts/test_pipeline.py`, 150 × 150 km subset): without
  connectivity both solvers give the identical objective; with w = 0.25
  min-cut is within 0.007% of the CBC optimum (inside CBC's 0.1% tolerance),
  0.9 s against 21 s.
- **Equivalence of constraints.** The area cap, locked-in baseline units,
  exclusion rules and score threshold are enforced identically by both
  solvers (the latter two by the same candidate filter). Without
  connectivity both reduce to selecting the highest-scoring candidates.
- **Limits of min-cut.** It supports exactly one area cap and non-negative
  pair rewards. Additional budget-type constraints (per-province budgets,
  per-theme or species representation targets) would make it a heuristic,
  and patch-size or contiguity rules cannot be expressed; those would need
  the CBC path on reduced problems.

## 5. Other data notes

- `pop_sum` has three negative values (-200, -400) and is so right-skewed
  that after rescaling it is ~0 almost everywhere; `mean_MarinePressure` is
  likewise ~0 almost everywhere. Both therefore have almost no effect.
- Oil blocks are no longer an exclusion; the newly selected area inside oil
  blocks (km², area-weighted by `oil_frac`, and % of newly selected area) is
  reported per scenario in `output/scenario_summary.csv`.

## 6. Raster georeferencing fix

GeoTIFFs exported before October 2026 are shifted 5 km west and 3 km north:
the raster origin was taken from `data/projection_info.csv` (the corner of
the Earth Engine export grid, which has 5 empty columns on the west and 3
empty rows at the top), while the raster array started at the first planning
unit. The origin is now derived from the planning units. The optimization
itself and the per-run PNG maps were not affected; the earlier consensus map
was (its PA mask was offset by the same amount).
