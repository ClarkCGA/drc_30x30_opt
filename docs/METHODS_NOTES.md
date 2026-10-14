# Methods notes

Running notes on methodological choices and data caveats in the optimizer,
written to be carried into the report once the analysis is final. Figures
refer to the 1 km planning-unit table (2,328,518 cells).

## 1. No-data handling of scored layers

All scored layers are rescaled to 0-1 (min-max) before weighting. Missing
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
  weights). Without species, KBA receives the full share.
- Score = Σ weight × rescaled maximize layer − Σ weight × rescaled minimize
  layer; cells with score ≤ 0 or failing an exclusion rule are not candidates.

## 3. Other data notes

- `pop_sum` has three negative values (-200, -400) and is so right-skewed
  that after rescaling it is ~0 almost everywhere; `mean_MarinePressure` is
  likewise ~0 almost everywhere. Both therefore have almost no effect.
- Oil blocks are no longer an exclusion; the newly selected area inside oil
  blocks (km², area-weighted by `oil_frac`, and % of newly selected area) is
  reported per scenario in `output/scenario_summary.csv`.

## 4. Raster georeferencing fix

GeoTIFFs exported before October 2026 are shifted 5 km west and 3 km north:
the raster origin was taken from `data/projection_info.csv` (the corner of
the Earth Engine export grid, which has 5 empty columns on the west and 3
empty rows at the top), while the raster array started at the first planning
unit. The origin is now derived from the planning units. The optimization
itself and the per-run PNG maps were not affected; the earlier consensus map
was (its PA mask was offset by the same amount).
