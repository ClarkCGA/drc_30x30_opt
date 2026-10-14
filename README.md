# 30 x 30 Spatial Planning Toolbox for the Democratic Republic of Congo (DRC)
This repository contains the code for a spatial planning tool to support implementation of the 30 x 30 initiative targets in the Democratic Republic of Congo (DRC). Specifically, this tool utilizes several geospatial datasets to find the most spatially optimized scenario for protecting new areas within DRC to meet the 30 x 30 target. 

The tool is built based on a dataset developed at spatial planning units of 1 km x 1 km. For each planning unit, a set of variables are calculated from existing products (e.g. above ground biomass). Existing protected areas are also mapped to the 1 km x 1 km grid, and they are input as constraints to the model.

This code should be used as a tool to run different scenarios for protection. The current scenario set (10 runs) is described below; the first round of scenarios is kept in `archive/`.

## Requirements
### Software
This toolbox is developed in Python. To ensure reproducibility of results, the tool is wrapped in a Docker environment. To run the code locally, you need to install [Docker](https://www.docker.com/products/docker-desktop/) for free and follow the instructions below. 

If you are running this on Windows, it's recommended to run it inside Windows Subsystem for Linux (WSL). Follow the instructions [here](https://learn.microsoft.com/en-us/windows/wsl/install) to enable WSL in your Windows. 

### Data
- **Planning units**: a CSV file (`data/drc_1km_data_planning_units.csv`, ~2.33M cells) containing all the required variables derived from products available on Google Earth Engine (GEE). You can use the tool implemented [here](https://github.com/ClarkCGA/drc_30x30_data_prep) to generate the CSV file. Convert it once to Parquet so each run reads only the columns it needs:
  ```bash
  python -m scripts.convert_to_parquet   # writes data/drc_1km_data_planning_units.parquet
  ```
  (If the Parquet file is missing, the CSV is read instead, with column selection.) The legacy column `cf_frac` is read as `cfcl_frac` (CFCL = Concession Forestière de Communauté Locale, Local Community Forest Concession).
- **STAR species layer**: produced by the STAR pipeline in `drc_30x30_data_prep`: a wide Parquet keyed by `grid_id` with one `sp_<id_no>` column per species (per-cell STAR values, already including the IUCN category weight), plus a species table (`id_no`, name, class, category, weight, global km², DRC km², source). Default paths: `data/star/star_1km_species.parquet` and `data/star/star_species.csv` (set in `configs/common/defaults.yml`, or override with the `DRC_STAR_PATH` / `DRC_STAR_SPECIES_TABLE` environment variables). The species columns are summed per cell into a single STAR total in row-group/column batches, so the ~4,000 columns are never loaded at once.

## Scenario design

### Baselines
- **PAs as baseline**: existing protected areas (`protected_frac` > 0.8) count towards the 30% and are locked in. CFCLs are not counted, and units with more than 30% CFCL cover are excluded.
- **PAs+CFCLs as baseline**: existing protected areas and CFCLs (`cfcl_frac` > 0.8) count towards the 30% and are locked in, assuming all CFCLs will eventually be recognized as OECMs.

The baseline areas are not "the areas to conserve": the optimization selects *new* areas that, together with the baseline, reach 30% of the country.

### Themes
All layers are grouped into seven themes (`configs/common/themes.yml`):

| Theme | Maximize | Minimize |
|---|---|---|
| Biodiversity | STAR species total (`sp_<id_no>` sum), `kba_frac` | |
| Ecological integrity | `mean_FLII`, `mean_PeatThickness`, `natural_semi_grassland_frac`, Shapiro classes (`DenseEvergreen_frac`, `SemiDeciduous_frac`, `SemiDeciduousPioneer_frac`, `Mangrove_frac`, `Maranthaceae_frac`, `OpenForest_frac`, `SwampForest_frac`), FACET classes (`FACET_primary_forest_frac`, `FACET_secondary_forest_frac`, `FACET_savanna_frac`, `FACET_water_frac`), `TMF_undisturbed_frac` | `mean_GlobalHumanModification`, `mean_Anthropogenic_current_pressure`, `mean_Anthropogenic_future_pressure`, `pop_sum`, `TMF_ongoing_dd_frac`, `TMF_regrowth_frac` |
| Climate mitigation (carbon) | `mean_esa_agb`, `mean_ornl_agb`, `mean_ornl_bgb`, `mean_GSOC`, `mean_PeatCarbon` | |
| Climate adaptation | | `mean_climate_vulnerability` |
| Freshwater | `mean_recharge`, `mean_discharge`, `mean_runoff_depth`, `lake_wetland_frac`, `river_density_frac` | `max_river_order`, `mean_erosion`, `mean_MarinePressure` |
| Ecosystem services | `mean_ESP` | |
| Conservation priorities & spatial planning | `mean_GCP_Dinerstein`, `gc_frac` | |

### Weighting
- Every layer is rescaled to 0-1 (min-max); no-data cells are set to 0 before rescaling.
- Balancing applies **only to maximize layers**: each theme gets a share of 1 (2 in its priority run), split equally across the theme's maximize layers.
- Biodiversity is split 50% KBA / 50% species. The species half is the per-cell sum of the STAR values (which already include the category weights), so species contribute in proportion to their STAR values. In runs without species, KBA gets the full Biodiversity share.
- Minimize layers are not balanced: each has weight 1.0 and is subtracted.
- Score per cell = Σ(weight × rescaled maximize layer) − Σ(1.0 × rescaled minimize layer). Units with score ≤ 0 or failing an exclusion rule are not candidates (baseline units are always included).
- Climate adaptation has no maximize layer, so in practice the maximize side sums to 6 (7 in a priority run) and the minimize side to 10.

### Scenarios (`configs/`)
For each baseline:

| Variant | Species | Priority |
|---|---|---|
| Balanced | yes | none |
| Balanced, no species | no | none |
| Biodiversity x2 | yes | Biodiversity share 2 |
| Ecological integrity x2 | yes | Ecological integrity share 2 |
| Climate mitigation x2 | yes | Climate mitigation share 2 |

Config files are `01`–`10_config_<baseline>_<variant>.yml` (e.g. `01_config_pas_baseline_balanced.yml`, `02_config_pas_cfcl_baseline_balanced.yml`), and scenario names follow `"<baseline> - <variant>"` (e.g. `PAs+CFCLs as baseline - Biodiversity x2`); output files are named after the scenario name.

### Exclusions, area cap and solver
- Exclusion rules (`configs/common/defaults.yml`): urban/peri-urban/rural settlement classes and TMF deforested/degraded above thresholds; plus CFCL > 30% in the *PAs as baseline* runs. Oil blocks are **not** excluded.
- Area cap: 30% of the total area (baseline included). CBC via PuLP, 2000 s time limit, 0.1% relative gap.

### Outputs
In `output/`, per scenario: `<name>.png/.tif` (selected areas), `<name> - Baseline Areas.png/.tif`, `<name> - Objective Score.png`, `<name> - Candidates.png`. `scenario_summary.csv` has one row per scenario: areas, newly selected area inside oil blocks (km², area-weighted by `oil_frac`, and % of the newly selected area), and score diagnostics (score distribution, number of eligible cells, share of cells where the minimize side outweighs the maximize side). `visualize_consensus.ipynb` writes `consensus_map.png/.tif`: the number of runs (out of 10) that newly select each unit, with baseline units as a separate class and oil-block outlines overlaid.

## Configurations
Each scenario config in `configs/` `extends` two shared files, which are merged first:
- `configs/common/themes.yml`: themes → maximize/minimize layers, Biodiversity shares, base share and minimize weight.
- `configs/common/defaults.yml`: data and STAR paths, exclusion rules, area cap and solver settings.

A scenario config only sets its `name`, `baseline`, `data.protected_classes`, `species.include`, `weighting.priority` (e.g. `{biodiversity: 2.0}`) and any `optimization.extra_constraint_cols`. To create a new scenario, copy one of the configs and change these keys; to change a theme for all scenarios, edit `themes.yml`.

## Instructions

### 1. Clone the Repository

```bash
git clone git@github.com:ClarkCGA/drc_30x30_opt.git
cd drc_30x30_opt
```

### 2. Build the Docker Image

```bash
docker build -t drc-30x30-opt .
```

### 3. Run the Docker Container

```bash
docker run -it -p 8888:8888 -v $(pwd):/app drc-30x30-opt
```
This will print out the URL to the Jupyter Lab (including its token). Copy the URL, and paste into a browser to launch Jupyter Lab. 

### 4. Prepare the Data

Convert the planning-unit CSV to Parquet once (`python -m scripts.convert_to_parquet`) and place the STAR outputs under `data/star/` (or set `DRC_STAR_PATH`).

### 5. Execute the Tool 

Open and run all cells in the Jupyter notebook `main_opt.ipynb`

After running the optimization for all (or a selection of) the configurations, you can run the Jupyter notebook `visualize_consensus.ipynb` to visualize the consensus map among the outputs. 

### Testing without the real STAR data

```bash
python -m scripts.make_dummy_star                       # synthetic STAR Parquet in data/star_dummy/ (gitignored)
python -m scripts.test_pipeline --out-dir output_test   # weight/reader checks, full-grid score diagnostics,
                                                        # end-to-end runs of all 10 scenarios on a 150 x 150 km subset + consensus map
```

## Known issues
- **Connectivity bonus is inactive.** `neighbors` is stored as a string (e.g. `"[4.0, 5.0]"`), so the connectivity term in `setup_model` iterates over characters and never adds a bonus. This is the behaviour of all runs so far and is kept unchanged pending a decision. Note also that the term as written, 0.5·(x_i + x_j) per neighbour pair, is linear: it rewards units by their number of candidate neighbours rather than rewarding adjacent selections.
- Several layers have large no-data areas that are set to 0 before rescaling, e.g. `mean_climate_vulnerability` (~68% of cells; range -0.42 to 1.44, so filled cells sit at ~0.23 after rescaling rather than at the best value) and the anthropogenic pressure layers (~28% of cells, 0 = no pressure).
- `pop_sum` has a few negative values (-200, -400) and is so skewed that after rescaling it is ~0 almost everywhere; `mean_MarinePressure` is likewise ~0 almost everywhere.


## Acknowledgements
This project is funded by the Wildlife Conservation Society (WCS) through a contract with Clark CGA. 

## Support
If you run into any issues running this code, or if you have questions about it you can open a ticket on this GitHub repository or contact Clark CGA team at `cga@clarku.edu`.