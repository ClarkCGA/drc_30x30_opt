# Archive

Configs (`archive/configs/`) and outputs (`archive/output/`) of the first round
of scenarios, kept for reference. They were produced by the previous version of
the code (see git history before branch `client-revisions-themes-star`) and use
the old scenario names ("PA Only Approach", "Mixed Approach", now renamed
"PAs as baseline" and "PAs+CFCLs as baseline"), the old per-layer weights, and
the oil-block exclusion. They are not read by the current notebooks.

Note: the archived GeoTIFFs are shifted by 5 km west / 3 km north relative to
the planning-unit grid (the raster origin was taken from `projection_info.csv`
instead of the planning units); this is fixed in `scripts/utils.export_to_geotiff`.
