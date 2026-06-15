# Dataset

`books.csv` is a derived subset (~1,500 rows) of the **Best Books Ever** dataset,
selected for description quality and genre diversity and reformatted for this app.

- Source: Casanova Lozano, L., & Costa Planells, S. (2020). *Best Books Ever
  Dataset* (Version 1.0.0). Zenodo. https://doi.org/10.5281/zenodo.4265096
- Repository: https://github.com/scostap/goodreads_bbe_dataset
- License: **CC BY-NC 4.0** (Attribution-NonCommercial). The dataset and this
  derived subset may be used for noncommercial purposes with attribution.

Cover images are referenced by URL from the source data and are not redistributed
here. Regenerate the subset with `uv run --extra collect python scripts/build_dataset.py`.
