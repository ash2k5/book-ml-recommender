# 3. code MIT, bundled dataset CC BY-NC

- Status: accepted
- Date: 2026-06-20

## Context
The app code is MIT. The bundled `api/data/books.csv` is a derived subset of the Best Books Ever
dataset, which is CC BY-NC 4.0 (noncommercial).

## Decision
Keep the root `LICENSE` as plain MIT, so it is the code license and GitHub detects it. Document the
dataset's source, attribution, and CC BY-NC terms in `api/data/README.md`, next to the data.

## Consequences
The dataset may be used for noncommercial purposes with attribution. Do not fold its terms back into
`LICENSE`. Anyone reusing the data reads `api/data/README.md`.
