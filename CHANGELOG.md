# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0/).

## [Unreleased]

### Security
- Search results are rebuilt with DOM APIs and book cards navigate through a
  delegated `data-book-id` listener, replacing the field-into-`innerHTML` and
  inline `onclick` interpolation that was a latent DOM-XSS sink.

### Changed
- Moved the scientific stack to pandas 3, numpy 2, and scikit-learn 1.9, and
  pinned each dependency to a tested major range (`pandas>=3,<4`, `numpy>=2.1,<3`,
  `scikit-learn>=1.9,<2`, `flask>=3.1,<4`) so `uv lock --upgrade` cannot cross a
  major on its own.

### Added
- `pip-audit` dependency scan in CI.

### Removed
- Stale `notebooks/` analysis from the original pickle workflow; evaluation now
  lives in `scripts/evaluate.py`.

## [1.0.0] - 2026-06-06

Rewrite from the original flat scripts into the `bookrec` package.

### Added
- `bookrec` package: a validated CSV loader, an in-memory TF-IDF recommender, and
  a `create_app` factory with a WSGI entry point.
- A ~1,500-book catalog (a subset of Best Books Ever) with a build script, an
  evaluation script, packaging, a lockfile, Docker/Render config, and CI.

### Fixed
- Book detail pages no longer 404: ids are read as strings so URL lookups match.
- Data loads under WSGI, not only when run as `__main__`.
- Search treats the query literally, so regex metacharacters can't crash it.

### Changed
- Recommendations are built in memory at startup, removing the pickled artifacts
  that could drift from the data.
