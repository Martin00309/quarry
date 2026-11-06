# Contributing

Thanks for your interest in Quarry. It's a personal project, but issues and
pull requests are welcome.

## Getting set up

```bash
git clone https://github.com/Martin00309/quarry
cd quarry
python -m venv .venv && source .venv/bin/activate
pip install -e .
pip install pytest ruff
```

## Before you open a pull request

- `pytest` passes.
- `ruff check .` is clean (the config lives in `pyproject.toml`).
- New behavior comes with tests. The suite is organized one file per module
  (`tests/test_<module>.py`); follow the nearest existing file for style.
- Keep changes focused. The codebase is deliberately layered — analysis, then
  postings, then segments, then the index, then query/search — so a change
  usually belongs to one layer.

## Design notes

- **Pure standard library.** Please don't add runtime dependencies.
- **Behavior, not internals.** Tests should assert on observable behavior
  (what a search returns, what a round-trip preserves) rather than private
  structure, so the internals stay free to change.
- **On-disk format changes** must bump the segment version and be described in
  `docs/format.md`.

## Reporting bugs

Open an issue with a minimal schema, the documents, the query and what you
expected versus what you got. A failing test is the ideal bug report.
