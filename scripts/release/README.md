# OWA Release Manager

CLI tool for managing OWA package releases with lockstep versioning and clean output.

## Installation

Note that if you execute script with `uv run`, you don't need to install anything.

```bash
pip install typer packaging rich tomli  # tomli only for Python < 3.11
```

## Quick Start

```bash
# Update all packages to version 1.0.0 with tagging
vuv run scripts/release/main.py version 1.0.0

# Publish to PyPI
export PYPI_TOKEN=your_token_here
vuv run scripts/release/main.py publish

# Update lock files
vuv run scripts/release/main.py lock --upgrade
```

## Release Workflow

1. **Create branch**: `git checkout -b release/v1.0.0`
2. **Update versions**: `vuv run scripts/release/main.py version 1.0.0`
3. **Create PR**: `git push origin release/v1.0.0`
4. **Merge with rebase** (maintains clean git history)
5. **Push tag**: `git push origin v1.0.0`
6. **Publish**: `vuv run scripts/release/main.py publish`

Above workflow can be automated with the release script:

```bash
$ vuv run scripts/release/main.py version $VERSION --tag --push
$ vuv run scripts/release/main.py publish
```

## Commands

- `version <ver>` - Update all package versions in lockstep
  - `--lock/--no-lock` - Update uv.lock files (default: on)
  - `--tag/--no-tag` - Create git tag and commit (default: on)
  - `--push` - Push to remote (default: off)
- `publish` - Build and publish all packages to PyPI
- `lock [args]` - Run `vuv lock` in all repositories