# Release Workflow

This document outlines the recommended workflow for creating and publishing OWA releases.

## Overview

Use a dedicated release branch with rebase merging to maintain clean git history and proper version tagging.

## Workflow Steps

### 1. Create Release Branch

```bash
# Start from latest main
git checkout main
git pull origin main

# Create release branch
git checkout -b release/v1.0.0
```

### 2. Update Versions

```bash
# Use the release script to update all package versions
python scripts/release/main.py version 1.0.0

# This will:
# - Detect first-party dependencies for each package
# - Update dependencies using 'vuv add x==v --locked'
# - Update package version using 'vuv version v' or 'hatch version v'
# - Create a commit with version changes and git tag (default)

# Optional flags:
python scripts/release/main.py version 1.0.0 --lock --push
```

### 3. Create Pull Request

```bash
# Push release branch
git push origin release/v1.0.0

# Create PR: release/v1.0.0 → main
# Use GitHub UI or gh CLI
```

### 4. Merge with Rebase

**Option A: GitHub UI**
- Use "Rebase and merge" button in the PR

**Option B: Command Line**
```bash
git checkout main
git rebase release/v1.0.0
git push origin main
```

### 5. Push Tag

```bash
# Push the version tag to remote
git push origin v1.0.0
```

### 6. Publish Packages

```bash
# Set PyPI token
export PYPI_TOKEN=your_token_here

# Publish all packages
python scripts/release/main.py publish
```

### 7. Clean Up

```bash
# Delete release branch
git branch -d release/v1.0.0
git push origin --delete release/v1.0.0
```

## Why Rebase?

- ✅ Clean linear git history
- ✅ Version tags point to actual commits, not merge commits
- ✅ Better compatibility with release tooling
- ✅ Easier to track version progression
- ✅ Cleaner release notes generation

## Release Script Commands

- `version <version>` - Update package versions using vuv/hatch with dependency management
  - `--lock` - Update uv.lock files after version changes
  - `--tag/--no-tag` - Create git tag and commit changes (default: true)
  - `--push` - Push changes to remote repository
- `lock [ARGS]` - Run `vuv lock ARGS` in all first-party repositories
  - Example: `lock --upgrade` to upgrade all dependencies
- `publish` - Build and publish to PyPI

See `python scripts/release/main.py --help` for full options.

**Note:** The `upgrade-all` command has been removed. Use `lock --upgrade` instead.
