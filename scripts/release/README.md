# OWA Release Manager

A CLI tool for managing OWA package releases with lockstep versioning and beautiful, readable output.

## ✨ Recent Improvements

The script output has been significantly improved for better readability:

### Before (cluttered output):
```
=======================
Processing package in projects/owa-core
Running: vuv add owa-msgs==1.0.0 --frozen
✓ Updated owa-msgs dependency to 1.0.0
Running: vuv version 1.0.0
✓ Updated owa-core version to 1.0.0 using vuv
=======================
```

### After (clean, organized output):
```
📦 owa-core (projects/owa-core)
   Dependencies: owa-msgs
   ✓ Updated owa-msgs → 1.0.0
   ✓ Updated version → 1.0.0 (vuv)

🔒 Updating lock files...
   ✓ owa-core

🎉 Success! Updated 8 packages to version 1.0.0
```

**Key improvements:**
- 🎨 **Rich formatting** with colors, emojis, and visual hierarchy
- 📦 **Clean package display** with organized information
- ⚡ **Spinner animations** for long-running operations
- 🚫 **Removed verbose command echoing** (commands run silently unless errors occur)
- 📋 **Better status reporting** with clear success/warning/error indicators
- 🎯 **Focused output** showing only essential information

## Installation

The script requires Python 3.8+ and the following dependencies:

```bash
pip install typer packaging rich tomli  # tomli only needed for Python < 3.11
```

## Release Workflow

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
# - Update dependencies using 'vuv add x==v --frozen'
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

## Commands

### Version Command

Update all package versions to a specific version:

```bash
python main.py version 1.0.0
```

Options:
- `--lock/--no-lock`: Update uv.lock files (default: enabled)
- `--tag/--no-tag`: Create git tag and commit changes (default: enabled)
- `--push`: Push changes to git remote (default: disabled)

### Publish Command

Build and publish all packages to PyPI:

```bash
export PYPI_TOKEN=your_token_here
python main.py publish
```

### Lock Command

Run `vuv lock` with optional arguments in all repositories:

```bash
python main.py lock
python main.py lock --upgrade
```

## Why Rebase?

- ✅ Clean linear git history
- ✅ Version tags point to actual commits, not merge commits
- ✅ Better compatibility with release tooling
- ✅ Easier to track version progression
- ✅ Cleaner release notes generation
