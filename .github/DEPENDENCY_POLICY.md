# Dependency Version Policy

**Last Updated**: 2025-10-30 | **Repository Version**: 0.6.2

## Constraint Strength Principle

**Foundation libraries** (e.g., `owa-core`, `owa-msgs`, `mcap-owa-support`) should have **weak/minimal constraints** to maximize compatibility and allow downstream packages flexibility.

**End-user libraries** (e.g., `owa-cli`, `owa-data`, `owa-mcap-viewer`) can have **stricter constraints** since they are leaf nodes in the dependency graph and don't need to accommodate downstream consumers.

**Rationale**: Foundation libraries are consumed by many packages, so overly strict constraints can cause dependency conflicts. End-user applications have no downstream consumers, so they can safely pin to specific versions for reproducibility.

---

## Version Pinning Strategy

| Type | Constraint | Example | Rationale |
|------|-----------|---------|-----------|
| **First-party** | `==X.Y.Z` | `owa-core==0.6.2` | Lockstep versioning |
| **MediaRef** | `~=X.Y.Z` | `mediaref~=0.3.1` | Compatible release (patch updates only) |
| **Breaking changes** | `>=X.Y.Z` | `pydantic>=2.0` | Minimum version for required API |
| **Unstable APIs** | `>=latest` | `fastapi>=0.115.12` | Pin to latest known-working version |
| **Stable APIs** | No constraint | `loguru` | Backward compatible |

---

## Dependency Rationale

### Core Dependencies (Breaking Changes)

| Package | Constraint | Rationale |
|---------|-----------|-----------|
| `pydantic` | `>=2.0` | Pydantic v2 has breaking API changes from v1 |
| `numpy` | `>=2.0` | NumPy 2.0 breaking changes (C API, dtype behavior) |
| `av` | `>=15.0` | FFmpeg 7.0 support introduced in PyAV 15.0 |
| `pillow` | `>=9.4.0` | PIL.Image.ExifTags introduced in 9.4.0 |
| `pyyaml` | `>=6.0` | Security fixes (CVE-2020-14343) + Python 3.11 |
| `packaging` | `>=20.0` | PEP440 version parsing |
| `requests` | `>=2.32.2` | Security fixes and compatibility |
| `torch` | `>=2.0` | PyTorch 2.x performance improvements |
| `datasets` | `>=4.0` | HuggingFace Datasets 4.x API improvements |
| `transformers` | `>=4.52.1` | InternVL support introduced in 4.52.1 |
| `huggingface_hub` | `>=0.30.0` | Aligned with transformers 4.52.1 requirements |
| `jsonargparse[signatures]` | `>=4.27.7` | I don't have idea but Lightning CLI has this constraint, so I believe them |
| `mcap` | `>=1.0.0` | MCAP 1.0 stable API |
| `typer` | `>=0.20.0` | Modern features and bugfixes which affects UI/UX directly |
| `rich` | `>=14.1.0` | Modern features and bugfixes which affects UI/UX directly |

### Unstable APIs (Pin to Latest)

| Package | Constraint | Rationale |
|---------|-----------|-----------|
| `fastapi[standard]` | `>=0.115.12` | Rapidly evolving API - pin to latest tested version |

### Platform-Specific

| Package | Constraint | Platform | Rationale |
|---------|-----------|----------|-----------|
| `pywin32` | `>=307` | Windows | Python 3.11 support |
| `pyobjc-framework-*` | `>=10.1` | macOS | macOS 11+ compatibility |
| `evdev` | `<1.9.2` | Linux | v1.9.2 build fails |
| `pynput` | `>=1.8.0` | All | Stability fixes |

### No Constraints (Stable APIs)

`loguru`, `tqdm`, `orjson`, `annotated-types`, `jinja2`, `python-dotenv`, `diskcache`, `griffe`, `plotext`, `line-profiler`, `webdataset`, `pygobject-stubs`, `pygetwindow`, `bettercam`, `pydantic-settings`, `python-multipart`

### Special Cases

- **MediaRef**: `~=0.3.1` (compatible release)
- **Conda pygobject**: `=3.50.0` (exact - breaks plugin detection if changed)

---

## Workflow

### Adding Dependencies
1. Check changelog for breaking changes
2. Choose constraint: First-party (`==`), Breaking (`>=`), Unstable (`>=latest`), Stable (none)
3. Document rationale in this file
4. Test: `pytest`
5. Lock: `uv lock --upgrade`

### Updating Dependencies
```bash
# First-party (lockstep)
uv run scripts/release/main.py version 0.7.0

# Third-party
uv lock --upgrade-package <package>
uv lock --upgrade  # all packages
```

---

## Verified Dependencies

All dependencies have been audited (2025-10-30) and confirmed as actively used:

### Core Packages
- **`griffe`** (owa-core): Used in `documentation/validator.py` for OEP-0004 validation
- **`pyyaml`** (owa-core): Used in `plugin_spec.py` for YAML plugin specs
- **`annotated-types`** (owa-msgs): Used in `desktop/keyboard.py` for type constraints
- **`requests`** (owa-cli, mcap-owa-support): GitHub API calls and remote MCAP file support
- **`plotext`** (owa-cli): Terminal-based visualizations in `video/probe.py`
- **`line-profiler`** (owa-data): Production profiling in `collator.py`
- **`transformers`** (owa-data): ML model integration for tokenization

### Removed Dependencies
- **`importlib-metadata`**: Removed from owa-core (Python <3.10 support, project requires >=3.11)

---

## Recent Changes (2025-10-30)

### Completed
1. ✅ **Removed dead code**: `importlib-metadata` from owa-core
2. ✅ **Fixed version constraints**:
   - `owa-data`: Added `>=0.6.2` to first-party deps
   - `owa-env-example`: Updated `owa-core>=0.4.0` → `>=0.6.2`
   - `owa-mcap-viewer`: Updated all first-party deps `>=0.5.6` → `>=0.6.2`
3. ✅ **Verified all dependencies**: All packages have minimal, required dependencies only
4. ✅ **Implemented full dependency policy** across all 9 packages:
   - **Breaking changes**: Applied `>=X.Y.Z` constraints (pydantic>=2.0, numpy>=2.0, av>=15.0, pillow>=9.4.0, pyyaml>=6.0, packaging>=20.0, requests>=2.32.2, torch>=2.0, datasets>=4.0, transformers>=4.52.1, huggingface_hub>=0.30.0, jsonargparse[signatures]>=4.27.7)
   - **Stable APIs**: Removed version constraints (loguru, rich, tqdm, typer, orjson, annotated-types, jinja2, python-dotenv, diskcache, griffe, plotext, line-profiler, webdataset, mcap, pygobject-stubs, pygetwindow, bettercam, pydantic-settings, python-multipart, opencv-python, opencv-python-headless)
   - **Unstable APIs**: Maintained pin to latest (fastapi[standard]>=0.115.12)
   - **Platform-specific**: Maintained constraints (pywin32>=307, pyobjc>=10.1, pynput>=1.8.0, evdev<1.9.2)
   - **Special cases**: Maintained MediaRef `~=0.3.1` and first-party `==0.6.2`
5. ✅ **Verified all changes**: All packages lock successfully, imports work, CLI functional
6. ✅ **Applied stricter HuggingFace constraints**:
   - `datasets>=4.0` (HuggingFace Datasets 4.x API improvements)
   - `transformers>=4.52.1` (InternVL support)
   - `huggingface_hub>=0.30.0` (aligned with transformers 4.52.1)
   - `jsonargparse[signatures]>=4.27.7` (Lightning CLI compatibility)
7. ✅ **Applied MCAP constraint**:
   - `mcap>=1.0.0` (MCAP 1.0 stable API)
8. ✅ **Cleaned up opencv-python dependencies**:
   - Removed from `owa-core` (not used)
   - Added to `owa-cli` (directly uses cv2)
   - Removed from `owa-env-desktop` (gets transitively from owa-msgs)
   - Removed version constraint `>=4.10.0` (no breaking changes requiring specific version)
9. ✅ **Applied typer and rich constraints**:
   - `typer>=0.20.0` (modern features and bugfixes)
   - `rich>=14.1.0` (modern features and bugfixes)
10. ✅ **Comprehensive dependency cleanup**:
   - **owa-core**: Removed `rich` runtime dependency (only used in `TYPE_CHECKING`)
   - **mcap-owa-support**: No changes (datasets/pyarrow are optional features)
   - **owa-cli**: Removed unused `pydantic`, added missing `numpy>=2.0` (pygetwindow is optional feature)
   - **owa-data**: Removed unused `jsonargparse`, `orjson`, `rich`, `typer`, `webdataset`; added missing `fsspec`, `loguru`, `numpy>=2.0`, `pillow>=9.4.0`, `pydantic>=2.0`
   - **owa-env-desktop**: Added missing `loguru` and `numpy>=2.0`; sorted dependencies alphabetically (platform-specific deps are conditionally imported and kept)
   - **owa-env-example**: Added missing `loguru`
   - **owa-env-gst**: Removed unused `pillow`, added missing `loguru` and `numpy>=2.0`
   - **owa-msgs**: No changes (pillow is optional feature for `to_pil_image()`)
   - **owa-mcap-viewer**: Added missing `fsspec`, `jinja2`, `numpy>=2.0`, `pydantic>=2.0`; removed unused `python-multipart` (transitive from fastapi)
11. ✅ **Standardized numpy version constraint**: All packages now consistently use `numpy>=2.0` (breaking changes constraint)
12. ✅ **Removed opencv-python version constraint**: Changed from `opencv-python>=4.10.0` to `opencv-python` (no breaking changes requiring specific version)
