# Dependency Version Policy

**Last Updated**: 2025-11-24 | **Repository Version**: 0.6.3

## Constraint Strength Principle

**Foundation libraries** (e.g., `owa-core`, `owa-msgs`, `mcap-owa-support`) should have **weak/minimal constraints** to maximize compatibility and allow downstream packages flexibility.

**End-user libraries** (e.g., `owa-cli`, `owa-data`, `owa-mcap-viewer`) can have **stricter constraints** since they are leaf nodes in the dependency graph and don't need to accommodate downstream consumers.

**Rationale**: Foundation libraries are consumed by many packages, so overly strict constraints can cause dependency conflicts. End-user applications have no downstream consumers, so they can safely pin to specific versions for reproducibility.

---

## Version Pinning Strategy

| Type | Constraint | Example | Rationale |
|------|-----------|---------|-----------|
| **First-party** | `==X.Y.Z` | `owa-core==0.6.2` | Lockstep versioning |
| **MediaRef** | `~=X.Y.Z` | `mediaref~=0.4.1` | Compatible release (patch updates only) |
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
| `jsonargparse[signatures]` | `>=4.41.0` | KEYWORD_ONLY parameter handling fix (#756) |
| `line-profiler` | `>=4.1.0` | Global `@line_profiler.profile` decorator introduced |
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

`loguru`, `tqdm`, `orjson`, `annotated-types`, `jinja2`, `python-dotenv`, `diskcache`, `griffe`, `plotext`, `webdataset`, `pygobject-stubs`, `pygetwindow`, `bettercam`, `pydantic-settings`, `python-multipart`

### Special Cases

- **MediaRef**: `~=0.4.1` (compatible release)
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
