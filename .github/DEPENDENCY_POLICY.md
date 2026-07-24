# Dependency Version Policy

**Last Updated**: 2026-07-24 | **Repository Version**: 0.6.6

## Dependency Scopes

### Runtime dependencies

Dependencies in `[project.dependencies]` are part of a package's published compatibility
contract.

- **Foundation libraries** (for example, `owa-core`, `owa-msgs`, and
  `mcap-owa-support`) should have weak/minimal constraints to maximize downstream
  compatibility.
- **End-user libraries** (for example, `owa-cli`, `owa-data`, and
  `owa-mcap-viewer`) may use stricter constraints when necessary because they are leaf
  nodes in the dependency graph.

### Optional dependencies

Extras in `[project.optional-dependencies]` are also published package metadata, but a
constraint is warranted only when the optional feature relies on an API introduced in
a particular version or a known-bad release must be excluded. Do not copy the version
that happened to be current when the dependency was added into a lower bound.

The `docs` extra intentionally leaves its direct dependencies unconstrained. The
repository-wide `exclude-newer` cutoff bounds the versions that may be resolved.

### Dependency groups

Groups in `[dependency-groups]` are local development inputs, not part of the
published package's runtime compatibility contract. Direct development requirements
should normally be unconstrained. A fixed `exclude-newer` timestamp makes resolution
repeatable without turning the versions that happened to be locked into compatibility
requirements.

Do not add a constraint merely to avoid migrating to a new tool release. In particular,
when a Ruff upgrade enables or changes a lint rule, fix the affected code. Change
`ruff.toml` only when the repository is making an intentional, version-independent
lint policy decision.

---

## Version Pinning Strategy

| Scope/type | Constraint | Example | Rationale |
|------------|------------|---------|-----------|
| **First-party runtime** | `==X.Y.Z` | `owa-core==0.6.6` | Lockstep versioning |
| **Required API** | `>=X.Y.Z` | `pydantic>=2.0` | Minimum version that provides the API in use |
| **Known-bad releases** | Narrow exclusion/range | `evdev<1.9.2` | Documented incompatibility |
| **Stable runtime API** | No constraint | `loguru` | No known compatibility boundary |
| **Optional feature** | No constraint by default | `mkdocs` | `exclude-newer` bounds resolution |
| **Development group** | No constraint by default | `pytest` | `exclude-newer` bounds resolution |

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
| `mediaref` | `>=1.1.1` | Reuses one PyAV reformatter per batch (about 10x faster conversion for small frames) |
| `typer` | `>=0.20.0` | Modern features and bugfixes which affects UI/UX directly |
| `rich` | `>=14.1.0` | Modern features and bugfixes which affects UI/UX directly |
| `lazyregistry` | `>=0.3.0` | API used in owa-core is stable from 0.3.0 onwards |

### Platform-Specific

| Package | Constraint | Platform | Rationale |
|---------|-----------|----------|-----------|
| `pywin32` | `>=307` | Windows | Python 3.11 support |
| `pyobjc-framework-*` | `>=10.1` | macOS | macOS 11+ compatibility |
| `evdev` | `<1.9.2` | Linux | v1.9.2 build fails |
| `pynput` | `>=1.8.0` | All | Stability fixes |

### No Constraints (Stable APIs)

`loguru`, `tqdm`, `orjson`, `annotated-types`, `jinja2`, `python-dotenv`, `diskcache`, `griffe`, `plotext`, `webdataset`, `pygobject-stubs`, `pygetwindow`, `bettercam`, `pydantic-settings`, `python-multipart`

### Optional and Development Dependencies

The direct packages in the `docs` extra and `dev` dependency group have no version
constraints. Their resolution window is bounded by `exclude-newer`.

### Special Cases

- **Conda pygobject**: `=3.50.0` (exact - breaks plugin detection if changed)

---

## Reproducible Resolution and Dependency Cooldown

Every project sets the same fixed RFC 3339 timestamp in its `pyproject.toml`:

```toml
[tool.uv]
exclude-newer = "2026-07-24T00:00:00Z"
```

The cutoff, rather than a direct-dependency lower bound or the incidental contents of
a lockfile, defines the repository's reproducible resolution horizon. It also creates
a review point before newly uploaded distributions enter the dependency graph,
reducing exposure to fresh supply-chain compromises.

`uv.lock` files are intentionally not committed. Local and CI environments resolve
against the cutoff so that the declared dependency policy, rather than an incidental
lockfile snapshot, remains the source of truth.

Advance the timestamp only as part of an intentional dependency update. Review the
full resolution diff and run the relevant checks before committing the new cutoff.

---

## Workflow

### Adding Dependencies

1. Identify whether the dependency is runtime, optional, or development-only.
2. Check the changelog for breaking changes.
3. For runtime and optional dependencies, add the weakest constraint justified by an
   API or compatibility boundary. For dependency groups, default to no constraint.
4. Document every non-first-party constraint in this file.
5. Set or intentionally advance `exclude-newer`, resolve dependencies, and migrate
   code or configuration for the resulting versions.
6. Run the relevant checks:
   - Runtime: `uv run pytest`
   - Documentation: `uv run --extra docs mkdocs build`
   - Development tools: their CI commands, including `ruff check` and
     `ruff format --check`

### Updating Dependencies

```bash
# First-party (lockstep)
uv run scripts/release/main.py version 0.7.0

# Third-party
# First advance the shared exclude-newer timestamp in every pyproject.toml.
uv sync --upgrade-package <package>
uv sync --upgrade  # all packages
```

Do not retain an obsolete direct-dependency lower bound as a substitute for the
repository's `exclude-newer` policy. If a newly resolved release is incompatible,
either migrate to its API or document the concrete incompatibility before adding a
constraint.
