# Dependency Version Policy

**Last Updated**: 2026-07-24 | **Repository Version**: 0.6.6

## Rules

- Runtime constraints are part of the published compatibility contract. Foundation
  libraries should keep them minimal; leaf packages may be stricter only when needed.
- Extras are also published metadata. Leave them unconstrained unless an optional
  feature needs a specific API or must exclude a known-bad release.
- Dependency groups are local development inputs and should be unconstrained by
  default.
- First-party runtime packages use exact versions for lockstep releases.
- Every other constraint requires a concrete API, compatibility, or security reason
  documented below. The version current when a dependency was added is not a reason.
- Tool upgrades do not justify dependency constraints. Keep Ruff's selected rules
  explicit in `ruff.toml`; fix code when an already-selected rule changes.

Accordingly, the root `docs` extra and `dev` group have no direct version constraints.

## Constraint Rationale

| Package | Constraint | Rationale |
|---------|------------|-----------|
| `pydantic` | `>=2.0` | Pydantic v2 API |
| `numpy` | `>=2.0` | NumPy 2.x API and ABI |
| `av` | `>=15.0` | FFmpeg 7 support |
| `pillow` | `>=9.4.0` | `PIL.Image.ExifTags` |
| `pyyaml` | `>=6.0` | Security fixes and Python 3.11 support |
| `packaging` | `>=20.0` | PEP 440 parsing |
| `requests` | `>=2.32.2` | Security fixes and compatibility |
| `torch` | `>=2.0` | PyTorch 2.x features |
| `datasets` | `>=4.0` | Datasets 4.x API |
| `transformers` | `>=4.52.1` | InternVL support |
| `huggingface_hub` | `>=0.30.0` | Transformers compatibility |
| `jsonargparse[signatures]` | `>=4.41.0` | `KEYWORD_ONLY` handling fix |
| `line-profiler` | `>=4.1.0` | Global `@line_profiler.profile` |
| `mcap` | `>=1.0.0` | Stable MCAP API |
| `mediaref` | `>=1.1.1` | Reused PyAV reformatter for faster batch conversion |
| `typer` | `>=0.20.0` | Required CLI behavior and fixes |
| `rich` | `>=14.1.0` | Required CLI behavior and fixes |
| `lazyregistry` | `>=0.3.0` | Stable API used by `owa-core` |

### Platform-specific

| Package | Constraint | Platform | Rationale |
|---------|------------|----------|-----------|
| `pywin32` | `>=307` | Windows | Python 3.11 support |
| `pyobjc-framework-*` | `>=10.1` | macOS | macOS 11+ compatibility |
| `evdev` | `<1.9.2` | Linux | 1.9.2 build failure |
| `pynput` | `>=1.8.0` | All | Required stability fixes |

Stable APIs without constraints include `loguru`, `tqdm`, `orjson`,
`annotated-types`, `jinja2`, `python-dotenv`, `diskcache`, `griffe`, `plotext`,
`webdataset`, `pygobject-stubs`, `pygetwindow`, `bettercam`, `pydantic-settings`,
and `python-multipart`.

Special case: Conda `pygobject` remains exactly `3.50.0` because changing it breaks
plugin detection.

## Resolution Cutoff

The root, `owa-cli`, `owa-data`, and `video-decoding-server` replace their former
lockfiles with:

```toml
[tool.uv]
exclude-newer = "2026-07-24T00:00:00Z"
```

This reviewed upload horizon provides repeatable resolution and delays exposure to
new supply-chain compromises. These environments do not commit `uv.lock`; advance
the cutoff only during an intentional dependency update and review the full result.

Library subprojects that already ignore `uv.lock` do not set `exclude-newer`. Their
downstream application or invoking development environment owns resolution policy.

## Workflow

1. Classify the dependency as runtime, optional, or development-only.
2. Add the weakest justified constraint; default extras and groups to unconstrained.
3. Document each non-first-party constraint here.
4. Advance `exclude-newer` only in cutoff-managed environments.
5. Resolve, perform required migrations, and run the relevant tests, documentation
   build, and development-tool checks.

```bash
# First-party lockstep release
uv run scripts/release/main.py version 0.7.0

# Third-party update after advancing the applicable cutoff
uv sync --upgrade-package <package>
uv sync --upgrade
```
