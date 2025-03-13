`open-world-agents` is a mono-repo which is composed with multiple sub-repository. e.g. `projects/mcap-owa-support, projects/owa-cli, projects/owa-core, projects/owa-env-desktop`.

Each sub-repository is a self-contained repository which may have other sub-repository as dependencies.

Most of subprojects inside `projects` are **python package in itself**; In other words, they are installable by `pip` or [`uv`](https://docs.astral.sh/uv/). Since we're utilizing `uv`, we recommend you to use `uv` as package manager.

We're adopting namespace packages. Most `owa`-related packages, including EnvPlugins, are installed in `owa` namespace, e.g. `owa.core`, `owa.cli`, `owa.env.desktop`. For more detail, see [Packaging namespace packages](https://packaging.python.org/en/latest/guides/packaging-namespace-packages/)

```
open-world-agents/
├── projects/
│   ├── mcap-owa-support
│   ├── owa-core/         
│   ├── owa-cli/
│   ├── owa-env-desktop/
│   ├── owa-env-example/
│   ├── owa-env-gst/
│   └── and also more! e.g. you may contribute owa-env-minecraft!
├── docs/              # Documentation
└── README.md         # Project overview
```