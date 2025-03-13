`open-world-agents` is a mono-repo which is composed with multiple sub-repository.

Each sub-repository is a self-contained repository which may have other sub-repository as dependencies.

We're adopting namespace packages. For more detail, see https://packaging.python.org/en/latest/guides/packaging-namespace-packages/

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