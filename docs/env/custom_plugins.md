# How to write your own EnvPlugin

You may write & contribute your own EnvPlugin.

1. Copy & Paste [owa-env-example](https://github.com/open-world-agents/open-world-agents/tree/main/projects/owa-env-example) directory. This directory contains following:
    ```sh
    owa-env-example
    ├── owa_env_example
    │   ├── example_callable.py
    │   ├── example_listener.py
    │   ├── example_runnable.py
    │   └── __init__.py
    ├── pyproject.toml
    ├── README.md
    ├── tests
    │   └── test_print.py
    └── uv.lock
    ```
2. Rename `owa-env-example` to your own EnvPlugin's name.
3. Write your own code in source folder.
4. Make sure your repository contains all dependencies. We recommend you to use `uv` as package manager.
5. Make a PR, following [Contributing Guide](../contributing.md)