# How to write your own EnvPlugin

You may write & contribute your own EnvPlugin.

1. Copy & Paste [owa-env-example](https://github.com/open-world-agents/open-world-agents/tree/main/projects/owa-env-example) directory. This directory contains following:
    ```sh
    owa-env-example
    ├── owa/env/example
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
3. Write your own code in the specific source folder.
    - **Important**: To maintain the [namespace package](https://packaging.python.org/en/latest/guides/packaging-namespace-packages/) structure, all source files must only be written inside the `owa/env/example` folder.
    - **What NOT to do**: Don't place source files in paths between `owa` and `owa/env/example` (e.g., `owa/some_file.py` or `owa/env/some_file.py`).
    - **Correct structure**:
        ```
        owa
        └── env
            └── example
                ├── your_code.py
                ├── your_module.py
                └── __init__.py
        ```
4. Make sure your repository contains all dependencies. We recommend you to use `uv` as package manager.
5. Make a PR, following [Contributing Guide](../contributing.md)




