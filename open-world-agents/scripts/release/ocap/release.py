"""
This script is used to create a conda environment tarball for the project. The script does the following:
1. Replace all `editable = true` with `editable = false` in all `pyproject.toml` files
2. Run `uv pip install .`
3. Run `conda pack -n owa`
4. Revert all `editable = false` back to `editable = true` in all `pyproject.toml` files

NOTE: `conda-pack` requires the packages to be installed without `--editable` flag.
"""

import subprocess
from pathlib import Path

ENV_NAME = "owa"


def update_pyproject_toml(revert=False):
    for pyproject in Path(".").rglob("pyproject.toml"):
        content = pyproject.read_text()
        if not revert:
            updated_content = content.replace("editable = true", "editable = false")
        else:
            updated_content = content.replace("editable = false", "editable = true")
        pyproject.write_text(updated_content)
        print(f"Updated: {pyproject}")


def install_project():
    subprocess.run(["uv", "pip", "install", "projects/ocap"], check=True)
    print("Installed project dependencies.")


def pack_conda_env():
    subprocess.run(["conda-pack", "-n", ENV_NAME, "--output", "scripts/release/ocap/env.tar.gz"], check=True)
    print("Packed conda environment.")


def main():
    # Step 1: Replace all `editable = true` with `editable = false` in all `pyproject.toml` files
    update_pyproject_toml()

    # Step 2: Run `uv pip install .`
    install_project()

    # Step 3: Run `conda pack -n owa`
    pack_conda_env()

    # Step 4: Revert all `editable = false` back to `editable = true` in all `pyproject.toml` files
    update_pyproject_toml(revert=True)

    print("Process completed successfully!")


if __name__ == "__main__":
    main()
