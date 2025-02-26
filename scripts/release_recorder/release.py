import subprocess
from pathlib import Path

ENV_NAME = "owa_deploy"


# Step 1: Replace all `editable = true` with `editable = false` in all `pyproject.toml` files
def update_pyproject_toml(revert=False):
    for pyproject in Path(".").rglob("pyproject.toml"):
        content = pyproject.read_text()
        if not revert:
            updated_content = content.replace("editable = true", "editable = false")
        else:
            updated_content = content.replace("editable = false", "editable = true")
        pyproject.write_text(updated_content)
        print(f"Updated: {pyproject}")


# Step 2: Run `uv pip install .`
def install_project():
    subprocess.run(["uv", "pip", "install", "projects/data_collection"], check=True)
    print("Installed project dependencies.")


# Step 3: Run `conda pack -n owa`
def pack_conda_env():
    subprocess.run(["conda", "pack", "-n", ENV_NAME, "--output", "env.tar.gz"], check=True)
    print("Packed conda environment.")


def main():
    update_pyproject_toml()
    install_project()
    pack_conda_env()
    update_pyproject_toml(revert=True)
    print("Process completed successfully!")


if __name__ == "__main__":
    main()
