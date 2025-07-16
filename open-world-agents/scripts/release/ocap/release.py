import subprocess

ENV_NAME = "owa"


def main():
    # Step 1: Run `uv pip install . --no-sources`
    # NOTE: `--no-sources` is needed to prevent editable install. See https://github.com/astral-sh/uv/issues/14609
    subprocess.run(["uv", "pip", "install", "projects/ocap", "--no-sources"], check=True)
    print("Installed project dependencies.")

    # Step 2: Run `conda pack -n owa`
    # NOTE: `conda-pack` requires the packages to be installed without `--editable` flag.
    subprocess.run(["conda-pack", "-n", ENV_NAME, "--output", "scripts/release/ocap/env.tar.gz"], check=True)
    print("Packed conda environment.")

    print("Process completed successfully!")


if __name__ == "__main__":
    main()
