import os
import subprocess
import sys
import tempfile


def main():
    print("Starting owl command restoration...")

    with tempfile.TemporaryDirectory() as temp_dir:
        # Create pyproject.toml file
        with open(os.path.join(temp_dir, "pyproject.toml"), "w") as f:
            f.write("""
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
bypass-selection = true

[project]
name = "owl-wrapper"
version = "0.1.0"
description = "Simple wrapper to restore owl command"

[project.scripts]
owl = "owa.cli:app"
""")

        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", temp_dir], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
            )
            print("Owl command restoration completed successfully.")
        except Exception as e:
            print(f"Error restoring owl command: {e}")


if __name__ == "__main__":
    main()
