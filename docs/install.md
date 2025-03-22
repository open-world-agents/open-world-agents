# Installation Guide

## Minimal Installation Guide

If you **do not require** efficient screen capture with `owa-env-gst`, installation is simple as:

=== "uv"

    ```sh
    $ git clone https://github.com/open-world-agents/open-world-agents
    $ cd open-world-agents # Ensure you're at project root

    $ uv sync --inexact
    ```

=== "pip"

    ```sh
    $ git clone https://github.com/open-world-agents/open-world-agents
    $ cd open-world-agents # Ensure you're at project root

    # Install core first
    $ pip install -e projects/owa-core

    # Install supporting packages
    $ pip install -e projects/mcap-owa-support
    $ pip install -e projects/owa-desktop
    $ pip install -e projects/owa-env-gst

    # Install CLI
    $ pip install -e projects/owa-cli
    ```

    !!! tip
        When using `pip` instead of `uv`, **the installation order matters** because `pip` can't recognize `[tool.uv.sources]` in `pyproject.toml`.

!!! tip 

    `open-world-agents` is a mono-repo which is composed with multiple sub-repository and most sub-repositories are `pip`-installable python package in itself.

    Only `owa-env-gst`, which utilizes GStreamer to efficiently capture screen deal with media processing, **requires installing `gstreamer` by `conda`.**

    If you are on **Windows OS** and **require screen capture** on the device, follow [Full Installation Guide](#full-installation-guide) instead.

---

## Full Installation Guide

If you **do require** efficient screen capture with `owa-env-gst`, follow this guide.

### Setup Virtual Environment (1/3)

Before installation, we recommend setting up a virtual environment.

=== "conda/mamba"

    1. Follow the [miniforge installation guide](https://github.com/conda-forge/miniforge?tab=readme-ov-file#install) to install `conda` and `mamba`. `mamba` is just a faster `conda`. If you've already installed `conda`, you may skip this step.

    2. Create & activate your virtual environment:
       ```sh
       $ conda create -n owa python=3.11 -y
       $ conda activate owa
       ```

    3. (Optional) For Windows users who need desktop recorder:
       ```sh
       $ mamba env update --name owa --file projects/owa-env-gst/environment.yml
       ```

!!! tip

    You can use other virtual environment tools, but to fully utilize `owa-env-gst`, you must install GStreamer with `conda/mamba`.
    
    Note: GStreamer is only needed if you plan to capture screens.

### Setup `uv` (2/3)

We recommend setting up `uv` next:

1. Follow the [uv installation guide](https://docs.astral.sh/uv/getting-started/installation/) or simply run `pip install uv` in your activated environment.

2. Install [`virtual-uv`](https://github.com/open-world-agents/vuv) package:
   ```sh
   $ pip install virtual-uv
   ```

!!! tip

    Always activate your virtual environment before running any `vuv` commands.

### Installation (3/3)

After all setup, installation is simple as:

=== "vuv"

    ```sh
    $ git clone https://github.com/open-world-agents/open-world-agents
    $ cd open-world-agents # Ensure you're at project root

    $ vuv install
    ```

!!! tip

    Always activate your virtual environment before running any `vuv` commands.

## Install from PyPI & conda-forge (WIP)

- PyPI packages:
    - `owa-core`: Contains only the core logic to manage OWA's EnvPlugin
    - `owa`: Contains several base EnvPlugins along with `owa-core` (requires separate GStreamer installation)

- Conda packages:
    - `owa`: Complete package with EnvPlugins and core functionality