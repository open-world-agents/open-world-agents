# How to run Devcontainer in VS Code

1. If base docker image is not present, follow the instructions in [../docker/README.md](../docker/README.md) to build it.

2. Build devcontainer image using `./docker-build.sh`

3. Adjust settings in `devcontainer.json` if needed (e.g. mount, runArgs, etc.)

4. Open in VS Code with devcontainer extension