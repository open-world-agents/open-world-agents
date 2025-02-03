# Installation

1. Install `uv`, following [installation guide](https://docs.astral.sh/uv/getting-started/installation/)

2. Setup `uv install`. This stage is optional.

- To use `uv install`: To deal with lacking feature in `uv` compared to `poetry`, following monkey-patch is needed to enable `uv install`. see [this issue](https://github.com/astral-sh/uv/issues/11152)
- You may use `uv sync --inexact` instead of `uv install`, but be careful not to omit `--inexact` arg. Without this, existing package of your virtual environment may be deleted accidentally.

```
uv() {
  if [ "$1" = "install" ]; then
    shift
    if uv export -o requirements.txt "$@" > /dev/null; then
      uv pip install -r requirements.txt
      rm -f requirements.txt
    else
      echo "Error: 'uv export' failed." >&2
      return 1
    fi
  else
    command uv "$@"
  fi
}
```

3.

set your own `UV_PROJECT_ENVIRONMENT` variable as **absolute path**. 

We recommend you to setup `.env` file as followed.
```
UV_PROJECT_ENVIRONMENT=(absolute path of open-world-agents cloned)
```

4.

At project root(under `open-world-agents`), run `uv install --group dev` or `uv sync --inexact --group dev`