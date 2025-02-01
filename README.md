# Installation

To deal with lacking feature in `uv` compared to `poetry`, following monkey-patch is needed to enable `uv install`. see [this issue](https://github.com/astral-sh/uv/issues/11152)

```
uv() {
  if [ "$1" = "install" ]; then
    shift
    no_root=false
    args=()
    for arg in "$@"; do
      if [ "$arg" = "--no-root" ]; then
        no_root=true
      else
        args+=("$arg")
      fi
    done
    if uv export -o requirements.txt "${args[@]}" > /dev/null; then
      uv pip install -r requirements.txt
      rm -f requirements.txt
      if [ "$no_root" = false ]; then
        uv pip install -e .
      fi
    else
      echo "Error: 'uv export' failed." >&2
      return 1
    fi
  else
    command uv "$@"
  fi
}
```