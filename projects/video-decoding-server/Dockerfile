FROM nvcr.io/nvidia/tritonserver:25.06-py3

# Install the uv package manager
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# To use the system Python environment by default, set the UV_SYSTEM_PYTHON variable:
# ENV UV_SYSTEM_PYTHON=1

# ======================= Setup the environment =======================

RUN uv venv /opt/venv
# Use the virtual environment automatically
ENV VIRTUAL_ENV=/opt/venv
# Place entry points in the environment at the front of the path
ENV PATH="/opt/venv/bin:$PATH"

RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install virtual-uv

# ======================= Install the dependencies =======================

# Install ffmpeg
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg

# Install dependencies
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    vuv install --frozen --no-install-project

# ======================= Copy the model repository =======================

WORKDIR /app

# Copy the model repository (assuming you have it in the `models` directory)
COPY model_repository /app/models

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

ENTRYPOINT ["tritonserver", "--model-repository=/app/models"]