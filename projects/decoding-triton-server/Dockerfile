# Use the NVIDIA Triton Inference Server base image
FROM nvcr.io/nvidia/tritonserver:25.02-py3

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

WORKDIR /app

# Install dependencies: THIS CREATES NEW VENV wtf 
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    vuv sync --frozen --no-install-project

# or  install the dependencies manually
# ADD . /app
# RUN --mount=type=cache,target=/root/.cache/uv \
#     --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
#     vuv pip install --system -r pyproject.toml


# ======================= Copy the model repository =======================

# Copy the model repository (assuming you have it in the `models` directory)
COPY model_repository /app/models

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Create wrapper script
# RUN echo '#!/bin/bash\n\
# source /app/.venv/bin/activate\n\
# exec tritonserver --model-repository=/app/models "$@"' > /entrypoint.sh && \
#     chmod +x /entrypoint.sh

ENTRYPOINT ["tritonserver", "--model-repository=/app/models"]