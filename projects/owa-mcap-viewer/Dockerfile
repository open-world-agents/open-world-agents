# This dockerfile is for https://huggingface.co/spaces/open-world-agents/visualize_dataset

# Configure image
ARG PYTHON_VERSION=3.11

FROM python:${PYTHON_VERSION}-slim
ARG PYTHON_VERSION
ARG DEBIAN_FRONTEND=noninteractive

# Install apt dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake git wget \
    libglib2.0-0 libgl1-mesa-glx libegl1-mesa ffmpeg \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Setup uv & vuv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install virtual-uv --system

# Create virtual environment
RUN uv venv /opt/venv
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN echo "source /opt/venv/bin/activate" >> /root/.bashrc

RUN useradd -m -u 1000 user

# Install OWA
ARG CACHE_BUST=1
RUN git clone --depth 1 https://github.com/open-world-agents/open-world-agents /owa
WORKDIR /owa
RUN --mount=type=cache,target=/home/user/.cache/uv \
    vuv install --frozen

WORKDIR /owa/projects/owa-mcap-viewer
RUN --mount=type=cache,target=/home/user/.cache/uv \
    # --mount=type=bind,source=uv.lock,target=uv.lock \
    # --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    vuv install --frozen

# Prepare example datasets
RUN --mount=type=cache,target=/home/user/.cache/uv \
    vuv pip install huggingface-hub
RUN huggingface-cli download open-world-agents/example_dataset --repo-type dataset --local-dir /data
ENV EXPORT_PATH=/data

RUN chown -R user:user /owa /opt/venv /data
COPY --chown=user . /owa/projects/owa-mcap-viewer
CMD ["uvicorn", "owa_viewer:app", "--host", "0.0.0.0", "--port", "7860"]