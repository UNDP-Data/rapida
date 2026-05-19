# ==========================================
# STAGE 1: Tippecanoe C++ Builder
# ==========================================
FROM ubuntu:22.04 AS tippecanoe-builder
RUN apt-get update && apt-get install -y build-essential libsqlite3-dev zlib1g-dev git && \
    git clone https://github.com/felt/tippecanoe && \
    cd tippecanoe && make -j$(nproc)

# ==========================================
# STAGE 2: Python Environment Builder (The "Fat" Stage)
# ==========================================
FROM ghcr.io/osgeo/gdal:ubuntu-small-3.10.0 AS python-builder

ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Install heavy C++ compilers required for building python wheels
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      python3-pip python3-dev gcc g++ git cmake libgeos-dev

WORKDIR /rapida

# 1. Define the venv location ONCE and add it to the system PATH
ENV VIRTUAL_ENV=/rapida/.venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

COPY pyproject.toml ./

# 2. Run clean, human-readable commands
RUN uv venv && \
    uv pip install --no-cache "GDAL==$(gdal-config --version)" && \
    uv pip install --no-cache -r pyproject.toml

# 3. Copy the rest of your local files (the `rapida` folder, README, etc.)
COPY . .

# 4. Install the app itself (the dot means "install from the current directory")
RUN uv pip install --no-cache .

# ==========================================
# STAGE 3: Production Runtime (The "Skinny" Stage)
# ==========================================
FROM ghcr.io/osgeo/gdal:ubuntu-small-3.10.0 AS prod

ENV DEBIAN_FRONTEND=noninteractive
ENV PLAYWRIGHT_BROWSERS_PATH=0
ENV PATH="/rapida/.venv/bin:$PATH"

# Copy compiled Tippecanoe binaries
COPY --from=tippecanoe-builder /tippecanoe/tippecanoe* /usr/local/bin/
COPY --from=tippecanoe-builder /tippecanoe/tile-join /usr/local/bin/

# Copy the fully compiled Python environment (Leaving all C++ compilers behind!)
COPY --from=python-builder --chown=1000:1000 /rapida/.venv /rapida/.venv

# Install ONLY runtime dependencies (Playwright needs system UI libraries for Chromium)
# We do this in one layer and immediately purge the apt cache
RUN apt-get update && \
    playwright install chromium --with-deps && \
    apt-get autoremove -y && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /rapida
USER 1000

# The default command (can be overridden by rapida-jupyter later)
CMD ["rapida"]