# Build only from the approved, already-local signed Ubuntu/Python/GHCP image ID.
# The caller must use docker build --pull=false --network=none and a private context
# containing this file, pyproject.toml, uv.lock, and the verified wheelhouse.
ARG BASE_IMAGE=pyrit-ghcp-python-uv:1.0.88-3.12-0.11.8
FROM ${BASE_IMAGE}

ARG UV_SHA256=646adf5cf12ba17d1a41fa77c8dd6496f73651dcfeeed6b5f4ec019b36bc7153
ARG GHCP_SHA256=0059754cf78c3f3bf2c9d4564dfa7e9e25f3a3f8f411f2f0cdad9363f5662748

USER root
COPY pyproject.toml uv.lock check_wheels.py /opt/pyrit/guest-deps/
COPY wheelhouse/ /opt/pyrit/wheelhouse/

RUN printf '%s  %s\n' "$UV_SHA256" /opt/pyrit/uv | sha256sum --check --status \
    && printf '%s  %s\n' "$GHCP_SHA256" /opt/pyrit/copilot | sha256sum --check --status \
    && /usr/bin/python3.12 /opt/pyrit/guest-deps/check_wheels.py before \
    && mkdir /opt/pyrit/guest-local \
    && cp /opt/pyrit/guest-deps/pyproject.toml /opt/pyrit/guest-deps/uv.lock /opt/pyrit/guest-local/ \
    && UV_CACHE_DIR=/opt/pyrit/uv-cache /opt/pyrit/uv lock \
        --directory /opt/pyrit/guest-local \
        --offline --no-index --find-links /opt/pyrit/wheelhouse --upgrade \
    && /usr/bin/python3.12 /opt/pyrit/guest-deps/check_wheels.py after \
    && sha256sum /opt/pyrit/guest-deps/uv.lock /opt/pyrit/guest-local/uv.lock \
    && UV_CACHE_DIR=/opt/pyrit/uv-cache UV_LINK_MODE=copy /opt/pyrit/uv sync \
        --directory /opt/pyrit/guest-local \
        --frozen --offline --no-index --find-links /opt/pyrit/wheelhouse \
        --python /usr/bin/python3.12 \
    && /opt/pyrit/guest-local/.venv/bin/python -c 'import copilot, httpx; print(httpx.__version__)' \
    && rm -rf /opt/pyrit/wheelhouse /opt/pyrit/uv-cache

ENV PATH="/opt/pyrit/guest-local/.venv/bin:${PATH}" \
    COPILOT_CLI_PATH="/opt/pyrit/copilot" \
    COPILOT_SKIP_CLI_DOWNLOAD="1" \
    COPILOT_HOME="/home/bridge/copilot-state"
USER 10001:10001
WORKDIR /workspace
