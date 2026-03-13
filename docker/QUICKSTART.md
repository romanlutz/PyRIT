# PyRIT Docker - Quick Start Guide

Docker container for PyRIT with support for both **Jupyter Notebook** and **GUI** modes.

## Prerequisites
- Docker installed and running
- `~/.pyrit/.env` with your API keys and Azure service principal credentials
- `~/.pyrit/.pyrit_conf` with your configuration (operator, operation, initializers)
- Optionally, `~/.pyrit/.env.local` for additional environment overrides

## Azure Authentication in Docker

Inside a container there is no interactive `az login`. Instead, use a
**service principal** by adding these variables to your `~/.pyrit/.env`:

```bash
AZURE_TENANT_ID=<your-tenant-id>
AZURE_CLIENT_ID=<your-client-id>
AZURE_CLIENT_SECRET=<your-client-secret>
```

The Azure SDK's `DefaultAzureCredential` picks these up automatically via
`EnvironmentCredential` and refreshes tokens without any manual intervention.

To create a service principal:
```bash
az ad sp create-for-rbac --name pyrit-docker --role "Cognitive Services OpenAI User" \
    --scopes /subscriptions/<sub-id>/resourceGroups/<rg>/providers/Microsoft.CognitiveServices/accounts/<account>
```

## Quick Start

### 1. Build the Image

Build from local source (includes frontend):
```bash
python docker/build_pyrit_docker.py --source local
```

Build from PyPI version:
```bash
python docker/build_pyrit_docker.py --source pypi --version 0.10.0
```

Rebuild base image (when devcontainer changes):
```bash
python docker/build_pyrit_docker.py --source local --rebuild-base
```

> **Note:** The build script automatically builds the devcontainer base image if needed.
> The base image is cached and reused for faster subsequent builds.

### 2. Run PyRIT

Jupyter mode (port 8888):
```bash
python docker/run_pyrit_docker.py jupyter
```

GUI mode (port 8000):
```bash
python docker/run_pyrit_docker.py gui
```

The run script automatically mounts these files from `~/.pyrit/`:
- `.env` — API keys and service principal credentials (required)
- `.env.local` — Additional environment overrides (optional)
- `.pyrit_conf` — PyRIT configuration: operator, operation, initializers (optional)

## Image Tags

Images are tagged with version information:
- PyPI: `pyrit:0.10.0`, `pyrit:latest`
- Local (clean): `pyrit:<full-commit-hash>`, `pyrit:latest`
- Local (modified): `pyrit:<full-commit-hash>-modified`, `pyrit:latest`

Run specific tag:
```bash
python docker/run_pyrit_docker.py gui --tag abc1234def5678
```

## Version Display

The GUI shows PyRIT version in a tooltip on the logo:
- PyPI builds: `0.10.0`
- Local builds: `abc1234def5678` or `abc1234def5678 + local changes`

## Docker Compose

Use profiles to run specific modes:

```bash
# Jupyter mode
docker-compose --profile jupyter up

# GUI mode
docker-compose --profile gui up
```

## Troubleshooting

**Image not found**: Run `python docker/build_pyrit_docker.py --source local` first

**.env missing**: Create `.env` file at `~/.pyrit/.env` with your API keys

**Azure auth fails in container**: Add `AZURE_TENANT_ID`, `AZURE_CLIENT_ID`, and
`AZURE_CLIENT_SECRET` to your `.env` file (see Azure Authentication section above)

**GUI frontend missing**: Build with `--source local` (PyPI builds before GUI release won't work)

For complete documentation, see [docker/README.md](./README.md)
