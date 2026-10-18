# 8. Pre-Commit Hooks

There are a number of pre-commit hooks available to run on files within the repo. Run these once you have code that you'd like to submit in a pull request to make sure they pass. These are meant to enforce style within the code base.

```bash
### Make sure all files are added with `git add` before running pre-commit

# run hooks on all files
uv run pre-commit run --all-files

# run hooks on a specific file
uv run pre-commit run --files <file_name>

```
Note: if you did not install PyRIT using uv, then omit uv run from the above commands

## Preparing Hook Environments with uv

pre-commit builds each Python hook environment with virtualenv and installs the hook with `python -m pip install .`, even when pre-commit itself is started with `uv run`. Running it with [pre-commit-uv](https://github.com/tox-dev/pre-commit-uv) makes it create the environments with `uv venv` and install the hooks with `uv pip install` instead. Hook versions and hook selection still come from `.pre-commit-config.yaml`.

pre-commit-uv is not one of the dev dependencies, so add it to a single command with `--with`. Pin it and uv so the preparation can be repeated, and check that it is active:

```bash
uv run --frozen --no-sync --with pre-commit-uv==4.4.0 --with uv==0.13.1 pre-commit --version
```

The output should end with `(pre-commit-uv=4.4.0, uv=0.13.1)`. If it does not, pre-commit-uv is not active and pre-commit falls back to virtualenv and pip.

### Preparing for an Offline Run

Preparation needs network access and is a separate step from the run. It installs the project environment, which the `ty-check` hook runs in, then clones the hook repositories and creates every hook environment:

```bash
uv sync --frozen --extra all
uv run --frozen --no-sync --with pre-commit-uv==4.4.0 --with uv==0.13.1 pre-commit install-hooks
```

Run the hooks with the same `--with` options. Without pre-commit-uv, pre-commit does not accept the environments that uv created and tries to install them again with pip:

```bash
UV_OFFLINE=1 GIT_ALLOW_PROTOCOL=file uv run --frozen --no-sync --with pre-commit-uv==4.4.0 --with uv==0.13.1 pre-commit run --all-files
```

`UV_OFFLINE=1` makes uv use only its cache, so a package that was not prepared fails the run instead of being downloaded. It does not cover git: if a hook repository was not prepared, pre-commit tries to clone it, and `GIT_ALLOW_PROTOCOL=file` makes that clone fail instead of reaching the network. In PowerShell, set `$env:UV_OFFLINE = "1"` and `$env:GIT_ALLOW_PROTOCOL = "file"` before the command instead, and clear them afterwards with `Remove-Item Env:UV_OFFLINE, Env:GIT_ALLOW_PROTOCOL`, since they stay set for the rest of the session.

The environments are stored in the pre-commit cache (`~/.cache/pre-commit`, or `PRE_COMMIT_HOME` if set) and their packages in the uv cache. To prepare from cold caches, point `PRE_COMMIT_HOME` and `UV_CACHE_DIR` at empty directories and keep the same values for the run.
