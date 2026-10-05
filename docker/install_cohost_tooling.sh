#!/bin/bash
set -euo pipefail

if [ "$(uname -s)" != "Linux" ] || [ "$(uname -m)" != "x86_64" ]; then
    echo "ERROR: cohost tooling requires Linux x86_64" >&2
    exit 1
fi

prefix="${1:-/opt/pyrit-cohost}"
if [ -e "$prefix" ]; then
    echo "ERROR: cohost tooling requires a new installation directory" >&2
    exit 1
fi

archive=$(mktemp)
trap 'rm -f "$archive"' EXIT
curl --fail --silent --show-error --location --connect-timeout 15 --max-time 180 \
    https://github.com/astral-sh/uv/releases/download/0.8.22/uv-x86_64-unknown-linux-gnu.tar.gz \
    --output "$archive"
if ! echo "741ff1f5742c5a4a25d2f829e8395355e43f7a5ae2ebc6368e9ae2df0efb69cf  $archive" | sha256sum --check --status; then
    echo "ERROR: pinned uv archive checksum mismatch" >&2
    exit 1
fi
mkdir -p "$prefix/bin"
tar --extract --gzip --file="$archive" --directory="$prefix/bin" --strip-components=1 \
    uv-x86_64-unknown-linux-gnu/uv
"$prefix/bin/uv" --version | grep -Eq '^uv 0\.8\.22( |$)'

# The checksum-pinned uv release also pins the standalone Python archive and its checksum.
export UV_PYTHON_INSTALL_DIR="$prefix/python"
"$prefix/bin/uv" python install --no-config --no-bin cpython-3.12.11-linux-x86_64-gnu
ln -s ../python/cpython-3.12.11-linux-x86_64-gnu/bin/python3.12 "$prefix/bin/python3.12"
"$prefix/bin/python3.12" -I -c \
    'import ctypes, sqlite3, ssl, sys; assert sys.version_info[:3] == (3, 12, 11); print(sys.version)'
