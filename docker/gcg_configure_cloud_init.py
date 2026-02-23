# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Configure the GCG cloud-init template by replacing placeholders with actual values.

Usage:
    python gcg_configure_cloud_init.py \
        --storage-account myaccount \
        --storage-key "base64key==" \
        --hf-token "hf_abc123" \
        --output gcg-cloud-init.sh

Values can also be provided via environment variables:
    STORAGE_ACCOUNT, STORAGE_KEY, CONTAINER, HF_TOKEN, PYRIT_BRANCH
"""

import argparse
import os
import sys
from pathlib import Path

TEMPLATE_FILE = Path(__file__).parent / "gcg_cloud_init_template.sh"  # both files live in docker/

PLACEHOLDERS = {
    "{{STORAGE_ACCOUNT}}": "storage_account",
    "{{STORAGE_KEY}}": "storage_key",
    "{{CONTAINER}}": "container",
    "{{HF_TOKEN}}": "hf_token",
    "{{PYRIT_REPO}}": "pyrit_repo",
    "{{PYRIT_BRANCH}}": "pyrit_branch",
}

ENV_VARS = {
    "storage_account": "STORAGE_ACCOUNT",
    "storage_key": "STORAGE_KEY",
    "container": "CONTAINER",
    "hf_token": "HF_TOKEN",
    "pyrit_repo": "PYRIT_REPO",
    "pyrit_branch": "PYRIT_BRANCH",
}

DEFAULTS = {
    "container": "gcg-results",
    "pyrit_repo": "https://github.com/Azure/PyRIT.git",
    "pyrit_branch": "main",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Configure the GCG cloud-init template with actual values.")
    parser.add_argument("--storage-account", help="Azure Storage account name")
    parser.add_argument("--storage-key", help="Azure Storage account key")
    parser.add_argument("--container", default=None, help="Blob container name (default: gcg-results)")
    parser.add_argument("--hf-token", help="HuggingFace API token")
    parser.add_argument("--pyrit-repo", default=None, help="PyRIT git repo URL (default: Azure/PyRIT)")
    parser.add_argument("--pyrit-branch", default=None, help="PyRIT git branch to install (default: main)")
    parser.add_argument(
        "--template",
        type=Path,
        default=TEMPLATE_FILE,
        help=f"Path to the template file (default: {TEMPLATE_FILE})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to write the configured cloud-init script",
    )
    args = parser.parse_args()

    # Resolve values: CLI arg > environment variable > default
    values = {}
    for arg_name, env_var in ENV_VARS.items():
        cli_value = getattr(args, arg_name, None)
        env_value = os.environ.get(env_var)
        default_value = DEFAULTS.get(arg_name)
        values[arg_name] = cli_value or env_value or default_value

    # Validate required values
    missing = [name for name in ("storage_account", "storage_key", "hf_token") if not values.get(name)]
    if missing:
        print(f"Error: Missing required values: {', '.join(missing)}", file=sys.stderr)
        print("Provide them via CLI arguments or environment variables.", file=sys.stderr)
        sys.exit(1)

    # Read template
    if not args.template.exists():
        print(f"Error: Template file not found: {args.template}", file=sys.stderr)
        sys.exit(1)

    content = args.template.read_text(encoding="utf-8")

    # Replace placeholders
    for placeholder, arg_name in PLACEHOLDERS.items():
        value = values[arg_name]
        if value:
            content = content.replace(placeholder, value)

    # Write output
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(content, encoding="utf-8")
    print(f"Configured cloud-init written to: {args.output}")


if __name__ == "__main__":
    main()
