# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Local pytest configuration for the Multimodal PGD integration tests."""

from __future__ import annotations


def pytest_configure(config) -> None:
    config.addinivalue_line(
        "markers",
        "gpu: marks a test that requires a CUDA GPU (real-VLM smoke tests). Collected "
        "only where a GPU is available; otherwise skipped at runtime.",
    )
