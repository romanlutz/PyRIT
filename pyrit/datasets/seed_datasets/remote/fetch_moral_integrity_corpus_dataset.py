# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import io
import json
import logging
import os
import zipfile
from typing import Optional

import requests

from pyrit.datasets.seed_datasets.remote.remote_dataset_loader import (
    _RemoteDatasetLoader,
)
from pyrit.models import SeedDataset, SeedPrompt

logger = logging.getLogger(__name__)


class _MICDataset(_RemoteDatasetLoader):
    """
    Loader for the SALT-NLP Moral Integrity Corpus (MIC) dataset.

    This dataset contains 113,817 conversations between humans and
    chatbots labeled with moral categories like loyalty, care,
    fairness, authority and sanctity.

    Reference: [@ziems2022mic]
    HuggingFace: https://huggingface.co/datasets/SALT-NLP/MIC

    Warning: Due to the nature of these prompts, consult your legal
    department before testing them with LLMs.
    """

    HF_DATASET_NAME = "SALT-NLP/MIC"
    VALID_SPLITS = ["train", "dev", "test"]
    harm_categories = {"care", "fairness", "loyalty", "authority", "sanctity"}
    modalities = ["text"]
    size = "huge"
    tags = ["moral", "ethics", "dialogue"]

    def __init__(
        self,
        *,
        token: Optional[str] = None,
    ) -> None:
        """
        Initialize the MIC dataset loader.

        Args:
            token: HuggingFace authentication token. If not provided,
                   reads from HUGGINGFACE_TOKEN env var.
        """
        self.source = "https://huggingface.co/datasets/SALT-NLP/MIC/resolve/main/MIC.zip"
        self.token = token if token is not None else os.environ.get("HUGGINGFACE_TOKEN")

    @property
    def dataset_name(self) -> str:
        """Return the dataset name."""
        return "moral_integrity_corpus"

    async def fetch_dataset_async(self, *, cache: bool = True) -> SeedDataset:
        """
        Fetch the MIC dataset and return as SeedDataset.

        Args:
            cache: Whether to cache the fetched dataset. Defaults to True.

        Returns:
            SeedDataset: A SeedDataset containing MIC prompts.

        Raises:
            ValueError: If the dataset is empty after loading.
        """
        logger.info("Downloading SALT-NLP MIC dataset...")

        response = requests.get(self.source)
        response.raise_for_status()

        seed_prompts = []

        with zipfile.ZipFile(io.BytesIO(response.content)) as zip_file:
            for split in self.VALID_SPLITS:
                filename = f"MIC/{split}.jsonl"
                with zip_file.open(filename) as f:
                    for line in f:
                        row = json.loads(line)
                        question = row.get("Q", "").strip()
                        if question:
                            seed_prompts.append(
                                SeedPrompt(
                                    value=question,
                                    data_type="text",
                                    dataset_name=self.dataset_name,
                                    source=self.source,
                                )
                            )

        if not seed_prompts:
            raise ValueError("SeedDataset cannot be empty.")

        logger.info(f"Successfully loaded {len(seed_prompts)} prompts from MIC dataset")

        return SeedDataset(seeds=seed_prompts, dataset_name=self.dataset_name)