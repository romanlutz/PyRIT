# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import asyncio
import io
import json
import logging
import zipfile

import requests

from pyrit.datasets.seed_datasets.remote.remote_dataset_loader import (
    _RemoteDatasetLoader,
)
from pyrit.models import SeedDataset, SeedPrompt

logger = logging.getLogger(__name__)


class _MICDataset(_RemoteDatasetLoader):
    """
    Loader for the SALT-NLP Moral Integrity Corpus (MIC) dataset.

    This dataset contains conversations between humans and chatbots
    labeled with moral categories like loyalty, care, fairness,
    authority, sanctity and liberty. After deduplication on the
    question field, the dataset yields tens of thousands of unique
    moral integrity prompts.

    Reference: [@ziems2022mic]
    HuggingFace: https://huggingface.co/datasets/SALT-NLP/MIC

    Warning: Due to the nature of these prompts, consult your legal
    department before testing them with LLMs.
    """

    HF_DATASET_NAME = "SALT-NLP/MIC"
    harm_categories = {"care", "fairness", "loyalty", "authority", "sanctity", "liberty"}
    modalities = ["text"]
    size = "huge"
    tags = ["moral", "ethics", "dialogue"]
    VALID_SPLITS = ["train", "dev", "test"]

    def __init__(self) -> None:
        """Initialize the MIC dataset loader."""
        self.source = "https://huggingface.co/datasets/SALT-NLP/MIC/resolve/main/MIC.zip"

    @property
    def dataset_name(self) -> str:
        """Return the dataset name."""
        return "moral_integrity_corpus"

    async def fetch_dataset_async(self) -> SeedDataset:
        """
        Fetch the MIC dataset and return as SeedDataset.

        Returns:
            SeedDataset: A SeedDataset containing MIC prompts.

        Raises:
            ValueError: If the dataset is empty after loading.
        """
        logger.info("Downloading SALT-NLP MIC dataset...")

        def _download_and_parse() -> list[SeedPrompt]:
            response = requests.get(self.source)
            response.raise_for_status()

            seed_prompts = []
            seen_questions: set = set()

            with zipfile.ZipFile(io.BytesIO(response.content)) as zip_file:
                for split in self.VALID_SPLITS:
                    filename = f"MIC/{split}.jsonl"
                    with zip_file.open(filename) as f:
                        for line in f:
                            row = json.loads(line)
                            question = row.get("Q", "").strip()

                            if not question:
                                continue

                            if question in seen_questions:
                                continue
                            seen_questions.add(question)

                            moral = row.get("moral")
                            if isinstance(moral, str):
                                categories = [m.strip() for m in moral.split("|") if m.strip()]
                            else:
                                categories = []

                            seed_prompts.append(
                                SeedPrompt(
                                    value=question,
                                    data_type="text",
                                    dataset_name=self.dataset_name,
                                    source=self.source,
                                    harm_categories=categories,
                                    authors=["Caleb Ziems", "Jane Yu", "Yi-Chia Wang", "Alon Halevy", "Diyi Yang"],
                                )
                            )

            return seed_prompts

        seed_prompts = await asyncio.to_thread(_download_and_parse)

        if not seed_prompts:
            raise ValueError("SeedDataset cannot be empty.")

        logger.info(f"Successfully loaded {len(seed_prompts)} prompts from MIC dataset")

        return SeedDataset(seeds=seed_prompts, dataset_name=self.dataset_name)