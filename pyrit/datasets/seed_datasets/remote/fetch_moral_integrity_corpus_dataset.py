# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import io
import json
import logging
import zipfile

import requests

from pyrit.models import SeedDataset, SeedPrompt

logger = logging.getLogger(__name__)


def fetch_moral_integrity_corpus_dataset(
    *,
    cache: bool = True,
) -> SeedDataset:
    """
    Fetch the SALT-NLP Moral Integrity Corpus (MIC) dataset.

    This dataset contains prompts used to capture moral assumptions in LLMs.
    It contains 113,000+ examples across various moral categories.

    Reference: https://aclanthology.org/2022.acl-long.261/
    HuggingFace: https://huggingface.co/datasets/SALT-NLP/MIC

    Warning: Due to the nature of these prompts, consult your legal department
    before testing them with LLMs.
    """

    source = "https://huggingface.co/datasets/SALT-NLP/MIC/resolve/main/MIC.zip"

    logger.info("Downloading SALT-NLP MIC dataset...")

    response = requests.get(source)
    response.raise_for_status()

    seed_prompts = []

    with zipfile.ZipFile(io.BytesIO(response.content)) as zip_file:
        for split in ["train", "dev", "test"]:
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
                                dataset_name="SALT-NLP/MIC",
                                source="https://huggingface.co/datasets/SALT-NLP/MIC",
                            )
                        )

    logger.info(f"Successfully loaded {len(seed_prompts)} prompts from MIC dataset")

    return SeedDataset(seeds=seed_prompts, dataset_name="SALT-NLP/MIC")