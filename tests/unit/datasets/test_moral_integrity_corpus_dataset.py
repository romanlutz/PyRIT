# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import io
import json
import zipfile
from unittest.mock import MagicMock, patch

from pyrit.datasets.seed_datasets.remote.moral_integrity_corpus_dataset import _MICDataset


class TestMICDataset:

    def test_dataset_name(self):
        """Test that dataset_name property returns correct value."""
        dataset = _MICDataset()
        assert dataset.dataset_name == "moral_integrity_corpus"

    def test_init_default(self):
        """Test default initialization."""
        dataset = _MICDataset()
        assert dataset.source == "https://huggingface.co/datasets/SALT-NLP/MIC/resolve/main/MIC.zip"

    def _make_zip(self, rows: list[dict]) -> bytes:
        """Helper to create a fake zip file with JSONL content."""
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w") as zf:
            for split in ["train", "dev", "test"]:
                content = "\n".join(json.dumps(row) for row in rows)
                zf.writestr(f"MIC/{split}.jsonl", content)
        zip_buffer.seek(0)
        return zip_buffer.read()

    async def test_fetch_dataset_async(self):
        """Test successful dataset fetch with mocked network."""
        fake_rows = [
            {"Q": "Is lying okay?", "moral": "fairness"},
            {"Q": "Am I a bad boyfriend?", "moral": "loyalty"},
            {"Q": "Can murder be justified?", "moral": "care"},
        ]

        mock_response = MagicMock()
        mock_response.content = self._make_zip(fake_rows)
        mock_response.raise_for_status = MagicMock()

        with patch("requests.get", return_value=mock_response):
            dataset = _MICDataset()
            result = await dataset.fetch_dataset_async()

        assert len(result.seeds) == 3
        assert result.dataset_name == "moral_integrity_corpus"
        assert result.seeds[0].value == "Is lying okay?"
        assert result.seeds[0].data_type == "text"
        assert "fairness" in result.seeds[0].harm_categories

    async def test_fetch_dataset_deduplicates(self):
        """Test that duplicate questions are skipped."""
        fake_rows = [
            {"Q": "Is lying okay?", "moral": "fairness"},
            {"Q": "Is lying okay?", "moral": "loyalty"},
            {"Q": "Different question?", "moral": "care"},
        ]

        mock_response = MagicMock()
        mock_response.content = self._make_zip(fake_rows)
        mock_response.raise_for_status = MagicMock()

        with patch("requests.get", return_value=mock_response):
            dataset = _MICDataset()
            result = await dataset.fetch_dataset_async()

        assert len(result.seeds) == 2

    async def test_fetch_dataset_skips_empty_questions(self):
        """Test that empty questions are skipped."""
        fake_rows = [
            {"Q": "Valid question?", "moral": "care"},
            {"Q": "", "moral": "fairness"},
            {"Q": "   ", "moral": "loyalty"},
        ]

        mock_response = MagicMock()
        mock_response.content = self._make_zip(fake_rows)
        mock_response.raise_for_status = MagicMock()

        with patch("requests.get", return_value=mock_response):
            dataset = _MICDataset()
            result = await dataset.fetch_dataset_async()

        assert len(result.seeds) == 1

    async def test_fetch_dataset_empty_raises(self):
        """Test that empty dataset raises ValueError."""
        fake_rows = [
            {"Q": "", "moral": "care"},
        ]

        mock_response = MagicMock()
        mock_response.content = self._make_zip(fake_rows)
        mock_response.raise_for_status = MagicMock()

        with patch("requests.get", return_value=mock_response):
            dataset = _MICDataset()
            try:
                await dataset.fetch_dataset_async()
                assert False, "Should have raised ValueError"
            except ValueError as e:
                assert "empty" in str(e).lower()