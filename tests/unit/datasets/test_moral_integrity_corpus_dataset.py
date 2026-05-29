# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from pyrit.datasets.seed_datasets.remote.fetch_moral_integrity_corpus_dataset import _MICDataset


class TestMICDataset:

    def test_dataset_name(self):
        """Test that dataset_name property returns correct value."""
        dataset = _MICDataset()
        assert dataset.dataset_name == "moral_integrity_corpus"

    def test_init_default(self):
        """Test default initialization."""
        dataset = _MICDataset()
        assert dataset.token is None

    def test_init_with_token(self):
        """Test initialization with token."""
        dataset = _MICDataset(token="test_token")
        assert dataset.token == "test_token"

    def test_init_token_from_env(self, monkeypatch):
        """Test token is read from environment variable."""
        monkeypatch.setenv("HUGGINGFACE_TOKEN", "env_token")
        dataset = _MICDataset()
        assert dataset.token == "env_token"

    @pytest.mark.asyncio
    async def test_fetch_dataset_async(self):
        """Test successful dataset fetch with mocked network."""
        import io
        import json
        import zipfile

        # Create fake JSONL data
        fake_rows = [
            {"Q": "Is lying okay?", "moral": "fairness"},
            {"Q": "Am I a bad boyfriend?", "moral": "loyalty"},
            {"Q": "Can murder be justified?", "moral": "care"},
        ]

        # Create fake zip file in memory
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w") as zf:
            for split in ["train", "dev", "test"]:
                content = "\n".join(json.dumps(row) for row in fake_rows)
                zf.writestr(f"MIC/{split}.jsonl", content)
        zip_buffer.seek(0)

        # Mock the requests.get call
        mock_response = MagicMock()
        mock_response.content = zip_buffer.read()
        mock_response.raise_for_status = MagicMock()

        with patch("requests.get", return_value=mock_response):
            dataset = _MICDataset()
            result = await dataset.fetch_dataset_async()

        # 3 rows x 3 splits = 9 prompts
        assert len(result.seeds) == 9
        assert result.dataset_name == "moral_integrity_corpus"
        assert result.seeds[0].value == "Is lying okay?"
        assert result.seeds[0].data_type == "text"

    @pytest.mark.asyncio
    async def test_fetch_dataset_skips_empty_questions(self):
        """Test that empty questions are skipped."""
        import io
        import json
        import zipfile

        fake_rows = [
            {"Q": "Valid question?", "moral": "care"},
            {"Q": "", "moral": "fairness"},  # empty - should be skipped
            {"Q": "   ", "moral": "loyalty"},  # whitespace - should be skipped
        ]

        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w") as zf:
            for split in ["train", "dev", "test"]:
                content = "\n".join(json.dumps(row) for row in fake_rows)
                zf.writestr(f"MIC/{split}.jsonl", content)
        zip_buffer.seek(0)

        mock_response = MagicMock()
        mock_response.content = zip_buffer.read()
        mock_response.raise_for_status = MagicMock()

        with patch("requests.get", return_value=mock_response):
            dataset = _MICDataset()
            result = await dataset.fetch_dataset_async()

        # Only 1 valid question x 3 splits = 3 prompts
        assert len(result.seeds) == 3