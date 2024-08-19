# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import abc
from collections import defaultdict
import logging
import uuid

from typing import Dict, List, Optional

from pyrit.memory import MemoryInterface, DuckDBMemory, MemoryExporter
from pyrit.models import PromptDataType, Identifier
from pyrit.models.prompt_request_piece import PromptRequestPiece
from pyrit.prompt_converter import PromptConverter
from pyrit.prompt_normalizer import NormalizerRequest, NormalizerRequestPiece

logger = logging.getLogger(__name__)


class Orchestrator(abc.ABC, Identifier):

    _memory: MemoryInterface

    def __init__(
        self,
        *,
        prompt_converters: Optional[list[PromptConverter]] = None,
        memory: Optional[MemoryInterface] = None,
        memory_labels: Optional[dict[str, str]] = None,
        verbose: bool = False,
    ):
        self._prompt_converters = prompt_converters if prompt_converters else []
        self._memory = memory or DuckDBMemory()
        self._verbose = verbose
        self._id = uuid.uuid4()

        self._global_memory_labels = memory_labels if memory_labels else {}

        if self._verbose:
            logging.basicConfig(level=logging.INFO)

    def __enter__(self):
        """Enter the runtime context related to this object."""
        return self  # You can return self or another object that should be used in the with-statement.

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the runtime context and perform any cleanup actions."""
        self.dispose_db_engine()

    def dispose_db_engine(self) -> None:
        """
        Dispose database engine to release database connections and resources.
        """
        self._memory.dispose_engine()

    def _create_normalizer_request(
        self, prompt_text: str, prompt_type: PromptDataType = "text", converters=None, metadata=None
    ):

        if converters is None:
            converters = self._prompt_converters

        request_piece = NormalizerRequestPiece(
            request_converters=converters,
            prompt_value=prompt_text,
            prompt_data_type=prompt_type,
            metadata=metadata,
        )

        request = NormalizerRequest([request_piece])
        return request

    def get_memory(self):
        """
        Retrieves the memory associated with this orchestrator.
        """
        return self._memory.get_prompt_request_piece_by_orchestrator_id(orchestrator_id=self._id)

    def get_score_memory(self):
        """
        Retrieves the scores of the PromptRequestPieces associated with this orchestrator.
        These exist if a scorer is provided to the orchestrator.
        """
        return self._memory.get_scores_by_orchestrator_id(orchestrator_id=self._id)
    
    def export_memory(self, *, file_path: str, export_type: str = "csv"):
        """Exports both conversation history and scores to a file.
        
        Args:
            file_path (str): Path to the file.
            export_type (str): The format for exporting data. Defaults to "csv".
        """
        # TODO add filtering mechanism to include/exclude only subset
        all_messages: List[PromptRequestPiece] = self.get_memory()

        all_messages.sort(key=lambda x: (x.conversation_id, x.sequence))

        # for message in all_messages:
        #     scores = self._memory.get_scores_by_prompt_ids(prompt_request_response_ids=[message.id])

        
        exporter = MemoryExporter()
        exporter.export_data(export_type=export_type, data=all_messages, file_path=file_path)

    def get_identifier(self) -> dict[str, str]:
        orchestrator_dict = {}
        orchestrator_dict["__type__"] = self.__class__.__name__
        orchestrator_dict["__module__"] = self.__class__.__module__
        orchestrator_dict["id"] = str(self._id)
        return orchestrator_dict
