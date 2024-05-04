# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import logging
from pathlib import Path
from typing import Optional
import uuid
from base64 import b64encode
import hashlib
import hmac
import os

from pyrit.common import net_utility
from pyrit.memory import MemoryInterface
from pyrit.memory.duckdb_memory import DuckDBMemory
from pyrit.models import PromptRequestResponse
from pyrit.prompt_target import PromptTarget

logger = logging.getLogger(__name__)


class CustomImageToTextTarget(PromptTarget):

    CUSTOM_IMAGE_TO_TEXT_SECRET_ENVIRONMENT_VARIABLE: str = "CUSTOM_IMAGE_TO_TEXT_SECRET"
    CUSTOM_IMAGE_TO_TEXT_ENDPOINT_URL_ENVIRONMENT_VARIABLE = "CUSTOM_IMAGE_TO_TEXT_ENDPOINT_URL"

    def __init__(
        self,
        *,
        memory: Optional[MemoryInterface] = None,
    ):
        super().__init__(memory=memory or DuckDBMemory())
        self._endpoint_url = os.environ[CustomImageToTextTarget.CUSTOM_IMAGE_TO_TEXT_ENDPOINT_URL_ENVIRONMENT_VARIABLE]
        self._encoded_secret = os.environ[CustomImageToTextTarget.CUSTOM_IMAGE_TO_TEXT_SECRET_ENVIRONMENT_VARIABLE]

    def send_prompt(self, *, prompt_request: PromptRequestResponse) -> PromptRequestResponse:

        self._validate_request(prompt_request=prompt_request)
        request = prompt_request.request_pieces[0]

        messages = self._memory.get_chat_messages_with_conversation_id(conversation_id=request.conversation_id)
        messages.append(request.to_chat_message())

        self._memory.add_request_response_to_memory(request=prompt_request)

        logger.info(f"Sending the following prompt to the prompt target: {request}")

        resp_text = self._complete_chat(
            messages=messages,
            temperature=self._temperature,
            top_p=self._top_p,
            repetition_penalty=self._repetition_penalty,
        )

        if not resp_text:
            raise ValueError("The chat returned an empty response.")

        logger.info(f'Received the following response from the prompt target "{resp_text}"')

        response_entry = self._memory.add_response_entries_to_memory(request=request, response_text_pieces=[resp_text])

        return response_entry

    async def send_prompt_async(self, *, prompt_request: PromptRequestResponse) -> PromptRequestResponse:

        self._validate_request(prompt_request=prompt_request)
        request = prompt_request.request_pieces[0]

        
        self._memory.add_request_response_to_memory(request=prompt_request)

        logger.info(f"Sending the following prompt to the prompt target: {request}")

        resp_text = await self._send_async(
            data_file=request.converted_value,
        )

        if not resp_text:
            raise ValueError("The endpoint returned an empty response.")

        logger.info(f'Received the following response from the prompt target "{resp_text}"')

        response_entry = self._memory.add_response_entries_to_memory(request=request, response_text_pieces=[resp_text])

        return response_entry

    async def _send_async(
        self,
        *,
        data_file: Path,
        lang: str = "en",
    ) -> str:
        url = f"{self._endpoint_url}?intent=RichDescriptionVNext&lang={lang}"
        with open(data_file, 'rb') as file:
            data = file.read()

        # Construct signature
        checksum = b64encode(hashlib.md5(data).digest()).decode('utf-8')
        input_params = f"appName=RedTeam&appVersion=1.0&checksum={checksum}&intent=RichDescriptionVNext&lang={lang}&os="
        secret = bytearray([int(s) for s in self._encoded_secret.split(",")])
        signer = hmac.new(secret, input_params.encode('utf-8'), hashlib.sha256)
        signature = b64encode(signer.digest()).decode('utf-8')
        headers: dict = {
            "user-agent": "RedTeam/1.0",
            "signature": signature,
            "content-type": "image/jpeg",
            "request-id": str(uuid.uuid4()).replace('-', '')
        }

        response = await net_utility.make_request_and_raise_if_error_async(
            endpoint_uri=url, method="POST", request_body=data, headers=headers, post_type="data"
        )
        return response.json()["RichDescriptionVNext"]["Content"]


    def _validate_request(self, *, prompt_request: PromptRequestResponse) -> None:
        if len(prompt_request.request_pieces) != 1:
            raise ValueError("This target only supports a single prompt request piece.")

        if prompt_request.request_pieces[0].converted_value_data_type != "image_path":
            raise ValueError("This target only supports text prompt input.")
