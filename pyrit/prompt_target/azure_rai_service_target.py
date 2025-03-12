# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
from typing import Optional
import uuid

from pyrit.models import (
    PromptRequestResponse,
    construct_response_from_request,
)
from pyrit.prompt_target import PromptChatTarget

logger = logging.getLogger(__name__)


class AzureRAIServiceTarget(PromptChatTarget):
    def __init__(
        self,
        *,
        client: GeneratedRAIClient,
        api_version: Optional[str] = None,
        model: Optional[str] = None,
    ) -> None:
        PromptChatTarget.__init__(self)
        self._client = client
        self._api_version = api_version
        self._model = model

    async def send_prompt_async(self, *, prompt_request: PromptRequestResponse) -> PromptRequestResponse:

        self._validate_request(prompt_request=prompt_request)
        request = prompt_request.request_pieces[0]

        messages = list(self._memory.get_chat_messages_with_conversation_id(conversation_id=request.conversation_id))

        messages.append(request.to_chat_message())

        logger.info(f"Sending the following prompt to the prompt target: {request}")

        # Extracting template key and objective from the memory labels is a hacky workaround.
        # There may be a cleaner way to do this in the future.
        memory_labels = prompt_request.request_pieces[0].labels
        template_key = memory_labels.get("templatekey")
        if not template_key:
            raise ValueError("Template key is required to send a prompt to the Azure RAI service.")
        objective = memory_labels.get("objective")
        if not objective:
            raise ValueError("Objective is required to send a prompt to the Azure RAI service.")
        template_parameters = {"objective": objective}
        # special template param for TAP
        if "desired_prefix" in memory_labels:
            template_parameters["desired_prefix"] = memory_labels["desired_prefix"]
        # special template param for Crescendo
        if "max_turns" in memory_labels:
            template_parameters["max_turns"] = memory_labels["max_turns"]
        prompt = prompt_request.request_pieces[0].converted_value
        payload = {"prompt": prompt}

        token = self.token_manager.get_token()
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "User-Agent": USER_AGENT,
            "X-CV": str(uuid.uuid4()),
            "X-ModelType": self._model or "",
        }

        params = {}
        if self._api_version:
            params["api-version"] = self._api_version
        
        simulation_dto = SimulationRequestDTO(
            url=self._simulation_submit_endpoint,
            headers=headers,
            payload=payload,
            params=params,
            templatekey=template_key,
            template_parameters=template_parameters
        )
        # TODO probably should do retries on 429 or similar
        session = self._create_async_client()
        async with session:
            response = await session.post(url=self._simulation_submit_endpoint, headers=headers, json=simulation_dto.to_json())
        # TODO: is it possible to get 202s here? If so, we should handle them
        if response.status_code == 200:
            response = response.json()
            if "choices" in response and len(response["choices"]) > 0 and "message" in response["choices"][0] and "content" in response["choices"][0]["message"]:
                response = response["choices"][0]["message"]["content"]
            response_entry = construct_response_from_request(request=prompt, response_text_pieces=[response])
            logger.info(
                f"Received the following response from the prompt target {response}"
            )
            return response_entry
        msg = (
            "Azure safety evaluation service is not available in your current region, "
            + "please go to https://aka.ms/azureaistudiosafetyeval to see which regions are supported"
        )
        raise EvaluationException(
            message=msg,
            internal_message=msg,
            target=ErrorTarget.RAI_CLIENT,
            category=ErrorCategory.UNKNOWN,
            blame=ErrorBlame.USER_ERROR,
        )

    def _validate_request(self, *, prompt_request: PromptRequestResponse) -> None:
        if len(prompt_request.request_pieces) != 1:
            raise ValueError("This target only supports a single prompt request piece.")

        if prompt_request.request_pieces[0].converted_value_data_type != "text":
            raise ValueError("This target only supports text prompt input.")
