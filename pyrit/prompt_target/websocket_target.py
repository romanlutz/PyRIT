# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
import logging
import os
from typing import Optional
import uuid
import websockets

from pyrit.common import default_values
from pyrit.exceptions import EmptyResponseException
from pyrit.models import PromptRequestResponse, construct_response_from_request
from pyrit.prompt_target import PromptChatTarget, limit_requests_per_minute

logger = logging.getLogger(__name__)


class WebsocketTarget(PromptChatTarget):

    websocket_uri_environment_variable: str = "WEBSOCKET_URI"
    websocket_access_token_environment_variable: str = "WEBSOCKET_ACCESS_TOKEN"

    def __init__(
        self,
        *,
        websocket_uri: str = None,
        access_token: str = None,
        max_requests_per_minute: Optional[int] = None,
    ) -> None:
        """
        Initializes an instance of the WebsocketTarget class.

        Args:
            websocket_uri (str, Optional): The URI of the websocket server.
            access_token (str, Optional): The access token for the websocket server.
            max_requests_per_minute (int, Optional): Number of requests the target can handle per
                minute before hitting a rate limit. The number of requests sent to the target
                will be capped at the value provided.
        """
        PromptChatTarget.__init__(self, max_requests_per_minute=max_requests_per_minute)

        self._initialize_vars(websocket_uri=websocket_uri, access_token=access_token)

    def _set_env_configuration_vars(
        self, websocket_uri_environment_variable: str = None, access_token_environment_variable: str = None
    ) -> None:
        """
        Sets the environment configuration variable names from which to pull the websocket uri and the access token
        to access the system via websocket. Use this function to set the environment variable names to
        however they are named in the .env file and pull the corresponding websock3et uri and access token.
        This is the recommended way to pass in a uri and token to access the system.
        Defaults to "WEBSOCKET_URI" and "WEBSOCKET_ACCESS_TOKEN".

        Args:
            websocket_uri_environment_variable (str): The environment variable name for the websocket uri.
            access_token_environment_variable (str): The environment variable name for the access token.

        Returns:
            None
        """
        self.websocket_uri_environment_variable = websocket_uri_environment_variable or "WEBSOCKET_URI"
        self.access_token_environment_variable = access_token_environment_variable or "WEBSOCKET_ACCESS_TOKEN"
        self._initialize_vars()

    def _initialize_vars(self, websocket_uri: str = None, access_token: str = None) -> None:
        """
        Sets the endpoint and key for accessing the system. Use this function to manually
        pass in your own websocket uri and access token. Defaults to the values in the .env file for the variables
        stored in self.websocket_uri_environment_variable and self.access_token_environment_variable (which default to
        "WEBSOCKET_URI" and "WEBSOCKET_ACCESS_TOKEN" respectively). It is recommended to set these variables
        in the .env file and call _set_env_configuration_vars rather than passing the uri and key directly to
        this function or the target constructor.

        Args:
            websocket_uri (str): The websocket uri for the system.
            access token (str): The access token for accessing the system.

        Returns:
            None
        """
        self._websocket_uri = default_values.get_required_value(
            env_var_name=self.websocket_uri_environment_variable, passed_value=websocket_uri
        )
        self._access_token = default_values.get_required_value(
            env_var_name=self.websocket_access_token_environment_variable, passed_value=access_token
        )

    @limit_requests_per_minute
    async def send_prompt_async(self, *, prompt_request: PromptRequestResponse) -> PromptRequestResponse:

        self._validate_request(prompt_request=prompt_request)
        request = prompt_request.request_pieces[0]

        messages = self._memory.get_chat_messages_with_conversation_id(conversation_id=request.conversation_id)

        messages.append(request.to_chat_message())

        logger.info(f"Sending the following prompt to the prompt target: {request}")

        uri = f"{self._websocket_uri}?&access_token={self._access_token}"

        async with websockets.connect(uri, max_size=None) as websocket:
            # handshake message
            await websocket.send_message("{\"protocol\":\"json\",\"version\":1}\u001e")
            received_message = await websocket.recv()
            message = json.dumps({
                "message": {
                    "text": request.converted_value,
                }
            }) + "\u001e"

            await websocket.send_message(message)

            # Filter status messages and concatenate the required messages into a single response string
            required_message = ""
            is_part_of_required_message = False
            while True:
                received_message = await websocket.recv()

                if received_message.startswith("{\"type\":2"):
                    is_part_of_required_message = True
                elif received_message.startswith("{\"type\":3"):
                    is_part_of_required_message = False
                    break

                if is_part_of_required_message:
                    required_message += received_message

            await websocket.close()

            if not required_message:
                raise EmptyResponseException(message="The chat returned an empty response.")
            
            parsed_response = self.parse_api_response_to_get_bot_response(required_message)

            response_entry = construct_response_from_request(request=request, response_text_pieces=[parsed_response])

        logger.info(
            "Received the following response from the prompt target"
            + f"{response_entry.request_pieces[0].converted_value}"
        )
        return response_entry

    def parse_api_response_to_get_bot_response(self, response):
        bot_response = ""
        if not response:
            return bot_response

        try:
            messages = json.loads(response).get('item', {}).get('messages', [])

            for message in messages:
                bot_response += message.get('text', '')
                bot_response += "\n"
        except Exception as ex:
            print(f"Error getting the response from websocket:\n{ex}")

        return bot_response

    def _validate_request(self, *, prompt_request: PromptRequestResponse) -> None:
        if len(prompt_request.request_pieces) != 1:
            raise ValueError("This target only supports a single prompt request piece.")

        if prompt_request.request_pieces[0].converted_value_data_type != "text":
            raise ValueError("This target only supports text prompt input.")
