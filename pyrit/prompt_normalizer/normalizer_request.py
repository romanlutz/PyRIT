# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from collections.abc import Mapping
from dataclasses import dataclass

from pyrit.message_normalizer import MessageListNormalizer
from pyrit.models import Message
from pyrit.prompt_normalizer.converter_configuration import (
    ConverterConfiguration,
)
from pyrit.prompt_target.common.target_capabilities import CapabilityName
from pyrit.prompt_target.common.target_normalization_context import TargetNormalizationContext


@dataclass
class NormalizerRequest:
    """
    Represents a single request sent to normalizer.
    """

    message: Message
    request_converter_configurations: list[ConverterConfiguration]
    response_converter_configurations: list[ConverterConfiguration]
    conversation_id: str | None
    normalizer_overrides: dict[CapabilityName, MessageListNormalizer[Message]]
    target_normalization_context: TargetNormalizationContext | None

    def __init__(
        self,
        *,
        message: Message,
        request_converter_configurations: list[ConverterConfiguration] | None = None,
        response_converter_configurations: list[ConverterConfiguration] | None = None,
        conversation_id: str | None = None,
        normalizer_overrides: Mapping[CapabilityName, MessageListNormalizer[Message]] | None = None,
        target_normalization_context: TargetNormalizationContext | None = None,
    ) -> None:
        """
        Initialize a normalizer request.

        Args:
            message (Message): The message to be normalized.
            request_converter_configurations (list[ConverterConfiguration]): Configurations for converting
                the request. Defaults to an empty list.
            response_converter_configurations (list[ConverterConfiguration]): Configurations for converting
                the response. Defaults to an empty list.
            conversation_id (str | None): The ID of the conversation. Defaults to None.
            normalizer_overrides: Optional per-send target normalizer overrides.
            target_normalization_context: Optional explicit persisted-history boundary.
        """
        if response_converter_configurations is None:
            response_converter_configurations = []
        if request_converter_configurations is None:
            request_converter_configurations = []
        self.message = message
        self.request_converter_configurations = request_converter_configurations
        self.response_converter_configurations = response_converter_configurations
        self.conversation_id = conversation_id
        self.normalizer_overrides = dict(normalizer_overrides or {})
        self.target_normalization_context = target_normalization_context
