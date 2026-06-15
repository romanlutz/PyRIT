# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import random
import string
from enum import StrEnum
from pyrit.models import PromptDataType
from pyrit.prompt_converter.prompt_converter import ConverterResult, PromptConverter


class BijectionType(StrEnum):
    LETTER = "letter"


class BijectionConverter(PromptConverter):
    """
    Converts a prompt using a random bijection (one-to-one) character mapping.
    This can be used to encode prompts to bypass safety filters.
    Based on the bijection attack from arXiv:2410.01294 (Haize Labs).
    """

    SUPPORTED_INPUT_TYPES = ("text",)
    SUPPORTED_OUTPUT_TYPES = ("text",)

    def __init__(
        self,
        *,
        bijection_type: BijectionType = BijectionType.LETTER,
        fixed_size: int = 0,
    ) -> None:
        """
        Args:
            bijection_type: Type of bijection mapping. Currently supports "letter".
            fixed_size: Number of letters to keep unchanged (identity mapping).
        """
        super().__init__()
        self._bijection_type = BijectionType(bijection_type)
        self._fixed_size = fixed_size
        self._mapping = self._generate_mapping()
        self._inverse_mapping = {v: k for k, v in self._mapping.items()}

    @property
    def mapping(self) -> dict:
        return self._mapping

    @property
    def inverse_mapping(self) -> dict:
        return self._inverse_mapping

    @property
    def fixed_size(self) -> int:
        return self._fixed_size

    def _build_identifier(self) -> dict:
        return self._create_identifier(params={
            "bijection_type": self._bijection_type,
            "fixed_size": self._fixed_size,
            "mapping": str(self._mapping),
        })

    def _generate_mapping(self) -> dict:
        """
        Generates a random bijection mapping of letters.
        """
        letters = list(string.ascii_lowercase)

        fixed_letters = letters[:self._fixed_size]
        letters_to_shuffle = letters[self._fixed_size:]
        shuffled = letters_to_shuffle.copy()
        random.shuffle(shuffled)

        mapping = {}
        for letter in fixed_letters:
            mapping[letter] = letter
        for original, replacement in zip(letters_to_shuffle, shuffled):
            mapping[original] = replacement

        return mapping

    async def convert_async(
        self,
        *,
        prompt: str,
        input_type: PromptDataType = "text"
    ) -> ConverterResult:
        """
        Encodes the prompt using the bijection mapping.

        Args:
            prompt: The prompt to be converted.
            input_type: Type of data.

        Returns:
            The encoded prompt using the secret mapping.

        Raises:
            ValueError: If the input type is not supported.
        """
        if not self.input_supported(input_type):
            raise ValueError("Input type not supported")

        encoded = ""
        for char in prompt:
            if char.lower() in self._mapping:
                if char.isupper():
                    encoded += self._mapping[char.lower()].upper()
                else:
                    encoded += self._mapping[char]
            else:
                encoded += char

        return ConverterResult(output_text=encoded, output_type="text")

    def decode(self, encoded_text: str) -> str:
        """
        Decodes an encoded response back to plain English using inverse mapping.

        Args:
            encoded_text: The encoded text to decode.

        Returns:
            The decoded plain English text.
        """
        decoded = ""
        for char in encoded_text:
            if char.lower() in self._inverse_mapping:
                if char.isupper():
                    decoded += self._inverse_mapping[char.lower()].upper()
                else:
                    decoded += self._inverse_mapping[char]
            else:
                decoded += char

        return decoded