# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import random
import string
from pyrit.models import PromptDataType
from pyrit.prompt_converter.prompt_converter import ConverterResult, PromptConverter


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
        bijection_type: str = "letter",
        fixed_size: int = 0,
        num_digits: int = 0,
    ) -> None:
        """
        Args:
            bijection_type: Type of bijection mapping. Currently supports "letter".
            fixed_size: Number of letters to keep unchanged (identity mapping).
            num_digits: Number of digits to include in the mapping.
        """
        super().__init__()
        self.bijection_type = bijection_type
        self.fixed_size = fixed_size
        self.num_digits = num_digits
        self.mapping = self._generate_mapping()
        self.inverse_mapping = {v: k for k, v in self.mapping.items()}

    def _generate_mapping(self) -> dict:
        """
        Generates a random bijection mapping of letters.
        """
        letters = list(string.ascii_lowercase)

        # these letters stay as themselves (identity)
        fixed_letters = letters[:self.fixed_size]

        # these letters get shuffled
        letters_to_shuffle = letters[self.fixed_size:]
        shuffled = letters_to_shuffle.copy()
        random.shuffle(shuffled)

        # combine fixed + shuffled into final mapping
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
            if char.lower() in self.mapping:
                # handle uppercase letters
                if char.isupper():
                    encoded += self.mapping[char.lower()].upper()
                else:
                    encoded += self.mapping[char]
            else:
                # spaces, punctuation stay the same
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
            if char.lower() in self.inverse_mapping:
                if char.isupper():
                    decoded += self.inverse_mapping[char.lower()].upper()
                else:
                    decoded += self.inverse_mapping[char]
            else:
                decoded += char

        return decoded