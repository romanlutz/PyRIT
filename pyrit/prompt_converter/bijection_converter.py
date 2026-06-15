# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import abc
import random
import string
from pyrit.models import PromptDataType
from pyrit.prompt_converter.prompt_converter import ConverterResult, PromptConverter


class BijectionConverter(PromptConverter, abc.ABC):
    """
    Abstract base class for bijection converters.
    Converts a prompt using a one-to-one character mapping to bypass safety filters.
    Based on the bijection attack from arXiv:2410.01294 (Haize Labs).
    """

    SUPPORTED_INPUT_TYPES = ("text",)
    SUPPORTED_OUTPUT_TYPES = ("text",)

    def __init__(
        self,
        *,
        mapping: dict[str, str] | None = None,
        seed: int | None = None,
    ) -> None:
        """
        Args:
            mapping: Optional explicit mapping dict. If provided, used directly.
            seed: Optional random seed for reproducibility.
        """
        super().__init__()
        rng = random.Random(seed)
        self._mapping = mapping if mapping is not None else self._generate_mapping(rng)
        self._inverse_mapping = {v: k for k, v in self._mapping.items()}

    @abc.abstractmethod
    def _generate_mapping(self, rng: random.Random) -> dict[str, str]:
        """Generate the bijection mapping."""
        ...

    @property
    def mapping(self) -> dict[str, str]:
        return self._mapping

    @property
    def inverse_mapping(self) -> dict[str, str]:
        return self._inverse_mapping

    def _build_identifier(self) -> dict:
        return self._create_identifier(params={
            "mapping": str(self._mapping),
        })

    async def convert_async(
        self,
        *,
        prompt: str,
        input_type: PromptDataType = "text"
    ) -> ConverterResult:
        """
        Encodes the prompt using the bijection mapping.
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
        Decodes an encoded response back to plain text using inverse mapping.
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


class LetterBijectionConverter(BijectionConverter):
    """
    Bijection converter that maps letters to other letters.
    """

    def __init__(
        self,
        *,
        fixed_size: int = 0,
        mapping: dict[str, str] | None = None,
        seed: int | None = None,
    ) -> None:
        self._fixed_size = fixed_size
        super().__init__(mapping=mapping, seed=seed)

    @property
    def fixed_size(self) -> int:
        return self._fixed_size

    def _generate_mapping(self, rng: random.Random) -> dict[str, str]:
        letters = list(string.ascii_lowercase)
        fixed_letters = letters[:self._fixed_size]
        letters_to_shuffle = letters[self._fixed_size:]
        shuffled = letters_to_shuffle.copy()
        rng.shuffle(shuffled)

        mapping = {}
        for letter in fixed_letters:
            mapping[letter] = letter
        for original, replacement in zip(letters_to_shuffle, shuffled):
            mapping[original] = replacement

        return mapping

    def _build_identifier(self) -> dict:
        return self._create_identifier(params={
            "fixed_size": self._fixed_size,
            "mapping": str(self._mapping),
        })


class DigitBijectionConverter(BijectionConverter):
    """
    Bijection converter that maps digits to other digits.
    """

    def __init__(
        self,
        *,
        num_digits: int = 10,
        mapping: dict[str, str] | None = None,
        seed: int | None = None,
    ) -> None:
        self._num_digits = min(num_digits, 10)
        super().__init__(mapping=mapping, seed=seed)

    @property
    def num_digits(self) -> int:
        return self._num_digits

    def _generate_mapping(self, rng: random.Random) -> dict[str, str]:
        digits = list(string.digits[:self._num_digits])
        shuffled = digits.copy()
        rng.shuffle(shuffled)

        mapping = {}
        for original, replacement in zip(digits, shuffled):
            mapping[original] = replacement

        return mapping

    def _build_identifier(self) -> dict:
        return self._create_identifier(params={
            "num_digits": self._num_digits,
            "mapping": str(self._mapping),
        })
    