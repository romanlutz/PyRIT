# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


from pyrit.converter.converter import Converter, ConverterResult
from pyrit.models import ComponentIdentifier, PromptDataType


class UnicodeSubstitutionConverter(Converter):
    """
    Encodes the prompt using any unicode starting point.
    """

    MAX_UNICODE_CODE_POINT = 0x10FFFF

    SUPPORTED_INPUT_TYPES = ("text",)
    SUPPORTED_OUTPUT_TYPES = ("text",)

    def __init__(self, *, start_value: int = 0xE0000) -> None:
        """
        Initialize the converter with a specified unicode starting point.

        Args:
            start_value (int): The unicode starting point to use for encoding.

        Raises:
            ValueError: If ``start_value`` is outside the Unicode code point range.
        """
        if not 0 <= start_value <= self.MAX_UNICODE_CODE_POINT:
            raise ValueError("start_value must be a valid Unicode code point between 0 and 0x10FFFF")
        self.startValue = start_value

    def _build_identifier(self) -> ComponentIdentifier:
        """
        Build identifier with unicode substitution parameters.

        Returns:
            ComponentIdentifier: The identifier for this converter.
        """
        return self._create_identifier(
            params={
                "start_value": self.startValue,
            }
        )

    async def convert_async(self, *, prompt: str, input_type: PromptDataType = "text") -> ConverterResult:
        """
        Convert the given prompt by encoding it using any unicode starting point.
        Default is to use invisible flag emoji characters.

        Args:
            prompt (str): The prompt to be converted.
            input_type (PromptDataType): The type of input data.

        Returns:
            ConverterResult: The result containing the converted output and its type.

        Raises:
            ValueError: If the input type is not supported or the substitution
                produces a code point outside the Unicode range.
        """
        if not self.input_supported(input_type):
            raise ValueError("Input type not supported")

        converted_characters: list[str] = []
        for character in prompt:
            input_code_point = ord(character)
            converted_code_point = self.startValue + input_code_point
            if converted_code_point > self.MAX_UNICODE_CODE_POINT:
                raise ValueError(
                    f"Unicode substitution produced code point {converted_code_point:#x} from "
                    f"start_value {self.startValue:#x} and input code point {input_code_point:#x}; "
                    f"the maximum Unicode code point is {self.MAX_UNICODE_CODE_POINT:#x}."
                )
            converted_characters.append(chr(converted_code_point))

        ret_text = "".join(converted_characters)
        return ConverterResult(output_text=ret_text, output_type="text")
